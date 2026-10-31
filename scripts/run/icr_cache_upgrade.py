"""Rebind verified performance-only cache changes without discarding training state."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import torch

from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE, atomic_save
from src.common.experiment import file_sha256, write_json
from src.evaluation.fingerprints import source_fingerprint
from src.evaluation.result_cache import FileLock
from src.training.icr_mmse import TRAINING_CODE


def upgrade(run, report_path):
    run, report_path = Path(run).resolve(), Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    before, after = report["training_code_before"], report["training_code_after"]
    if (not report.get("passed") or report.get("input_view") != "compensated"
            or sorted(k for k in before if before[k] != after[k]) !=
                ["src.data.icr_dense_training", "src.data.icr_training"]
            or after != source_fingerprint(TRAINING_CODE)
            or report["inference_code"] != source_fingerprint(INFERENCE_CODE)
            or set(report["channels"]) != {"NOF1", "NCS1"}):
        raise ValueError("A successful paired numerical check of only the cache pipeline is required")
    candidates = [p for p in run.rglob("*.pt") if "_control" not in p.relative_to(run).parts]
    plans = []
    for path in candidates:
        original_hash = file_sha256(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if file_sha256(path) != original_hash:
            raise RuntimeError(f"Checkpoint changed during validation: {path}")
        identity = state.get("identity", {})
        if identity.get("code") == after:
            continue
        if (identity.get("code") != before or identity.get("controls", {}).get("input_view") != "compensated"
                or identity["controls"].get("operator_backend") != "dense_device"
                or state.get("input_view", "compensated") != "compensated"):
            raise ValueError(f"Checkpoint is not from the verified compensated baseline: {path}")
        plans.append((path, state, original_hash))
    if not plans:
        return dict(complete=True, checkpoints=[], report=str(report_path))
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    backup = run/"_control"/("cache_upgrade_"+timestamp)
    rows = []
    for path, state, original_hash in plans:
        with FileLock(path.parent/".training.lock"):
            if file_sha256(path) != original_hash:
                raise RuntimeError(f"Checkpoint changed before migration: {path}")
            destination = backup/path.relative_to(run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            provenance = dict(reason="exact compensated reference cache and shared transfer",
                report=str(report_path), report_sha256=file_sha256(report_path),
                backup=str(destination), previous_sha256=file_sha256(destination),
                previous_code=before, replacement_code=after)
            state["identity"] = dict(state["identity"], code=after)
            state.setdefault("performance_migrations", []).append(provenance)
            atomic_save(path, state)
            rows.append(dict(path=str(path), backup=str(destination),
                stage_index=state.get("stage_index"), epoch=state.get("epoch"),
                history_length=len(state.get("history", [])), sha256=file_sha256(path)))
    result = dict(complete=True, checkpoints=rows, report=str(report_path),
        preserved="all model tensors, optimizer state, RNG state, selected weights, epochs and data/control identities")
    write_json(backup/"migration.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verified-report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.run, args.verified_report)))


if __name__ == "__main__":
    main()
