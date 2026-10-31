"""Run formal A–D ablation using audited current B/D weights and a newly trained C.

Training and selection use train/validation only. All four checkpoints are frozen
before validation/test scoring; reports never require a favorable performance ordering.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import torch
from threadpoolctl import threadpool_limits

from src.algorithms.icr_mmse.checkpoint import load_model, validate_dataset
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.fingerprints import DataIdentity, source_fingerprint
from src.evaluation.icr_ablation import GROUPS, SNR, save_report, summarize, validate_report
from src.evaluation.icr_mmse import evaluate
from src.evaluation.icr_unified import audit_study
from src.evaluation.result_cache import FileLock
from src.training.icr_mmse import TRAINING_CODE, train
from src.training.icr_workflow import DEFAULT_CONFIG, resolve_config

LOG = logging.getLogger(__name__)


def linear_options(resolved, group):
    """The linear detector has no A fitting; C fits its own iterative corrector."""
    if group not in ("A", "C"):
        raise ValueError("Linear training is only defined for A/C")
    model, controls = deepcopy(resolved["model"]), deepcopy(resolved["training"])
    model.update(detector_mode="lmmse", correction_mode="fixed" if group == "A" else "iterative")
    controls.pop("warmup_checkpoint", None)
    controls["stage_epochs"] = [0, 0, 0] if group == "A" else [0, *controls["stage_epochs"][1:]]
    controls["neural_learning_rate_scale"] = 1.
    return dict(model_config=model, **controls)


def preflight(config_path, trained_root):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("input_view") != "compensated" or set(config["datasets"]) != {"NOF1", "NCS1"}:
        raise ValueError("Formal ablation requires both accepted compensated datasets")
    checkpoints = {f"{branch}_{ch.lower()}": trained_root/branch/ch.lower()/"icr_mmse.pt"
                   for ch in config["datasets"] for branch in ("fixed", "joint")}
    audit = audit_study(checkpoints)
    guards = {}
    for channel, directory in config["datasets"].items():
        directory = Path(directory)
        guard = DataIdentity({name: directory/name for name in
                              ("train.h5", "validation.h5", "test.h5", "model.h5")})
        guards[channel] = guard
        for split, frames in (("train", 525), ("validation", 75), ("test", 600)):
            with CompactDataset(directory/(split+".h5"), input_view="compensated") as ds:
                if (ds.channel != channel or ds.split != split or ds.frame_count != frames
                        or ds.snr_db.tolist() != SNR or len(ds._indices) != 512 or ds._config.get("smoke")):
                    raise ValueError(f"{channel}/{split}: not the accepted complete dataset")
                for branch in ("fixed", "joint"):
                    path = checkpoints[f"{branch}_{channel.lower()}"]
                    _, payload = load_model(path)
                    validate_dataset(payload, ds)
        for branch in ("fixed", "joint"):
            _, payload = load_model(checkpoints[f"{branch}_{channel.lower()}"])
            resolved = resolve_config(config_path, channel=channel, stage=branch, output=trained_root)
            if payload["model_config"] != ICRMMSEnet(**resolved["model"]).config:
                raise ValueError(f"{channel}/{branch}: model differs from current config")
            identity = payload["identity"]
            if identity["code"] != source_fingerprint(TRAINING_CODE):
                raise ValueError(f"{channel}/{branch}: training source differs")
            for name in ("train.h5", "validation.h5", "model.h5"):
                if identity["data"][name] != guard.files[name]:
                    raise ValueError(f"{channel}/{branch}: training data differs")
            for name, expected in resolved["training"].items():
                if name == "warmup_checkpoint":
                    expected = str(Path(expected).resolve())
                default = "all" if name == "detector_parameters" else 1. if name == "neural_learning_rate_scale" else None
                if identity["controls"].get(name, default) != expected:
                    raise ValueError(f"{channel}/{branch}: training setting differs: {name}")
    return config, checkpoints, audit, guards


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args):
    output, trained_root = args.output.resolve(), args.trained_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = dict(status="running", phase="preflight", complete=False, pid=os.getpid(),
                 started_at=datetime.now(timezone.utc).isoformat())

    def phase(name):
        state.update(phase=name, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(output/"progress.json", state)
        LOG.info("Starting %s", name)

    with ExitStack() as stack:
        stack.enter_context(FileLock(output/".ablation.lock"))
        if (output/"manifest.json").exists() and not args.resume:
            raise FileExistsError("Experiment exists; use --resume or a new output directory")
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("Formal experiments require a clean committed checkout")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if args.device == "cuda":
            uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid",
                                             "--format=csv,noheader"], text=True).strip()
            stack.enter_context(FileLock(Path("outputs/runs/_gpu_leases")/(uuid+".lock")))
            pids = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                                            "--format=csv,noheader"], text=True).strip()
            if pids:
                raise RuntimeError(f"GPU has active compute processes: {pids}")
        phase("preflight")
        try:
            if shutil.disk_usage(output).free < 5*2**30:
                raise RuntimeError("Formal ablation requires at least 5 GiB free disk")
            config, existing, audit, guards = preflight(args.config, trained_root)
            identity = dict(commit=commit, config=config, trained_root=str(trained_root),
                checkpoints={name: dict(path=str(path), sha256=file_sha256(path)) for name, path in existing.items()},
                data={ch: guard.files for ch, guard in guards.items()}, device=args.device, threads=args.threads)
            if (output/"manifest.json").exists():
                previous = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
                if previous["identity"] != identity:
                    raise ValueError("Resume requires original code, config, data, weights and execution settings")
            write_json(output/"manifest.json", dict(identity=identity, groups=GROUPS,
                c_training="Independent LMMSE-feedback corrector; same 8 rounds, bandwidth and B/C budgets as D",
                b_d_reuse="Current complete weights; matching training source/data/config and shared warmup audited",
                selection="Train/validation only; all checkpoints frozen before test; no required metric ordering"))
            write_json(output/"training_config.json", config)
            write_json(output/"detector_learning_audit.json", audit)
            backend = DenseBackend(args.device)
            all_checkpoints = {}
            for channel, directory in config["datasets"].items():
                checkpoints = {"B": existing[f"fixed_{channel.lower()}"],
                               "D": existing[f"joint_{channel.lower()}"]}
                resolved = resolve_config(args.config, channel=channel, stage="joint", output=output)
                for group in ("A", "C"):
                    phase(f"{channel.lower()}_{group}_training")
                    checkpoints[group] = train(directory, output/"models"/channel.lower()/group, backend,
                        resume=args.resume, **linear_options(resolved, group))
                    _, payload = load_model(checkpoints[group])
                    if not payload.get("complete") or payload.get("smoke"):
                        raise ValueError("Linear branch did not finish formal training")
                    release_memory()
                all_checkpoints[channel] = {g: dict(path=str(checkpoints[g].resolve()),
                    sha256=file_sha256(checkpoints[g])) for g in GROUPS}
            frozen = output/"frozen_checkpoints.json"
            if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != all_checkpoints:
                raise ValueError("A frozen ablation checkpoint changed")
            write_json(frozen, all_checkpoints)
            summaries = {}
            for split in ("validation", "test"):
                for channel, directory in config["datasets"].items():
                    reports = {}
                    for group in GROUPS:
                        phase(f"{channel.lower()}_{group}_{split}")
                        pinned = all_checkpoints[channel][group]
                        checkpoint = Path(pinned["path"])
                        if file_sha256(checkpoint) != pinned["sha256"]:
                            raise ValueError("Checkpoint changed after freezing")
                        run_directory = output/split/channel.lower()/group
                        report_path = run_directory/"metrics.json"
                        if args.resume and report_path.exists():
                            report = json.loads(report_path.read_text(encoding="utf-8"))
                            names = (split+".h5", "model.h5", "train.h5", "validation.h5")
                            expected_data = {name: guards[channel].files[name] for name in names}
                            if (report["checkpoint_sha256"] != pinned["sha256"]
                                    or report["data_identity"] != expected_data):
                                raise ValueError("Existing ablation evaluation identity differs")
                        else:
                            report = evaluate(Path(directory)/(split+".h5"), checkpoint, run_directory, backend,
                                require_test=split == "test", include_baselines=False,
                                batch_size=config["evaluation"]["batch_size"],
                                bootstrap_samples=config["evaluation"]["bootstrap_samples"])
                        validate_report(report, channel, group, split)
                        reports[group] = report
                        release_memory()
                    phase(f"{channel.lower()}_{split}_report")
                    summary = summarize(reports)
                    save_report(summary, output/split/channel.lower())
                    summaries[f"{channel}_{split}"] = summary
                    write_json(output/"results_digest.json", dict(complete=False, channels=summaries))
            for guard in guards.values():
                guard.assert_unchanged()
            write_json(output/"results_digest.json", dict(complete=True, channels=summaries))
            phase("package_reports")
            with zipfile.ZipFile(output/"reports.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output.rglob("*")):
                    if (path.is_file() and path != output/"progress.json"
                            and path.suffix in (".json", ".csv", ".png", ".svg")):
                        archive.write(path, path.relative_to(output))
            state.update(status="completed", phase="finished", complete=True)
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            LOG.exception("Formal ablation failed")
            raise
        finally:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(output/"progress.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trained-root", type=Path, required=True, help="Completed current shared-warmup B/D training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        run(args)


if __name__ == "__main__":
    main()
