"""Sequential frozen evaluation of both channels and CSI-correction settings."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil

import torch
from threadpoolctl import threadpool_limits

from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.evaluation.fingerprints import source_fingerprint
from src.evaluation.icr_budgeted import compare, EVALUATION_CODE
from src.evaluation.icr_unified import audit_study


LEGACY_ROOT = Path("outputs/runs/icr_budgeted_20261031")
ROOT = Path("outputs/runs/icr_budgeted_unified_20261031")
CASES = (
    ("fixed_ncs1", "ncs1", "ncs1_compact_20261026_nmse", "fixed_final", "A", "B"),
    ("fixed_nof1", "nof1", "nof1_compact_20261026_final", "fixed_final", "A", "B"),
    ("joint_ncs1", "ncs1", "ncs1_compact_20261026_nmse", "joint_training", "C", "D"),
    ("joint_nof1", "nof1", "nof1_compact_20261026_final", "joint_training", "C", "D"),
)


def acceptance(report):
    models = report["models"]
    new = models["repaired"]
    old = models["historical_unfolding"]
    return dict(
        beats_same_weights_lmmse=new["mean_ber"] < models["lmmse"]["mean_ber"],
        beats_historical_lmmse=new["mean_ber"] < models["historical_lmmse"]["mean_ber"],
        within_original_total_budget=new["seconds_per_block"] <= old["seconds_per_block"],
        within_original_warm_model_budget=(new["benchmark"]["median_seconds_per_block"]
                                          <= old["benchmark"]["median_seconds_per_block"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revision", default="exact_legacy", help="Separate output for the source-authenticated legacy baseline")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    backend = DenseBackend(args.device)
    source = source_fingerprint(EVALUATION_CODE)
    root = ROOT
    checkpoints = {name: (LEGACY_ROOT if channel == "ncs1" else ROOT)
                   /training
                   /channel/"icr_mmse.pt" for name, channel, _, training, _, _ in CASES}
    # Check the entire study before starting, so a missing final model cannot
    # silently turn a four-case comparison into a partial one.
    hashes = {name: file_sha256(path) for name, path in checkpoints.items()}
    write_json(root/"detector_learning_audit.json", audit_study(checkpoints))
    if args.split == "test":
        for name in checkpoints:
            validation = json.loads((root/("frozen_validation_"+args.revision)/name/"comparison.json").read_text())
            if (not validation.get("complete") or validation["checkpoint_sha256"] != hashes[name]
                    or validation["source"] != source or validation["split"] != "validation"):
                raise ValueError(f"{name}: final model needs matching frozen validation")
            if not all(acceptance(validation).values()):
                raise ValueError(f"{name}: validation performance/budget acceptance failed")
    output = root/("frozen_"+args.split+"_"+args.revision)
    summary = dict(split=args.split, checkpoint_sha256=hashes, cases={}, complete=False,
                   detector_policy="all_physical_and_neural")
    with threadpool_limits(args.threads):
        for name, channel, dataset, _, linear, unfolded in CASES:
            run = output/name
            path = run/"comparison.json"
            reused = False
            if channel == "ncs1" and not path.exists():
                previous = LEGACY_ROOT/("frozen_"+args.split+"_"+args.revision)/name
                previous_report = json.loads((previous/"comparison.json").read_text())
                if (not previous_report.get("complete") or previous_report["checkpoint_sha256"] != hashes[name]
                        or previous_report["source"] != source or previous_report["split"] != args.split):
                    raise ValueError(f"{name}: unchanged NCS1 report identity differs; cannot reuse")
                for filename, identity in previous_report["data_identity"].items():
                    current = Path("data/processed")/dataset/filename
                    if current.stat().st_size != identity["bytes"] or file_sha256(current) != identity["sha256"]:
                        raise ValueError(f"{name}: data changed; cannot reuse frozen NCS1 results")
                shutil.copytree(previous, run)
                write_json(run/"reused_from.json", dict(path=str(previous),
                    report_sha256=file_sha256(previous/"comparison.json"),
                    reason="Weights, data, inference/evaluation source and all-parameter policy are unchanged"))
                reused = True
            if path.exists() and (args.resume or reused):
                report = json.loads(path.read_text())
                if (not report.get("complete") or report["checkpoint_sha256"] != hashes[name]
                        or report["source"] != source or report["split"] != args.split):
                    raise ValueError(f"{name}: existing comparison identity differs")
            else:
                print(f"Starting {args.split}: {name}", flush=True)
                old = Path("outputs/runs/icr_ad_ablation_20261030")/channel/"seed_20261028"
                report = compare(Path("data/processed")/dataset/(args.split+".h5"),
                    checkpoints[name], run, backend, split=args.split, batch_size=2,
                    historical={"historical_lmmse": old/linear/"icr_mmse.pt",
                                "historical_unfolding": old/unfolded/"icr_mmse.pt"})
            summary["cases"][name] = dict(acceptance=acceptance(report),
                models={key: dict(mean_ber=value["mean_ber"], seconds_per_block=value["seconds_per_block"],
                                 median_model_seconds_per_block=value["benchmark"]["median_seconds_per_block"])
                        for key, value in report["models"].items()})
            write_json(output/"summary.json", summary)
            print(name, json.dumps(summary["cases"][name]), flush=True)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    summary["complete"] = True
    write_json(output/"summary.json", summary)


if __name__ == "__main__":
    main()
