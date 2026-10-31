"""Train the current ICR branches, then publish complete seven-equalizer reports.

Run under nohup on the server. Each phase has its own process/log; progress.json
records failures as well as completion. Existing experiments are never replaced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from src.common.experiment import file_sha256, write_json
from src.evaluation.registry import SUPPORTED_CURVES, normalize_report_labels
from src.evaluation.result_cache import FileLock
from src.training.icr_workflow import DEFAULT_CONFIG, STAGES, resolve_config

LOG = logging.getLogger(__name__)
SNR = [0, 5, 10, 15, 20, 25, 30]
LABELS = SUPPORTED_CURVES


def comparison_config(training, comparison, output):
    """Bind evaluation to this run's new models, retaining valid baseline paths."""
    if training.get("input_view") != "compensated" or comparison.get("input_view") != "compensated":
        raise ValueError("The full experiment requires compensated CSI")
    if set(training["datasets"]) != {"NOF1", "NCS1"}:
        raise ValueError("The full experiment requires both channels")
    result = deepcopy(comparison)
    result["output"] = str(output/"comparison")
    result["datasets"] = {ch: str(Path(path)/"test.h5") for ch, path in training["datasets"].items()}
    for ch in training["datasets"]:
        checkpoints = result["checkpoints"][ch]
        checkpoints["icr_mmse"] = str(output/"joint"/ch.lower()/"icr_mmse.pt")
        if not Path(checkpoints["udnet"]).is_file():
            checkpoints["udnet"] = str(output/"udnet"/ch.lower()/"best.pt")
    return result


def preflight(training, comparison):
    import torch
    from src.common.numerics import DenseBackend
    from src.data.compact import CompactDataset
    from src.evaluation.unified import _preflight

    for ch, directory in training["datasets"].items():
        for split, count in (("train", 525), ("validation", 75), ("test", 600)):
            with CompactDataset(Path(directory)/(split+".h5"), input_view="compensated") as ds:
                if (ds.channel != ch or ds.split != split or ds.frame_count != count
                        or ds.snr_db.tolist() != SNR or len(ds._indices) != 512 or ds._config.get("smoke")):
                    raise ValueError(f"{ch}/{split}: requires the accepted complete dataset")
        paths = {k: v for k, v in comparison["checkpoints"][ch].items()
                 if k != "icr_mmse" and (k != "udnet" or Path(v).is_file())}
        _preflight(Path(directory)/"test.h5", paths, DenseBackend("cpu"))
    return dict(torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
                reused_checkpoints={ch: {k: dict(path=v, sha256=file_sha256(v))
                    for k, v in paths.items() if k != "icr_mmse" and Path(v).is_file()}
                    for ch, paths in comparison["checkpoints"].items()})


def validate_report(report, channel):
    """Do not declare success for partial coverage or historical ICR weights."""
    report = normalize_report_labels(report)
    if (not report.get("complete") or report.get("smoke") or report.get("input_view") != "compensated"
            or report.get("channel") != channel or report.get("snr_db") != SNR
            or set(report.get("curves", {})) != set(LABELS)):
        raise ValueError(f"{channel}: incomplete seven-algorithm report")
    for curve in report["curves"].values():
        if curve["total_bits"] != [1228800]*7 or len(curve["ber"]) != 7:
            raise ValueError(f"{channel}: incomplete test bit coverage")
    return dict(ber={name: row["ber"] for name, row in report["curves"].items()},
                mean_ber={name: sum(row["errors"])/sum(row["total_bits"])
                          for name, row in report["curves"].items()}, nmse=report["nmse"])


def execute(output, state, name, arguments):
    log = output/"logs"/(name+".log")
    state.update(status="running", phase=name, updated_at=datetime.now(timezone.utc).isoformat(),
                 command=[sys.executable, "-u", "-m", *arguments], log=str(log))
    write_json(output/"progress.json", state)
    LOG.info("Starting %s; log=%s", name, log)
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(state["command"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    state.setdefault("completed_phases", []).append(name)
    write_json(output/"progress.json", state)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output/".experiment.lock"):
        if (output/"manifest.json").exists() and not args.resume:
            raise FileExistsError("Experiment exists; use --resume or a new output directory")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if dirty:
            raise ValueError("Formal server experiments require a clean committed checkout")
        training = json.loads(args.config.read_text(encoding="utf-8"))
        original_comparison = json.loads(args.comparison_config.read_text(encoding="utf-8"))
        comparison = comparison_config(training, original_comparison, output)
        udnet = json.loads(Path("configs/training/udnet.json").read_text(encoding="utf-8"))
        if udnet["datasets"] != training["datasets"] or udnet["training"].get("input_view") != "compensated":
            raise ValueError("UDNet training must use the same accepted compensated datasets")
        for ch in training["datasets"]:
            for stage in STAGES:
                resolve_config(args.config, channel=ch, stage=stage, output=output)
        identity = dict(commit=commit, training=training, comparison=comparison, udnet=udnet,
                        device=args.device, threads=args.threads)
        if (output/"manifest.json").exists():
            previous = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
            if previous["identity"] != identity:
                raise ValueError("Resume requires the original code, configs and execution settings")
        state = dict(status="running", phase="preflight", started_at=datetime.now(timezone.utc).isoformat(),
                     pid=os.getpid(), complete=False)
        write_json(output/"progress.json", state)
        try:
            environment = preflight(training, comparison)
            if args.device == "cuda" and not environment["cuda_available"]:
                raise ValueError("CUDA requested but unavailable")
            write_json(output/"manifest.json", dict(identity=identity, environment=environment))
            write_json(output/"training_config.json", training)
            write_json(output/"comparison_config.json", comparison)
            write_json(output/"udnet_config.json", udnet)
            (output/"logs").mkdir(exist_ok=True)
            common = ["--device", args.device, "--threads", str(args.threads)]
            for ch in training["datasets"]:
                for stage in STAGES:
                    command = ["scripts.train.icr_mmse", "--config", str(output/"training_config.json"),
                               "--channel", ch, "--stage", stage, "--output", str(output), *common]
                    if args.resume:
                        command.append("--resume")
                    execute(output, state, f"{ch.lower()}_{stage}", command)
            from src.evaluation.icr_unified import audit_study
            state.update(phase="detector_learning_audit", log=None)
            write_json(output/"progress.json", state)
            checkpoints = {f"{branch}_{ch.lower()}": output/branch/ch.lower()/"icr_mmse.pt"
                           for ch in training["datasets"] for branch in ("fixed", "joint")}
            write_json(output/"detector_learning_audit.json", audit_study(checkpoints))
            for ch in training["datasets"]:
                target = output/"udnet"/ch.lower()/"best.pt"
                if Path(comparison["checkpoints"][ch]["udnet"]) == target:
                    command = ["scripts.train.udnet", "--config", str(output/"udnet_config.json"),
                               "--channel", ch, "--output", str(output/"udnet"), *common]
                    if args.resume and (target.parent/"latest.pt").exists():
                        command.append("--resume")
                    execute(output, state, f"{ch.lower()}_udnet", command)
                execute(output, state, f"{ch.lower()}_comparison", ["scripts.evaluate.compare", "--config",
                        str(output/"comparison_config.json"), "--channel", ch, *common])
                execute(output, state, f"{ch.lower()}_fixed_evaluation", ["scripts.evaluate.icr_mmse", "--config",
                        str(output/"training_config.json"), "--channel", ch, "--checkpoint",
                        str(output/"fixed"/ch.lower()/"icr_mmse.pt"), "--run",
                        str(output/"fixed_evaluation"/ch.lower()), *common])
            summaries = {}
            for ch in training["datasets"]:
                directory = output/"comparison"/ch.lower()
                report = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
                summaries[ch] = validate_report(report, ch)
                if Path(report["checkpoint_paths"]["icr_mmse"]).resolve() != Path(comparison["checkpoints"][ch]["icr_mmse"]).resolve():
                    raise ValueError("Comparison did not use this experiment's joint checkpoint")
                if report["icr_mmse"]["sha256"] != file_sha256(comparison["checkpoints"][ch]["icr_mmse"]):
                    raise ValueError("Comparison joint checkpoint hash changed")
                for name in ("ber_snr.png", "nmse_snr.png", "report.html"):
                    if not (directory/name).is_file():
                        raise FileNotFoundError(directory/name)
            write_json(output/"results_digest.json", dict(complete=True, channels=summaries))
            state.update(status="completed", phase="finished", complete=True)
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(output/"progress.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--comparison-config", type=Path, default=Path("configs/evaluation/ber_comparison.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
