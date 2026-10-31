"""Run the A-D ablations serially, with checkpoint recovery and atomic status."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import traceback


ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/training/icr_ad_ablation.json")
    parser.add_argument("--project-root", type=Path, default=ROOT,
                        help="Data/output root; may differ from this immutable code worktree")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("input_view") != "compensated" or config.get("channels") != ["NOF1", "NCS1"]:
        parser.error("Formal A-D runs require compensated CSI and both channels")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        parser.error("Seeds must be distinct nonnegative integers")
    if args.threads < 1 or args.gpu < 0 or config.get("deterministic") is not True:
        parser.error("Positive thread count, nonnegative GPU and deterministic training are required")
    project = args.project_root.resolve()
    config["datasets"] = {channel: str((project/path).resolve()) for channel, path in config["datasets"].items()}
    output = (project/config["output"]).resolve()
    control = output/"_control"
    control.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads))
    gpu = None
    if args.device == "cuda":
        gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import numpy  # Load the numerical runtime before torch on Windows CPU checks.
    import torch
    from threadpoolctl import threadpool_limits
    from src.common.experiment import write_json
    from src.common.numerics import DenseBackend
    from src.evaluation.icr_ad_ablation import run_quartet, save_comparison
    from src.evaluation.result_cache import FileLock

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with ExitStack() as stack:
        stack.enter_context(FileLock(control/"supervisor.lock"))
        if gpu:
            stack.enter_context(FileLock(project/"outputs/runs/_gpu_leases"/(gpu+".lock")))
        stack.enter_context(threadpool_limits(args.threads))
        identity = dict(config=config, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        code_root=str(ROOT), project_root=str(project), device=args.device,
                        threads=args.threads, gpu_uuid=gpu)
        manifest = control/"manifest.json"
        if manifest.exists():
            if not args.resume or json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
                raise ValueError("Resume requires unchanged code/configuration/data roots/device")
        else:
            write_json(manifest, dict(identity=identity, started_at=now()))
        write_json(control/"config.json", config)
        jobs = [(channel, seed) for seed in seeds for channel in config["channels"]]
        state = dict(state="running", complete=False, pid=os.getpid(), updated_at=now(),
                     jobs={f"{channel}/{seed}": dict(state="queued") for channel, seed in jobs})
        write_json(control/"status.json", state)
        results = []
        for channel, seed in jobs:
            item = state["jobs"][f"{channel}/{seed}"]

            def phase(value):
                item.update(state="running", phase=value, phase_started_at=now())
                state["updated_at"] = now()
                write_json(control/"status.json", state)
                logging.info("%s seed=%d %s", channel, seed, value)

            try:
                result = run_quartet(config, channel, seed, output, DenseBackend(args.device),
                                     resume=args.resume, phase_callback=phase)
                results.append(result)
                save_comparison(results, output)
                item.update(state="completed", phase="finished", finished_at=now())
                # Selected checkpoints and training histories are retained.
                for name in ("A", "B", "C", "D", "shared_warmup"):
                    (output/channel.lower()/f"seed_{seed}"/name/"latest.pt").unlink(missing_ok=True)
            except Exception as error:
                item.update(state="failed", error=str(error), finished_at=now())
                (control/f"{channel.lower()}_{seed}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                logging.exception("%s seed=%d failed", channel, seed)
            state["updated_at"] = now()
            write_json(control/"status.json", state)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        state["complete"] = all(item["state"] == "completed" for item in state["jobs"].values())
        state.update(state="completed" if state["complete"] else "completed_with_errors", finished_at=now())
        write_json(control/"status.json", state)
        return 0 if state["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
