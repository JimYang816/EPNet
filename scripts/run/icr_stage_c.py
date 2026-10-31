"""Supervise both stage-C controls; suitable for a detached systemd user service."""
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
    parser.add_argument("--config", type=Path, default=ROOT/"configs/training/icr_stage_c_lr1e4.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.gpu < 0:
        parser.error("Threads must be positive and GPU index nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("input_view") != "compensated" or config.get("channels") != ["NOF1", "NCS1"]:
        parser.error("This run requires compensated CSI and NOF1 followed by NCS1")
    os.chdir(ROOT)
    os.environ.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads))
    gpu = None
    if args.device == "cuda":
        gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from threadpoolctl import threadpool_limits
    from src.common.experiment import write_json
    from src.common.numerics import DenseBackend
    from src.evaluation.icr_stage_c import run_channel, save_comparison
    from src.evaluation.result_cache import FileLock

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    output = (ROOT/config["output"]).resolve()
    control = output/"_control"
    control.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(FileLock(control/"supervisor.lock"))
        if gpu:
            stack.enter_context(FileLock(ROOT/"outputs/runs/_gpu_leases"/(gpu+".lock")))
        stack.enter_context(threadpool_limits(args.threads))
        snapshot = control/"config.json"
        if snapshot.exists() and json.loads(snapshot.read_text()) != config:
            raise ValueError("Saved supervisor configuration differs")
        write_json(snapshot, config)
        state = dict(state="running", complete=False, started_at=now(), updated_at=now(),
            pid=os.getpid(), gpu_uuid=gpu, channels={name: dict(state="queued") for name in config["channels"]})
        write_json(control/"manifest.json", dict(config=config,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            device=args.device, threads=args.threads, gpu_uuid=gpu, started_at=state["started_at"]))
        write_json(control/"status.json", state)
        results = {}
        for channel in config["channels"]:
            item = state["channels"][channel]

            def phase(value):
                item.update(state="running", phase=value, phase_started_at=now())
                state["updated_at"] = now()
                write_json(control/"status.json", state)
                logging.info("%s %s", channel, value)

            try:
                phase("checking_source")
                results[channel] = run_channel(ROOT/config["source_root"], output, channel,
                    DenseBackend(args.device), learning_rate=config["learning_rate"],
                    resume=args.resume, phase_callback=phase,
                    correction_learning_rate=config.get("correction_learning_rate"),
                    nmse_max_relative_degradation=config.get("nmse_max_relative_degradation"),
                    previous_root=ROOT/config["previous_root"] if config.get("previous_root") else None)
                item.update(state="completed", phase="finished", finished_at=now())
                save_comparison(results, output)
            except Exception as error:
                item.update(state="failed", error=str(error), finished_at=now())
                (control/(channel.lower()+"_failure.txt")).write_text(traceback.format_exc(), encoding="utf-8")
                logging.exception("%s failed", channel)
            state["updated_at"] = now()
            write_json(control/"status.json", state)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        state["complete"] = all(item["state"] == "completed" for item in state["channels"].values())
        state.update(state="completed" if state["complete"] else "completed_with_errors", finished_at=now())
        write_json(control/"status.json", state)
        return 0 if state["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
