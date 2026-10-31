"""Run a two-epoch validation-only final-detector diagnostic under one GPU lease."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/training/icr_final_detector.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.gpu < 0:
        parser.error("Threads must be positive and GPU index nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    os.chdir(ROOT)
    os.environ.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
        NUMEXPR_NUM_THREADS=str(args.threads), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    gpu = None
    if args.device == "cuda":
        gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch
    from threadpoolctl import threadpool_limits
    from src.common.experiment import write_json
    from src.common.numerics import DenseBackend
    from src.evaluation.result_cache import FileLock
    from src.training.icr_final_detector import run_pair

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = ROOT/config["output"]
    control = output/"_control"
    control.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(FileLock(control/"supervisor.lock"))
        if gpu:
            stack.enter_context(FileLock(ROOT/"outputs/runs/_gpu_leases"/(gpu+".lock")))
        stack.enter_context(threadpool_limits(args.threads))
        identity = dict(config=config, commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        device=args.device, threads=args.threads, gpu_uuid=gpu)
        manifest = control/"manifest.json"
        if manifest.exists() and (not args.resume or json.loads(manifest.read_text())["identity"] != identity):
            raise ValueError("Existing diagnostic requires --resume with unchanged identity")
        write_json(manifest, dict(identity=identity))
        state = dict(state="running", complete=False, pid=os.getpid(), test_access=False)

        def phase(value):
            state.update(phase=value, updated_at=datetime.now(timezone.utc).isoformat())
            write_json(control/"status.json", state)
            logging.info("%s", value)

        try:
            phase("checking_source")
            run_pair(ROOT/config["source_root"], output, config["channel"], DenseBackend(args.device),
                     epochs=config["epochs"], learning_rate=config["learning_rate"],
                     resume=args.resume, phase_callback=phase)
            state.update(state="completed", complete=True)
            phase("finished")
        except Exception as error:
            state.update(state="failed", error=str(error))
            phase("failed")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
