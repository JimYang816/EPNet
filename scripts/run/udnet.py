"""Run NOF1 then NCS1: train, complete evaluation, remove disposable features."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.data.udnet_cache import remove_features
from src.evaluation.udnet import evaluate
from src.training.udnet import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/udnet.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"resolved_config.json", config)
    torch.set_num_threads(args.threads)
    reports = {}
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        for channel in ("NOF1", "NCS1"):
            directory, run = Path(config["datasets"][channel]), root/channel.lower()
            best = train(directory, run, backend, model_config=config["model"],
                         resume=args.resume and (run/"latest.pt").exists(), **config["training"])
            report = evaluate(directory/"test.h5", best, run/"evaluation", backend, **config["evaluation"])
            reports[channel] = dict(complete=True, best=str(best), evaluation=str(run/"evaluation"),
                                    ber=report["curves"]["UDNet"]["ber"], ser=report["curves"]["UDNet"]["ser"])
            feature_root = run/"features"
            if feature_root.exists():
                for path in feature_root.iterdir():
                    if path.is_dir():
                        remove_features(path, feature_root)
                    elif path.suffix == ".lock":
                        path.unlink()
                feature_root.rmdir()
            torch.cuda.empty_cache() if backend.device == "cuda" else None
            write_json(root/"completion.json", dict(complete=len(reports) == 2, channels=reports))


if __name__ == "__main__":
    main()
