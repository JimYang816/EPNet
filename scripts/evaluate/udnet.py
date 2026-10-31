"""Independent complete UDNet/four-baseline comparison."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.evaluation.udnet import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/udnet.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        for channel, directory in config["datasets"].items():
            if args.channel and channel != args.channel:
                continue
            run = (args.run_root or Path(config["output"]))/channel.lower()
            evaluate(Path(directory)/"test.h5", run/"best.pt", run/"evaluation", DenseBackend(args.device),
                     cache_dir=args.cache_dir, **config["evaluation"])


if __name__ == "__main__":
    main()
