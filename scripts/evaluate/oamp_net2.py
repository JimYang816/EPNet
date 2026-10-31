"""Score OAMP-Net2 and traditional comparators on the complete test split."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.evaluation.oamp_net2 import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/oamp_net2.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Refresh OAMP-Net2; retain valid traditional caches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        for channel, directory in config["datasets"].items():
            if args.channel and args.channel != channel:
                continue
            run = Path(config["output"])/channel.lower()
            evaluate(Path(directory)/"test.h5", run/"best.pt", run/"evaluation", backend,
                     force=args.force, **config["evaluation"])


if __name__ == "__main__":
    main()
