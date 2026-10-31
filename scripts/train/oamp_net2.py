"""Train channel-specific OAMP-Net2 models on the fixed compensated-CSI splits."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.training.oamp_net2 import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/oamp_net2.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-frames", type=int, help="CPU smoke check only; its checkpoint is rejected by formal evaluation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = dict(config["training"])
    for name in ("epochs", "batch_size"):
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    if args.max_frames is not None:
        options["frame_limit"] = args.max_frames
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        for channel, directory in config["datasets"].items():
            if args.channel and args.channel != channel:
                continue
            train(directory, (args.output or Path(config["output"]))/channel.lower(), backend,
                  model_config=config["model"], resume=args.resume, **options)


if __name__ == "__main__":
    main()
