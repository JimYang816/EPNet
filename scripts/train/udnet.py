"""Train one independent UDNet per selected channel on all SNRs and D1/D2."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.training.udnet import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/udnet.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, help="Explicit override; omit for required CUDA resource probes")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-frames", type=int, help="CPU smoke check only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = dict(config["training"])
    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    if args.epochs is not None:
        options["epochs"] = args.epochs
    if args.max_frames is not None:
        options["frame_limit"] = args.max_frames
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        for channel, directory in config["datasets"].items():
            if args.channel and channel != args.channel:
                continue
            train(directory, (args.output or Path(config["output"]))/channel.lower(), DenseBackend(args.device),
                  model_config=config["model"], resume=args.resume, **options)


if __name__ == "__main__":
    main()
