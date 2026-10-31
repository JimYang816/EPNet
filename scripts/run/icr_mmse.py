"""Train the current shared warmup, fixed-CSI and joint ICR-MMSE branches."""
import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.training.icr_workflow import DEFAULT_CONFIG, run_channel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Small CPU smoke run only")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    channels = json.loads(args.config.read_text(encoding="utf-8"))["datasets"]
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        for channel in channels:
            if not args.channel or args.channel == channel:
                run_channel(args.config, channel, DenseBackend(args.device), output=args.output,
                            resume=args.resume, frame_limit=args.max_frames)


if __name__ == "__main__":
    main()
