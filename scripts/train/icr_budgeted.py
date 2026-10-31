"""Finish fixed-CSI detector training from the shared budgeted warmup."""
import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.training.icr_mmse import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=("NOF1", "NCS1"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, help="Root for fixed-CSI final models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    channel = args.channel.lower()
    config = json.loads(Path(f"configs/training/icr_budgeted_{channel}.json").read_text())
    if config["training"].get("detector_parameters", "all") != "all":
        raise ValueError("Unified detector training requires both physical and neural parameters")
    controls = dict(config["training"], stage_epochs=[10, 0, 10], stage_c_learning_rate=.0001,
        warmup_checkpoint=str(Path(config["output"])/channel/"warmup.pt"))
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        output = args.output or Path(config["output"]).parent/"fixed_final"
        train(config["datasets"][args.channel], output/channel,
              DenseBackend(args.device), model_config=config["model"], resume=args.resume, **controls)


if __name__ == "__main__":
    main()
