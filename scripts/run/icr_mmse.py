"""Run train/validation coverage and selection, then the configured seed runs."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.evaluation.icr_selection import run_channel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/icr_mmse.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--precision", choices=("complex64", "complex128"), help="Override model precision")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.precision is not None:
        config["model"]["precision"] = args.precision
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/"resolved_config.json", config)
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        for channel, directory in config["datasets"].items():
            if args.channel and channel != args.channel:
                continue
            run_channel(directory, root/channel.lower(), backend, config, resume=args.resume,
                        udnet_checkpoint=config.get("udnet_checkpoints", {}).get(channel))


if __name__ == "__main__":
    main()
