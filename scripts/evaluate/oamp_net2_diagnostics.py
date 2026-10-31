"""Audit and fine-tune OAMP-Net2 using only existing training/validation splits."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.training.oamp_net2_diagnostics import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--joint-epochs", type=int, default=50)
    parser.add_argument("--snr-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        diagnose(args.data, args.checkpoint, args.output, DenseBackend(args.device),
                 joint_epochs=args.joint_epochs, snr_epochs=args.snr_epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
