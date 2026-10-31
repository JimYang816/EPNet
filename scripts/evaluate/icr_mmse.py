"""Score ICR-MMSE-Net with full-matrix NMSE and record-grouped statistics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.evaluation.icr_mmse import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/icr_mmse.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--udnet-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--operator-backend", choices=("dense", "dpss"), default="dense")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("input_view", "compensated") != "compensated":
        parser.error("Default ICR evaluation is fixed to the accepted compensated CSI")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        evaluate(Path(config["datasets"][args.channel])/"test.h5", args.checkpoint, args.run,
                 DenseBackend(args.device), udnet_checkpoint=args.udnet_checkpoint,
                 operator_backend=args.operator_backend, **config.get("evaluation", {}))


if __name__ == "__main__":
    main()
