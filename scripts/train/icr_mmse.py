"""Train independent NOF1/NCS1 ICR-MMSE-Net models in stages A/B/C."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.training.icr_mmse import train
from src.training.icr_workflow import resolve_config, STAGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/icr_mmse.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    parser.add_argument("--stage", choices=STAGES, default="joint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames", type=int, help="CPU smoke only; marks checkpoint as smoke")
    parser.add_argument("--stage-epochs", type=int, nargs=3, metavar=("A", "B", "C"))
    parser.add_argument("--precision", choices=("complex64", "complex128"), help="Override model precision")
    parser.add_argument("--noise-mode", choices=("diagonal", "full"), help="Override noise covariance weighting")
    parser.add_argument("--operator-backend", choices=("dense", "dpss", "dense_device"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prefetch-batches", type=int)
    parser.add_argument("--reference-cache-mib", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    if configuration.get("input_view", "compensated") != "compensated":
        parser.error("Default ICR training is fixed to the accepted compensated CSI")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        for channel, directory in configuration["datasets"].items():
            if args.channel and channel != args.channel:
                continue
            resolved = resolve_config(args.config, channel=channel, stage=args.stage, output=args.output)
            for name in ("precision", "noise_mode"):
                if getattr(args, name) is not None:
                    resolved["model"][name] = getattr(args, name)
            controls = resolved["training"]
            for name in ("operator_backend", "batch_size", "prefetch_batches", "reference_cache_mib"):
                if getattr(args, name) is not None:
                    controls[name] = getattr(args, name)
            if args.max_frames is not None:
                controls["frame_limit"] = args.max_frames
            if args.stage_epochs is not None:
                controls["stage_epochs"] = args.stage_epochs
            train(directory, Path(resolved["output"]) / channel.lower(),
                  DenseBackend(args.device), model_config=resolved["model"], resume=args.resume,
                  **controls)


if __name__ == "__main__":
    main()
