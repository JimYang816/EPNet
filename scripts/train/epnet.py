"""Train paper EPNet on the fixed NCS1/NOF1 compact-v2 train/validation data."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.training.epnet import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/epnet.json"))
    parser.add_argument("--channel", choices=("NCS1", "NOF1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--optimizer", choices=("lstm", "adam"))
    parser.add_argument("--meta-epochs", type=int)
    parser.add_argument("--adaptation-steps", type=int)
    parser.add_argument("--max-frames", type=int, help="Limit each split for a CPU smoke check")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = dict(config.get("training", {}))
    for name in ("batch_size", "epochs", "optimizer", "adaptation_steps"):
        value = getattr(args, name)
        if value is not None:
            options[name] = value
    if args.max_frames is not None:
        options["frame_limit"] = args.max_frames
    if args.meta_epochs is not None:
        options["meta_config"] = dict(options.get("meta_config", {}), epochs=args.meta_epochs)
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        import torch
        torch.set_num_threads(args.threads)
        for channel, directory in config["datasets"].items():
            if args.channel and channel != args.channel:
                continue
            train(directory, (args.output or Path(config["output"]))/channel.lower(), backend,
                  model_config=config.get("model"), resume=args.resume, **options)


if __name__ == "__main__":
    main()
