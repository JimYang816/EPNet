"""Run the four-equalizer BER comparison on complete compact-v2 test splits."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.evaluation.comparison import evaluate, replot
from src.evaluation.registry import IDENTIFIERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/ber_comparison.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--force", nargs="+", choices=(*IDENTIFIERS, "all"), default=[],
                       help="Recompute these detectors; still publish all four curves")
    modes.add_argument("--plot-only", action="store_true", help="Redraw saved complete reports without data or GPU")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.plot_only:
        for channel in config["datasets"]:
            if not args.channel or channel == args.channel:
                replot(Path(config["output"])/channel.lower())
        return
    with threadpool_limits(args.threads):
        backend = DenseBackend(args.device)
        if backend.torch:
            backend.torch.set_num_threads(args.threads)
        for channel, dataset in config["datasets"].items():
            if args.channel and channel != args.channel:
                continue
            evaluate(dataset, Path(config["output"])/channel.lower(), backend,
                     batch_size=args.batch_size if args.batch_size is not None else config["batch_size"],
                     algorithm_parameters=config.get("algorithms"), force=args.force,
                     cache_dir=config.get("cache_dir"), **config.get("ep", {}))


if __name__ == "__main__":
    main()
