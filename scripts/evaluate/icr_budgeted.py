"""Compare a frozen same-budget unfolding checkpoint against LMMSE and controls."""
import argparse
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.evaluation.icr_budgeted import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--historical-lmmse", type=Path)
    parser.add_argument("--historical-unfolding", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    historical = {name: path for name, path in (("historical_lmmse", args.historical_lmmse),
                  ("historical_unfolding", args.historical_unfolding)) if path is not None}
    with threadpool_limits(args.threads):
        report = compare(args.dataset/(args.split+".h5"), args.checkpoint, args.output,
            DenseBackend(args.device), split=args.split, batch_size=args.batch_size, historical=historical)
    for name, row in report["models"].items():
        print(name, "mean BER", row["mean_ber"], "seconds/block", row["seconds_per_block"], flush=True)


if __name__ == "__main__":
    main()
