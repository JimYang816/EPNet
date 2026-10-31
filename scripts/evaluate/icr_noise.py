"""Compare full and diagonal noise with one ICR-MMSE checkpoint and fixed data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.common.numerics import DenseBackend
from src.evaluation.icr_noise import compare_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--frame-selection", choices=("first", "evenly_spaced"), default="evenly_spaced")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(args.threads):
        report = compare_noise(args.dataset, args.checkpoint, args.run, DenseBackend(args.device),
            batch_size=args.batch_size, frame_limit=args.max_frames, timing_repeats=args.timing_repeats,
            frame_selection=args.frame_selection)
    print(report["channel"], report["differences"], flush=True)


if __name__ == "__main__":
    main()
