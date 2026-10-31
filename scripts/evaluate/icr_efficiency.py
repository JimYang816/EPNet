"""Benchmark four compensated-CSI detectors in four fresh, sequential processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.icr_efficiency import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--udnet-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = benchmark(args.dataset, args.comparison_report, args.checkpoint, args.udnet_checkpoint,
        output=args.output, device=args.device, batch_size=args.batch_size,
        frame_limit=args.max_frames, threads=args.threads)
    if not args.output:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
