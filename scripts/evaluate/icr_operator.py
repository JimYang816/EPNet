"""Validate dense/DPSS gradients, then compare same-checkpoint accuracy/resources."""
import argparse
from pathlib import Path

from src.evaluation.icr_operator import compare_operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--diagnose-native-drift", action="store_true",
                        help="Keep native strict failure, require a double-precision control, then measure native drift")
    args = parser.parse_args()
    report = compare_operators(args.dataset, args.checkpoint, args.run, device=args.device,
        batch_size=args.batch_size, frame_limit=None if args.full_test else args.max_frames,
        timing_repeats=args.timing_repeats, threads=args.threads,
        diagnose_native_drift=args.diagnose_native_drift)
    print(report["channel"], report["differences"], flush=True)


if __name__ == "__main__":
    main()
