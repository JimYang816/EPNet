"""Compare frozen ICR validation BER with CPU full-matrix LMMSE on identical dataset inputs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.evaluation.icr_validation_compare import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with threadpool_limits(args.threads):
        result = evaluate(args.dataset, args.candidate_report, args.output, batch_size=args.batch_size)
    logging.info("Complete: %s", result["overall_ber"])


if __name__ == "__main__":
    main()
