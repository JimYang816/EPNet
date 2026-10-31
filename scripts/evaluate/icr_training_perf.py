"""Validate ICR equivalence, then select a batch by isolated training trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.icr_training_perf import benchmark_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/icr_mmse.json"))
    parser.add_argument("--channel", choices=("NOF1", "NCS1"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--operator-backend", choices=("dense_device", "dpss"), default="dense_device")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--stages", choices=("A", "B", "C"), nargs="+", default=["A", "B", "C"])
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--validation-frames", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--reference-cache-mib", type=int, default=512)
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = benchmark_training(config["datasets"][args.channel], args.output, model_config=config["model"],
        checkpoint=args.checkpoint, device=args.device, batch_sizes=args.batch_sizes, stages=args.stages,
        frame_limit=args.frames, validation_frames=args.validation_frames, repeats=args.repeats,
        threads=args.threads, prefetch_batches=args.prefetch_batches,
        reference_cache_mib=args.reference_cache_mib, ablations=not args.skip_ablations,
        operator_backend=args.operator_backend)
    print(json.dumps(dict(channel=report["channel"], selection=report["selection"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
