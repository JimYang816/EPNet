"""Label-free inference from a compact dataset or saved sufficient statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.algorithms.udnet.checkpoint import load_model, validate_dataset
from src.algorithms.udnet.inference import equalize_batch, predict_features
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.udnet import FEATURES, WindowPlan, LocalPreprocessor
from src.evaluation.inputs import prepare_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--features", type=Path, help="NPZ: gram, matched, zf, data_mask; no labels")
    parser.add_argument("--snr", type=float, default=0)
    parser.add_argument("--frames", default="0", help="Comma-separated sorted frame indices")
    parser.add_argument("--d", type=int, choices=(1, 2), default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(args.checkpoint, args.device)
    with threadpool_limits(args.threads):
        if args.features:
            with np.load(args.features, allow_pickle=False) as source:
                features = {name: source[name] for name in FEATURES}
            probabilities = predict_features(model, features)
            np.savez_compressed(args.output, probabilities=probabilities,
                                classes=probabilities[:, -1].argmax(axis=-1), data_mask=features["data_mask"])
        else:
            with CompactDataset(args.dataset) as ds:
                validate_dataset(payload, ds)
                selected = np.flatnonzero(np.isclose(ds.snr_db, args.snr, rtol=0, atol=1e-8))
                frames = [int(value) for value in args.frames.split(",")]
                if len(selected) != 1 or frames != sorted(set(frames)) or not frames or min(frames) < 0 or max(frames) >= ds.frame_count:
                    raise ValueError("Invalid SNR/frame selection")
                plan = WindowPlan.build(ds._config["nfft"], ds._indices)
                prepared = LocalPreprocessor(plan, ds._shared["unit_rf_noise_covariance"][:], args.device)
                backend = DenseBackend(args.device)
                result = equalize_batch(prepare_batch(ds, int(selected[0]), frames, args.d-1, backend), backend,
                                        model=model, preprocessor=prepared)
                np.savez_compressed(args.output, symbols=result.symbols, rejected_sites=result.rejected_sites,
                                    data_indices=ds._indices, frames=frames, snr_db=args.snr, d=args.d)


if __name__ == "__main__":
    main()
