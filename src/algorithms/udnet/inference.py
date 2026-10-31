"""Label-free local UDNet adapter returning the existing EqualizerResult."""
from __future__ import annotations

import numpy as np
import torch

from src.algorithms.interface import EqualizerResult
from src.data.udnet import FEATURES


@torch.no_grad()
def predict_features(model, features, *, window_batch_size=2000):
    if window_batch_size < 1:
        raise ValueError("Window batch size must be positive")
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(features["gram"]), window_batch_size):
        inputs = [torch.as_tensor(features[name][start:start+window_batch_size], device=device) for name in FEATURES]
        outputs.append(model(*inputs).cpu().numpy())
    if not outputs:
        raise ValueError("No inference windows")
    return np.concatenate(outputs)


def equalize_batch(batch, backend, *, model, preprocessor, window_batch_size=2000):
    features = preprocessor(batch)
    probabilities = predict_features(model, features, window_batch_size=window_batch_size)
    classes = probabilities[:, -1].argmax(axis=-1)
    constellation = model.constellation.detach().cpu().numpy()
    symbols = constellation[classes, 0]+1j*constellation[classes, 1]
    plan = preprocessor.plan
    symbols = symbols.reshape(len(batch.variances), len(plan.positions), 32)
    return EqualizerResult(plan.stitch(symbols), np.zeros(len(symbols), dtype=np.int64))
