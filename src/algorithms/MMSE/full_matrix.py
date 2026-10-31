"""Unit-energy LMMSE using every receive row and every data-column coupling."""
from __future__ import annotations

import numpy as np

from src.algorithms.interface import EqualizerResult
from src.algorithms.likelihood import batch_likelihood


def lmmse(system, observation, noise_variance):
    gram, matched = system.likelihood(observation, noise_variance)
    b = system.backend
    return b.numpy(b.solve(gram+b.array(np.eye(system.dimension, dtype=complex)), matched))


def equalize_batch(batch, backend, *, whitener, **parameters):
    if parameters:
        raise ValueError(f"Unknown full-matrix parameters: {sorted(parameters)}")
    b = backend
    gram, matched = batch_likelihood(batch, whitener, b)
    identity = b.array(np.eye(len(batch.data_indices), dtype=complex))
    solver = b.torch.linalg.solve if b.torch else np.linalg.solve
    estimate = solver(gram+identity, matched[..., None])[..., 0]
    return EqualizerResult(b.numpy(estimate), np.zeros(len(batch.variances), dtype=np.int64))
