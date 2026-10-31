"""Shared Gaussian likelihood operations; no detector iterations or labels."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from src.common.numerics import hermitian


def make_whitener(covariance, backend):
    factor = linalg.cholesky(hermitian(covariance), lower=True, check_finite=False)
    return backend.array(linalg.solve_triangular(factor, np.eye(len(factor)), lower=True, check_finite=False))


def likelihood_batch(channels, observations, variances, whitener, backend):
    b = backend
    on_backend = lambda x: x if b.torch and isinstance(x, b.torch.Tensor) else b.array(x)
    h = whitener @ on_backend(channels)
    y = (whitener @ on_backend(observations)[..., None])[..., 0]
    ht = h.conj().transpose(-2, -1) if b.torch else h.conj().swapaxes(-2, -1)
    scale = b.array(np.asarray(variances))
    return (ht@h)/scale[..., None, None], (ht@y[..., None])[..., 0]/scale[..., None]


def batch_likelihood(batch, whitener, backend):
    """Memoize pure likelihood preparation; detectors treat these arrays as read-only."""
    key = ("complex", id(whitener))
    if key not in batch.likelihoods:
        batch.likelihoods[key] = likelihood_batch(batch.channel[..., batch.data_indices],
            batch.cancelled_observations, batch.variances, whitener, backend)
    return batch.likelihoods[key]
