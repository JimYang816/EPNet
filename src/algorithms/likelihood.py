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


def real_likelihood(gram, matched):
    return (2*np.concatenate((np.concatenate((gram.real, -gram.imag), axis=-1),
                              np.concatenate((gram.imag, gram.real), axis=-1)), axis=-2),
            2*np.concatenate((matched.real, matched.imag), axis=-1))


def backend_real_likelihood(gram, matched, backend):
    if not backend.torch:
        return real_likelihood(gram, matched)
    t = backend.torch
    return (2*t.cat((t.cat((gram.real, -gram.imag), dim=-1),
                      t.cat((gram.imag, gram.real), dim=-1)), dim=-2),
            2*t.cat((matched.real, matched.imag), dim=-1))


def batch_likelihood(batch, whitener, backend):
    """Memoize pure likelihood preparation; detectors treat these arrays as read-only."""
    key = ("complex", id(whitener))
    if key not in batch.likelihoods:
        batch.likelihoods[key] = likelihood_batch(batch.channel[..., batch.data_indices],
            batch.cancelled_observations, batch.variances, whitener, backend)
    return batch.likelihoods[key]


def batch_real_likelihood(batch, whitener, backend):
    key = ("real", id(whitener))
    if key not in batch.likelihoods:
        gram, matched = batch_likelihood(batch, whitener, backend)
        batch.likelihoods[key] = backend_real_likelihood(gram, matched, backend)
    return batch.likelihoods[key]
