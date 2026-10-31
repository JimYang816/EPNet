"""Unit-energy single-tap MMSE on raw data-carrier observations."""
from __future__ import annotations

import numpy as np

from src.algorithms.interface import EqualizerResult


def mmse(channel, observations, data_mask, noise_variance):
    h, y = np.asarray(channel), np.asarray(observations)
    mask = np.asarray(data_mask, bool)
    if h.shape != (len(mask), len(mask)) or y.shape != (len(mask),):
        raise ValueError("Single-tap channel/observation dimensions disagree")
    diagonal = np.diag(h)[mask]
    variance = np.asarray(noise_variance, float)
    if variance.shape == (len(mask),):
        variance = variance[mask]
    if variance.ndim and variance.shape != diagonal.shape:
        raise ValueError("Noise variance must be scalar or per carrier")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid single-tap inputs")
    denominator = abs(diagonal)**2+variance
    result = np.zeros(len(diagonal), np.complex128)
    np.divide(diagonal.conj()*y[mask], denominator, out=result, where=denominator > 0)
    return result


def equalize_batch(batch, backend, **parameters):
    """Use raw observations, including the original data-carrier interference."""
    if parameters:
        raise ValueError(f"Unknown single-tap parameters: {sorted(parameters)}")
    b, indices = backend, batch.data_indices
    diagonal = (batch.channel.diagonal(dim1=-2, dim2=-1) if b.torch
                else np.diagonal(batch.channel, axis1=-2, axis2=-1))[:, indices]
    noise = b.array(np.diag(batch.noise_covariance).real[indices])
    denominator = abs(diagonal)**2+b.array(batch.variances)[:, None]*noise
    numerator = diagonal.conj()*batch.raw_observations[:, indices]
    if b.torch:
        safe = b.torch.where(denominator > 0, denominator, b.torch.ones_like(denominator))
        estimate = b.torch.where(denominator > 0, numerator/safe, b.torch.zeros_like(numerator))
    else:
        estimate = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=estimate, where=denominator > 0)
    return EqualizerResult(b.numpy(estimate), np.zeros(len(batch.variances), dtype=np.int64))
