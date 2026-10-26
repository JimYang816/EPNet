"""Unit-energy single-tap MMSE on raw data-carrier observations."""
from __future__ import annotations

import numpy as np


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
