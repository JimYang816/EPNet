"""Block MMSE-DFE on the complete, thermally whitened likelihood.

For G = H^H R_n^-1 H and g = H^H R_n^-1 y, factor I+G = L L^H.
The feedforward output is z = L^-1 g. Back substitution through L^H
uses hard unit-energy QPSK decisions, rather than linear estimates, as
feedback. Data columns retain their supplied order; detection proceeds
from the last column to the first, without channel-dependent ordering.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from src.algorithms.interface import EqualizerResult
from src.algorithms.likelihood import batch_likelihood


def mmse_dfe(gram, matched, backend):
    """Return pre-slicer MMSE estimates, using only hard decisions in feedback.

    Supports arbitrary leading batch axes. The identity loading represents
    the unit symbol energy after thermal whitening, not a ZF approximation.
    Zero real/imaginary components slice to the positive QPSK component,
    matching the project's BER decision rule. No labels enter detection.
    """
    t = backend.torch
    xp = t if t else np
    n = matched.shape[-1]
    if gram.shape != (*matched.shape, n) or n < 1:
        raise ValueError("MMSE-DFE likelihood dimensions disagree")
    if not bool(xp.isfinite(gram).all()) or not bool(xp.isfinite(matched).all()):
        raise ValueError("Nonfinite MMSE-DFE likelihood")
    lower = xp.linalg.cholesky(gram+backend.array(np.eye(n, dtype=np.complex128)))
    if t:
        filtered = t.linalg.solve_triangular(lower, matched[..., None], upper=False)[..., 0]
        upper = lower.conj().transpose(-2, -1)
    else:
        # scipy's triangular solve also supports older runtimes without batched
        # linalg APIs; keep the O(n^2) solve for each independent block.
        filtered = np.stack([linalg.solve_triangular(factor, rhs, lower=True, check_finite=False)
                             for factor, rhs in zip(lower.reshape(-1, n, n), matched.reshape(-1, n))])
        filtered = filtered.reshape(matched.shape)
        upper = lower.conj().swapaxes(-2, -1)
    estimates = xp.zeros_like(matched)
    decisions = xp.zeros_like(matched)
    amplitude = np.sqrt(.5)
    for i in range(n-1, -1, -1):
        feedback = (upper[..., i, i+1:]*decisions[..., i+1:]).sum(-1)
        value = (filtered[..., i]-feedback)/upper[..., i, i].real
        estimates[..., i] = value
        decisions[..., i] = (xp.where(value.real < 0, -amplitude, amplitude)
                            +1j*xp.where(value.imag < 0, -amplitude, amplitude))
    if not bool(xp.isfinite(estimates).all()):
        raise FloatingPointError("Nonfinite MMSE-DFE estimate")
    return estimates


def equalize_batch(batch, backend, *, whitener, **parameters):
    if parameters:
        raise ValueError(f"Unknown MMSE-DFE parameters: {sorted(parameters)}")
    gram, matched = batch_likelihood(batch, whitener, backend)
    symbols = mmse_dfe(gram, matched, backend)
    return EqualizerResult(backend.numpy(symbols), np.zeros(len(batch.variances), dtype=np.int64))
