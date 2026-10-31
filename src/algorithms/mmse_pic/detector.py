"""MMSE-PIC on the complete, thermally whitened data-column likelihood.

This is an uncoded soft-feedback adaptation of the MMSE-PIC demapper, not a
turbo receiver. CSI stays fixed; previous decisions cancel OTHER symbols.
The target symbol's own prior is removed from its scalar likelihood.
"""
from __future__ import annotations

import numpy as np

from src.algorithms.interface import EqualizerResult
from src.algorithms.likelihood import batch_likelihood


def controls(*, iterations=10, damping=.5, variance_floor=1e-10):
    if (not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1
            or not np.isfinite(damping) or not 0 < damping <= 1
            or not np.isfinite(variance_floor) or not 0 < variance_floor < 1):
        raise ValueError("Invalid MMSE-PIC controls")
    return dict(iterations=iterations, damping=float(damping), variance_floor=float(variance_floor))


def mmse_pic(gram, matched, backend, *, iterations=10, damping=.5, variance_floor=1e-10):
    """Return QPSK posterior means; support arbitrary leading batch axes.

    G = H^H R_n^-1 H and g = H^H R_n^-1 y. The Hermitian solve below is
    algebraically (I + G V)^-1 [G, g-G m], without forming receive-space
    inverses. Its diagonal gains give the unbiased per-symbol observation
    z=m+u/k and EXTRINSIC complex variance 1/k-v. Uniform QPSK priors are
    used in the scalar denoiser; feedback is damped between iterations.
    """
    controls(iterations=iterations, damping=damping, variance_floor=variance_floor)
    t = backend.torch
    xp = t if t else np
    clip = (lambda x, lo, hi: t.clamp(x, min=lo, max=hi)) if t else np.clip
    diagonal = (lambda x: x.diagonal(dim1=-2, dim2=-1)) if t else (lambda x: np.diagonal(x, axis1=-2, axis2=-1))
    concatenate = (lambda xs: t.cat(xs, dim=-1)) if t else (lambda xs: np.concatenate(xs, axis=-1))
    solve = t.linalg.solve if t else np.linalg.solve
    n = matched.shape[-1]
    if gram.shape != (*matched.shape, n) or n < 1:
        raise ValueError("MMSE-PIC likelihood dimensions disagree")
    if not bool(xp.isfinite(gram).all()) or not bool(xp.isfinite(matched).all()):
        raise ValueError("Nonfinite MMSE-PIC likelihood")
    identity = backend.array(np.eye(n, dtype=np.complex128))
    mean = xp.zeros_like(matched)
    variance = xp.ones_like(matched.real)
    amplitude = np.sqrt(.5)
    for _ in range(iterations):
        root = xp.sqrt(variance)
        residual = matched-(gram@mean[..., None])[..., 0]
        matrix = identity+root[..., :, None]*gram*root[..., None, :]
        rhs = root[..., :, None]*concatenate((gram, residual[..., None]))
        filtered = solve(matrix, rhs)/root[..., :, None]
        gain = diagonal(filtered[..., :n]).real
        observable = gain > 0
        safe_gain = xp.where(observable, gain, xp.ones_like(gain))
        z = mean+filtered[..., -1]/safe_gain
        noise = clip(1/safe_gain-variance, variance_floor, float("inf"))
        proposal = amplitude*(xp.tanh(2*amplitude*z.real/noise)
                              +1j*xp.tanh(2*amplitude*z.imag/noise))
        proposal = xp.where(observable, proposal, xp.zeros_like(proposal))
        mean = (1-damping)*mean+damping*proposal
        # A mixture of unit-energy QPSK distributions still has E|x|^2=1.
        variance = clip(1-mean.real**2-mean.imag**2, variance_floor, 1.)
    if not bool(xp.isfinite(proposal).all()):
        raise FloatingPointError("Nonfinite MMSE-PIC estimate")
    return proposal


def equalize_batch(batch, backend, *, whitener, **parameters):
    gram, matched = batch_likelihood(batch, whitener, backend)
    symbols = mmse_pic(gram, matched, backend, **parameters)
    return EqualizerResult(backend.numpy(symbols), np.zeros(len(batch.variances), dtype=np.int64))
