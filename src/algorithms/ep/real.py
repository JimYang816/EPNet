"""Real-valued QPSK EP, Ghavami & Naraghi-Pour (2017), equations 23--34.

DOI: 10.1109/TVT.2017.2683527. This is the ordinary EP implementation.
Site mean/variance damping follows equations 33--34, rather than damping
natural parameters. Complex covariance is converted with its imaginary blocks.
"""
from __future__ import annotations

import numpy as np

from src.algorithms.interface import EqualizerResult
from src.algorithms.likelihood import batch_real_likelihood, real_likelihood


def paper_ep(gram, matched, backend, *, iterations=10, damping=.2, epsilon=5e-7):
    """Batched equations 23--34; all arrays stay on the selected CPU/GPU.

    gram/matched are real likelihood parameters, with optional batch axes.
    Initial site moments are zero and Es/2=.5. Invalid site variances retain
    their previous values. The iteration budget is fixed for this call.
    """
    if (not isinstance(iterations, int) or iterations < 1 or not 0 < damping < 1
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("Invalid paper EP controls")
    b, torch = backend, backend.torch
    if torch:
        xp = torch
        diagonal = lambda x: x.diagonal(dim1=-2, dim2=-1)
        inverse = lambda x: torch.cholesky_inverse(torch.linalg.cholesky(x))
        finite = torch.isfinite
        maximum = lambda x, v: torch.clamp(x, min=v)
        diag_embed = torch.diag_embed
    else:
        xp = np
        diagonal = lambda x: np.diagonal(x, axis1=-2, axis2=-1)
        inverse = np.linalg.inv
        finite = np.isfinite
        maximum = np.maximum
        diag_embed = lambda x: x[..., :, None]*np.eye(x.shape[-1])
    site_mean = matched*0
    site_variance = site_mean+.5
    rejected = xp.zeros_like(site_mean[..., 0])
    amplitude = np.sqrt(.5)
    for _ in range(iterations):
        covariance = inverse(gram+diag_embed(1/site_variance))
        mean = (covariance @ (matched+site_mean/site_variance)[..., None])[..., 0]
        marginal = maximum(diagonal(covariance), 1e-15)
        cavity_precision = 1/marginal-1/site_variance
        valid = finite(cavity_precision) & (cavity_precision > 1e-14)
        safe_precision = xp.where(valid, cavity_precision, cavity_precision*0+1)
        cavity_variance = 1/safe_precision
        cavity_mean = cavity_variance*(mean/marginal-site_mean/site_variance)
        tilted_mean = amplitude*xp.tanh(amplitude*cavity_mean/cavity_variance)
        tilted_variance = maximum(.5-tilted_mean**2, 1e-15)
        proposal_precision = 1/tilted_variance-safe_precision
        proposal_variance = 1/xp.where(proposal_precision > 0, proposal_precision, proposal_precision*0+1)
        proposal_mean = proposal_variance*(tilted_mean/tilted_variance-cavity_mean/cavity_variance)
        accept = (valid & (proposal_precision > 0) & finite(proposal_variance)
                  & (proposal_variance >= epsilon) & finite(proposal_mean))
        rejected = rejected+(~accept).sum(axis=-1)
        site_variance = xp.where(accept, damping*proposal_variance+(1-damping)*site_variance, site_variance)
        site_mean = xp.where(accept, damping*proposal_mean+(1-damping)*site_mean, site_mean)
    precision = gram+diag_embed(1/site_variance)
    natural = matched+site_mean/site_variance
    if torch:
        mean = torch.linalg.solve(precision, natural[..., None])[..., 0]
    else:
        mean = np.linalg.solve(precision, natural[..., None])[..., 0]
    result = b.numpy(mean)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Paper EP produced nonfinite estimates")
    n = result.shape[-1]//2
    return result[..., :n]+1j*result[..., n:], b.numpy(rejected).astype(np.int64)


def equalize_batch(batch, backend, *, whitener, **parameters):
    gram, matched = batch_real_likelihood(batch, whitener, backend)
    symbols, rejected = paper_ep(gram, matched, backend, **parameters)
    return EqualizerResult(symbols, rejected)
