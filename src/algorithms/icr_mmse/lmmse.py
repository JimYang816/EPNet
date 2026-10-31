"""Full-matrix LMMSE with a fixed Gaussian soft-output mapping.

The unit-energy data prior is reset for each solve. Previous decisions affect
this detector only through the current CSI, never through a learned prior.
"""
from __future__ import annotations

import torch
from torch import nn


class FullMatrixLMMSELayer(nn.Module):
    """No trainable detector parameters; gradients still reach corrected CSI."""

    def forward(self, channel, observations, known, mean, variance, data_mask, pilot_mask,
                null_mask, positions, rf_variance, covariance_factor, constellation,
                use_uncertainty=True, covariance_diagonal=None, column_weights=None,
                base_column_energy=None, return_diagnostics=True,
                previous_log_probabilities=None, return_log_probabilities=False):
        if torch.any(rf_variance <= 0):
            raise ValueError("Soft-output LMMSE requires positive RF noise variance")
        matrix = channel.dense()[..., data_mask]
        cancelled = observations - channel.matvec(known)
        if covariance_factor is None:
            if covariance_diagonal is None:
                raise ValueError("LMMSE requires the common noise weighting")
            weighted = matrix / covariance_diagonal[None, :, None]
            weighted_y = cancelled / covariance_diagonal
        else:
            weighted = torch.cholesky_solve(matrix, covariance_factor)
            weighted_y = torch.cholesky_solve(cancelled[..., None], covariance_factor).squeeze(-1)
        gram = matrix.mH @ weighted
        gram = (gram + gram.mH) * .5
        identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
        system = gram + rf_variance[:, None, None] * identity
        factor = torch.linalg.cholesky(system)
        matched = (matrix.mH @ weighted_y[..., None]).squeeze(-1)
        estimate = torch.cholesky_solve(matched[..., None], factor).squeeze(-1)
        # Gaussian-prior posterior covariance V = sigma^2 (H^H W H + sigma^2 I)^-1.
        # The debiased scalar estimate has gain g=1-Vjj and variance Vjj/g.
        # Cancel the symbol-independent squared term to avoid division by g:
        # logits = (2 Re(x_hat * a*) - g |a|^2) / Vjj.
        posterior_variance = rf_variance[:, None] * torch.cholesky_inverse(factor).diagonal(
            dim1=-2, dim2=-1).real
        posterior_variance = posterior_variance.clamp(min=torch.finfo(estimate.real.dtype).eps, max=1.)
        gain = 1 - posterior_variance
        logits = (2 * (estimate[..., None] * constellation.conj()).real
                  - gain[..., None] * constellation.abs().square()) / posterior_variance[..., None]
        data_probabilities = logits.softmax(-1)
        probabilities = observations.real.new_zeros((*observations.shape, len(constellation)))
        probabilities[:, data_mask] = data_probabilities
        next_mean = (probabilities * constellation).sum(-1) + known
        next_variance = (1 - next_mean.abs().square()).clamp_min(0) * data_mask
        diagnostics = dict(pilot_cancelled=cancelled, linear_estimate=estimate,
                           posterior_variance=posterior_variance, gain=gain,
                           residual=observations - channel.matvec(mean)) if return_diagnostics else {}
        result = probabilities, next_mean, next_variance, diagnostics
        if return_log_probabilities:
            logp = observations.real.new_zeros((*observations.shape, len(constellation)))
            logp[:, data_mask] = logits.log_softmax(-1)
            return (*result, logp)
        return result
