"""Real EPNet, DOI 10.1109/TWC.2020.3030882, equations (8)--(14), (20).

Inputs are likelihood sufficient statistics after pilot cancellation/whitening.
No labels, reference CSI, noise inflation, or learned matrix weights enter here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass
class EPNetOutput:
    extrinsic_mean: torch.Tensor       # [..., layer, Re/Im coordinate]
    extrinsic_variance: torch.Tensor
    rejected_sites: torch.Tensor      # [...], sum over layers/coordinates

    @property
    def symbols(self):
        value = self.extrinsic_mean[..., -1, :]
        n = value.shape[-1]//2
        return torch.complex(value[..., :n], value[..., n:])

    @property
    def bit_llrs(self):
        """QPSK log P(bit=0)/P(bit=1), ordered [carrier, real/imag bit]."""
        natural = math.sqrt(2)*self.extrinsic_mean[..., -1, :]/self.extrinsic_variance[..., -1, :]
        n = natural.shape[-1]//2
        return torch.stack((natural[..., :n], natural[..., n:]), dim=-1)


class EPNet(nn.Module):
    def __init__(self, layers=5, initial_damping=.1, variance_floor=1e-8, initial_precision=.5):
        super().__init__()
        if (not isinstance(layers, int) or isinstance(layers, bool) or layers < 1
                or not math.isfinite(initial_damping) or not 0 < initial_damping < 1
                or not math.isfinite(variance_floor) or variance_floor <= 0
                or not math.isfinite(initial_precision) or initial_precision <= 0):
            raise ValueError("Invalid EPNet controls")
        self.config = dict(layers=layers, initial_damping=initial_damping,
                           variance_floor=variance_floor, initial_precision=initial_precision)
        self.beta = nn.Parameter(torch.full((layers,), math.log(initial_damping/(1-initial_damping)),
                                            dtype=torch.float64))

    @property
    def damping(self):
        return self.beta.sigmoid()

    def forward(self, gram, matched):
        """L linear solves, marginal cavities, discrete moments, natural damping.

        Literal paper initialization Lambda=1/(2*Es), with Es=1 here. The
        physically moment-matched QPSK initialization would be 2/Es; callers
        may explicitly request that variant with initial_precision=2.
        The last output is the L-th cavity, as in Fig. 2 / eq. (20), BEFORE
        the final site update. Consequently beta[L-1] has zero loss gradient.
        """
        if (gram.ndim < 2 or matched.ndim != gram.ndim-1
                or gram.shape[:-1] != matched.shape or gram.shape[-1] != matched.shape[-1]
                or matched.shape[-1] % 2 or matched.shape[-1] == 0
                or gram.dtype != self.beta.dtype or matched.dtype != self.beta.dtype
                or gram.device != self.beta.device or matched.device != self.beta.device
                or not torch.isfinite(gram).all() or not torch.isfinite(matched).all()):
            raise ValueError("EPNet requires finite real likelihood statistics of matching shape, dtype and device")
        precision = torch.full_like(matched, self.config["initial_precision"])
        natural = torch.zeros_like(matched)
        rejected = torch.zeros(matched.shape[:-1], dtype=torch.int64, device=matched.device)
        means, variances = [], []
        amplitude = math.sqrt(.5)
        for damping in self.damping.unbind():
            factor = torch.linalg.cholesky(gram+torch.diag_embed(precision))
            covariance = torch.cholesky_inverse(factor)
            mean = torch.cholesky_solve((matched+natural).unsqueeze(-1), factor).squeeze(-1)
            marginal = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-15)
            cavity_precision = marginal.reciprocal()-precision
            cavity_natural = mean/marginal-natural
            valid = (cavity_precision > 1e-12) & torch.isfinite(cavity_precision) & torch.isfinite(cavity_natural)
            safe_precision = torch.where(valid, cavity_precision, torch.full_like(precision, 1e-12))
            safe_natural = torch.where(valid, cavity_natural, torch.zeros_like(natural))
            cavity_mean = safe_natural/safe_precision
            means.append(cavity_mean)
            variances.append(safe_precision.reciprocal())
            # Exact uniform +/-1/sqrt(2) moments in eq. (11); stable log-domain
            # Bernoulli formula, equivalent to explicitly enumerating the alphabet.
            tilted_mean = amplitude*torch.tanh(amplitude*safe_natural)
            tilted_variance = (.5-tilted_mean.square()).clamp_min(self.config["variance_floor"])
            proposal_precision = tilted_variance.reciprocal()-safe_precision
            proposal_natural = tilted_mean/tilted_variance-safe_natural
            accept = (valid & (proposal_precision > 0) & torch.isfinite(proposal_precision)
                      & torch.isfinite(proposal_natural))
            rejected = rejected+(~accept).sum(dim=-1)
            proposal_precision = torch.where(accept, proposal_precision, precision)
            proposal_natural = torch.where(accept, proposal_natural, natural)
            # Equations (12), (14): damp precisions/natural means, not moments.
            precision = (1-damping)*precision+damping*proposal_precision
            natural = (1-damping)*natural+damping*proposal_natural
        result = EPNetOutput(torch.stack(means, dim=-2), torch.stack(variances, dim=-2), rejected)
        if not torch.isfinite(result.extrinsic_mean).all():
            raise FloatingPointError("EPNet produced nonfinite extrinsic means")
        return result


def layerwise_loss(output, symbols):
    """Eq. (20): mean over layers/examples, SUM over real symbol coordinates."""
    target = torch.cat((symbols.real, symbols.imag), dim=-1)
    if target.shape != output.extrinsic_mean.shape[:-2]+output.extrinsic_mean.shape[-1:]:
        raise ValueError("EPNet supervision shape mismatch")
    return (output.extrinsic_mean-target.unsqueeze(-2)).square().sum(dim=-1).mean()
