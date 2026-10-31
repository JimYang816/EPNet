"""Complex OAMP-Net2, He et al., TSP 2020, equations (19), (24)--(28).

Thermal whitening gives R_n=I. A full Gram eigendecomposition evaluates the
LMMSE filter exactly without repeating a dense inverse at every layer.
There is no window truncation, CSI refinement, or oracle error covariance.
"""
from __future__ import annotations

import math

import torch
from torch import nn


FEATURES = ("eigenvalues", "eigenvectors", "projected_matched", "observation_energy", "receive_count")


class OAMPNet2(nn.Module):
    def __init__(self, layers=10, variance_floor=5e-13):
        super().__init__()
        if (not isinstance(layers, int) or isinstance(layers, bool) or layers < 1
                or not math.isfinite(variance_floor) or not 0 < variance_floor < 1):
            raise ValueError("Invalid OAMP-Net2 configuration")
        self.config = dict(layers=layers, variance_floor=variance_floor)
        # Exactly four unrestricted, layer-specific real scalars (4T total).
        self.gamma = nn.Parameter(torch.ones(layers, dtype=torch.float64))
        self.theta = nn.Parameter(torch.ones(layers, dtype=torch.float64))
        self.phi = nn.Parameter(torch.ones(layers, dtype=torch.float64))
        self.xi = nn.Parameter(torch.zeros(layers, dtype=torch.float64))

    def forward(self, eigenvalues, eigenvectors, projected_matched,
                observation_energy, receive_count, *, return_layers=False):
        n = eigenvalues.shape[-1]
        if (n < 1 or eigenvectors.shape != (*eigenvalues.shape, n)
                or projected_matched.shape != eigenvalues.shape
                or observation_energy.shape != eigenvalues.shape[:-1]
                or receive_count.shape != observation_energy.shape):
            raise ValueError("OAMP-Net2 spectral feature dimensions disagree")
        eps = self.config["variance_floor"]
        trace = eigenvalues.sum(dim=-1)
        observable = trace > 0
        safe_trace = torch.where(observable, trace, torch.ones_like(trace))
        estimate = torch.zeros_like(projected_matched)
        adjoint = eigenvectors.mH
        layers = []
        for layer in range(self.config["layers"]):
            spectral = (adjoint@estimate.unsqueeze(-1)).squeeze(-1)
            residual_energy = (observation_energy
                -2*(spectral.conj()*projected_matched).real.sum(dim=-1)
                +(eigenvalues*spectral.abs().square()).sum(dim=-1))
            variance = ((residual_energy-receive_count)/safe_trace).clamp_min(eps)
            filt = variance.unsqueeze(-1)/(1+variance.unsqueeze(-1)*eigenvalues)
            trace_wh = (filt*eigenvalues).sum(dim=-1)
            safe_wh = torch.where(observable, trace_wh, torch.ones_like(trace_wh))
            scale = torch.where(observable, n/safe_wh, torch.zeros_like(trace_wh))
            filt = scale.unsqueeze(-1)*filt
            r = estimate+self.gamma[layer]*(eigenvectors@(
                filt*(projected_matched-eigenvalues*spectral)).unsqueeze(-1)).squeeze(-1)
            # theta (not gamma) defines C=I-theta W H in equation (27).
            c = 1-self.theta[layer]*filt*eigenvalues
            tau = ((c.square().sum(dim=-1)*variance
                +self.theta[layer].square()*(filt.square()*eigenvalues).sum(dim=-1))/n).clamp_min(eps)
            amplitude = math.sqrt(.5)
            posterior = amplitude*(torch.tanh(2*amplitude*r.real/tau.unsqueeze(-1))
                                  +1j*torch.tanh(2*amplitude*r.imag/tau.unsqueeze(-1)))
            estimate = self.phi[layer]*(posterior-self.xi[layer]*r)
            layers.append(estimate)
        return torch.stack(layers, dim=-2) if return_layers else estimate
