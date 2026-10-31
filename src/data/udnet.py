"""Label-free, double-precision local statistics on physical FFT positions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import cholesky, solve_triangular


FEATURES = ("gram", "matched", "zf", "data_mask")
PREPROCESSING = dict(window_size=32, stride=16, context=8, boundary="zero_padding",
                     whitening="local_principal_unit_rf_covariance", snr_scaling=False,
                     power_normalization="per_window_mean_active_gram_diagonal",
                     zero_power_policy="unit_scale",
                     real_order="all_real_then_all_imag", pinv_rtol=1e-12,
                     precision="complex128/float64_to_float32", outside_ici="truncated")


@dataclass(frozen=True)
class WindowPlan:
    positions: np.ndarray
    valid: np.ndarray
    data_mask: np.ndarray
    nfft: int
    data_indices: np.ndarray

    @classmethod
    def build(cls, nfft, data_indices):
        indices = np.asarray(data_indices, dtype=np.int64)
        if (nfft < 1 or indices.ndim != 1 or not len(indices)
                or np.any(indices < 0) or np.any(indices >= nfft)
                or not np.array_equal(indices, np.unique(indices))):
            raise ValueError("Need sorted unique physical data indices")
        data = np.zeros(nfft, dtype=bool)
        data[indices] = True
        centers = [start for start in range(0, nfft, 16) if data[start:min(start+16, nfft)].any()]
        positions = np.array(centers)[:, None]-8+np.arange(32)
        valid = (positions >= 0) & (positions < nfft)
        mask = valid & data[np.clip(positions, 0, nfft-1)]
        plan = cls(positions, valid, mask, nfft, indices)
        central = positions[:, 8:24][mask[:, 8:24]]
        if not np.array_equal(central, indices):
            raise ValueError("Window centers must cover every data carrier exactly once")
        return plan

    def stitch(self, values):
        """[frame, window, 32, ...] -> [frame, physical data carrier, ...]."""
        if values.shape[1:3] != self.positions.shape:
            raise ValueError("Window output dimensions do not match the plan")
        return values[:, :, 8:24][:, self.data_mask[:, 8:24]]


class LocalPreprocessor:
    def __init__(self, plan, unit_covariance, device="cpu"):
        self.plan, self.device = plan, device
        covariance = np.asarray(unit_covariance, dtype=np.complex128)
        if covariance.shape != (plan.nfft, plan.nfft) or not np.isfinite(covariance).all():
            raise ValueError("Invalid unit RF covariance")
        factors = []
        for positions, valid in zip(plan.positions, plan.valid):
            selected = np.clip(positions, 0, plan.nfft-1)
            local = covariance[np.ix_(selected, selected)].copy()
            local[~valid, :] = 0
            local[:, ~valid] = 0
            local[np.diag_indices(32)] += ~valid
            factor = cholesky((local+local.conj().T)*.5, lower=True)
            factors.append(solve_triangular(factor, np.eye(32), lower=True))
        self.whitener = torch.as_tensor(np.stack(factors), dtype=torch.complex128, device=device)
        self.positions = torch.as_tensor(np.clip(plan.positions, 0, plan.nfft-1), device=device)
        self.valid = torch.as_tensor(plan.valid, device=device)
        self.mask = torch.as_tensor(plan.data_mask, device=device)
        # Periodic carrier masks share one reduced-column SVD batch.
        patterns, inverse = np.unique(plan.data_mask, axis=0, return_inverse=True)
        self.groups = [(torch.as_tensor(np.flatnonzero(inverse == i), device=device),
                        torch.as_tensor(np.flatnonzero(mask), device=device))
                       for i, mask in enumerate(patterns)]

    @torch.no_grad()
    def __call__(self, batch):
        """Whiten H/y, then normalize G/u by the same label-free window power."""
        h = torch.as_tensor(batch.channel, dtype=torch.complex128, device=self.device)
        y = torch.as_tensor(batch.cancelled_observations, dtype=torch.complex128, device=self.device)
        p, valid, mask = self.positions, self.valid, self.mask
        local = h[:, p[:, :, None], p[:, None, :]]
        local = local*valid[None, :, :, None]*mask[None, :, None, :]
        observation = y[:, p]*valid
        local = self.whitener@local
        observation = (self.whitener@observation.unsqueeze(-1)).squeeze(-1)
        transposed = local.conj().transpose(-1, -2)
        gram = transposed@local
        matched = (transposed@observation.unsqueeze(-1)).squeeze(-1)
        zf = torch.zeros_like(matched)
        for windows, columns in self.groups:
            if not len(columns):
                continue
            reduced = local[:, windows][..., columns]
            solution = (torch.linalg.pinv(reduced, rtol=PREPROCESSING["pinv_rtol"])
                        @observation[:, windows].unsqueeze(-1)).squeeze(-1)
            zf[:, windows[:, None], columns] = solution
        # Receiver gain can differ by orders of magnitude between channels.
        # A common positive scale preserves the local least-squares solution,
        # but keeps G*s and H^H*y commensurate with the O(1) symbol input.
        # Average only active data columns: pilots/padding must not dilute it.
        power = gram.diagonal(dim1=-2, dim2=-1).real.sum(-1)/mask.sum(-1)
        if not torch.isfinite(power).all() or torch.any(power < 0):
            raise FloatingPointError("Invalid local UDNet channel power")
        scale = torch.where(power > 0, power, torch.ones_like(power))
        gram = gram/scale[..., None, None]
        matched = matched/scale[..., None]
        real_gram = torch.cat((torch.cat((gram.real, -gram.imag), dim=-1),
                               torch.cat((gram.imag, gram.real), dim=-1)), dim=-2)
        result = dict(gram=real_gram.flatten(0, 1),
                      matched=torch.cat((matched.real, matched.imag), dim=-1).flatten(0, 1),
                      zf=torch.cat((zf.real, zf.imag), dim=-1).flatten(0, 1),
                      data_mask=mask.expand(len(h), -1, -1).flatten(0, 1))
        arrays = {name: value.cpu().numpy().astype(bool if name == "data_mask" else np.float32)
                  for name, value in result.items()}
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError("Nonfinite local UDNet statistics")
        return arrays
