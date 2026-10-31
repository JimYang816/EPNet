"""Full estimated channel plus a differentiable, non-circular band increment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .dpss import DPSSChannel, base_local


def band_indices(size, bandwidth, device=None):
    rows = torch.arange(size, device=device)[:, None]
    columns = rows + torch.arange(-bandwidth, bandwidth + 1, device=device)[None, :]
    valid = (columns >= 0) & (columns < size)
    return columns.clamp(0, size - 1), valid


@dataclass(frozen=True)
class BandChannel:
    """Keep every entry of base; restrict only the additive correction."""

    base: torch.Tensor | DPSSChannel
    bands: torch.Tensor

    def __post_init__(self):
        if (self.base.ndim != 3 or self.base.shape[-1] != self.base.shape[-2]
                or self.bands.ndim != 3 or self.bands.shape[:2] != self.base.shape[:2]
                or self.bands.shape[-1] % 2 != 1 or not self.base.is_complex()
                or self.bands.dtype != self.base.dtype or self.bands.device != self.base.device):
            raise ValueError("Expected complex [batch,N,N] base and [batch,N,2B+1] increments")

    @property
    def bandwidth(self):
        return self.bands.shape[-1] // 2

    def _indices(self):
        return band_indices(self.base.shape[-1], self.bandwidth, self.base.device)

    def local_entries(self):
        _, valid = self._indices()
        return (base_local(self.base, self.bandwidth) + self.bands) * valid

    def matvec(self, value):
        columns, valid = self._indices()
        result = (self.base.matvec(value) if isinstance(self.base, DPSSChannel)
                  else (self.base @ value.unsqueeze(-1)).squeeze(-1))
        return result + (
            self.bands * value[:, columns] * valid).sum(-1)

    def adjoint(self, value):
        columns, valid = self._indices()
        extra = self.bands.conj() * value[..., None] * valid
        result = (self.base.adjoint(value) if isinstance(self.base, DPSSChannel)
                  else (self.base.mH @ value.unsqueeze(-1)).squeeze(-1))
        return result.scatter_add(1, columns.reshape(1, -1).expand(len(value), -1),
                                  extra.reshape(len(value), -1))

    def dense(self):
        columns, valid = self._indices()
        base = self.base.dense() if isinstance(self.base, DPSSChannel) else self.base
        return base.scatter_add(-1, columns[None].expand(len(base), -1, -1),
                                     self.bands * valid)

    def column_energy(self, row_weights=None, *, base_energy=None):
        """Diagonal preconditioner; no Gram matrix or updated dense H needed."""
        columns, valid = self._indices()
        weight = torch.ones_like(self.bands.real[..., 0]) if row_weights is None else row_weights
        if base_energy is None:
            base_energy = (self.base.column_energy(row_weights) if isinstance(self.base, DPSSChannel)
                           else (self.base.abs().square() * weight[..., None]).sum(-2))
        local_base = base_local(self.base, self.bandwidth)
        extra = (2 * (local_base.conj() * self.bands).real + self.bands.abs().square())
        extra = extra * valid * weight[..., None]
        return base_energy.scatter_add(1, columns.reshape(1, -1).expand(len(self.bands), -1),
                                        extra.reshape(len(self.bands), -1)).clamp_min(0)

    def error_energy(self, reference):
        """Exact full-matrix error norm, accumulating only band cross terms."""
        columns, valid = self._indices()
        if isinstance(self.base, DPSSChannel):
            difference = base_local(self.base, self.bandwidth) - base_local(reference, self.bandwidth)
            energy = self.base.error_energy(reference)
        else:
            full_difference = self.base - reference
            difference = base_local(full_difference, self.bandwidth)
            energy = full_difference.abs().square().sum((-2, -1))
        extra = (2 * (difference.conj() * self.bands).real
                 + self.bands.abs().square()) * valid
        return (energy + extra.sum((-2, -1))).clamp_min(0)
