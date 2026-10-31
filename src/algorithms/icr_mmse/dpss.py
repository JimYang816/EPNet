"""Exact product-convolution representation of the saved raw DPSS channel.

All FFTs on symbols are orthonormal. Coefficient spectra use the ordinary
FFT. Frequency vectors/entries at the public boundary use fftshift order.
Only the additive correction in BandChannel has non-circular support.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DPSSReference:
    """Exact full-reference sufficient statistics, used only by supervision.

    Projection is an inner product, not a fitted/truncated reference channel.
    energy64 retains energy outside the DPSS/delay subspace. The denominator
    uses the original reference precision, just like full-matrix NMSE.
    """

    projection: torch.Tensor
    energy64: torch.Tensor
    denominator: torch.Tensor
    local: torch.Tensor

    @classmethod
    def from_dense(cls, reference, basis, length, bandwidth):
        n = basis.size
        ref = reference.to(torch.complex128)
        circular = torch.fft.fft(torch.fft.ifft(torch.fft.ifftshift(ref, dim=(-2, -1)),
                                               dim=-2, norm="ortho"), dim=-1, norm="ortho")
        rows = torch.arange(n, device=reference.device)[:, None]
        columns = (rows - torch.arange(length, device=reference.device)) % n
        projection = torch.einsum("nq,bnl->bql", basis.values.to(torch.complex128).conj(),
                                   circular[:, rows, columns])
        return cls(projection, ref.abs().square().sum((-2, -1)),
                   reference.abs().square().sum((-2, -1)), base_local(reference, bandwidth))

    def map_tensors(self, fn):
        return type(self)(*(fn(value) for value in (
            self.projection, self.energy64, self.denominator, self.local)))

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in (
            self.projection, self.energy64, self.denominator, self.local))


def reference_energy(reference):
    return (reference.denominator if isinstance(reference, DPSSReference)
            else reference.abs().square().sum((-2, -1)))


class DPSSBasis:
    """One saved, sampled D1/D2 basis; runtime caches are shared across batches."""

    def __init__(self, values):
        if values.ndim != 2 or not torch.isfinite(values).all():
            raise ValueError("Expected a finite [N,Q] sampled DPSS basis")
        self.values = values
        self.size, self.count = values.shape
        self.spectral = torch.fft.fft(values.T, dim=-1) / self.size
        self.gram = values.mH @ values
        self.frequencies = torch.fft.fftshift(torch.arange(self.size, device=values.device))
        self._weight_cache = None

    def weighted_kernel(self, weights):
        """Exact M[q,r,j] = sum_k w[k] conj(a_q[k-j]) a_r[k-j]."""
        if weights.ndim != 1 or weights.shape != (self.size,):
            raise ValueError("DPSS row weights must be a shared vector [N]")
        cacheable = not weights.requires_grad and not self.values.requires_grad
        cached = self._weight_cache
        if cacheable and cached is not None and torch.equal(cached[0], weights):
            return cached[1]
        # Correlation, not convolution: h[j] = sum_d w[j+d] s[d].
        products = self.spectral.conj()[:, None, :] * self.spectral[None, :, :]
        transformed = torch.fft.fft(products.conj(), dim=-1).conj()
        kernel = torch.fft.ifft(torch.fft.fft(torch.fft.ifftshift(weights))[None, None, :]
                                 * transformed, dim=-1)
        if cacheable:
            self._weight_cache = (weights.detach().clone(), kernel.detach())
        return kernel


class DPSSChannel:
    """Complete raw channel, without a dense per-sample N x N allocation."""

    ndim = 3

    def __init__(self, coefficients, basis, *, spectrum=None):
        if (coefficients.ndim != 3 or not coefficients.is_complex()
                or coefficients.shape[1] != basis.count
                or not 0 < coefficients.shape[2] <= basis.size
                or coefficients.dtype != basis.values.dtype
                or coefficients.device != basis.values.device):
            raise ValueError("Expected complex [batch,Q,L] coefficients and matching sampled basis")
        self.coefficients, self.basis = coefficients, basis
        self.spectrum = (torch.fft.fft(coefficients, n=basis.size, dim=-1)
                         if spectrum is None else spectrum)
        self.shape = (len(coefficients), basis.size, basis.size)
        self.dtype, self.device = coefficients.dtype, coefficients.device
        self._locals, self._columns = {}, None
        self._energy, self._reference = None, None

    def is_complex(self):
        return True

    def finite(self):
        return torch.isfinite(self.coefficients).all() & torch.isfinite(self.spectrum).all()

    def clear_sample_cache(self):
        """Repeat timing must include per-sample work on every forward pass."""
        self._locals, self._columns = {}, None
        self._energy, self._reference = None, None

    def to(self, *, dtype):
        if dtype == self.dtype:
            return self
        basis = DPSSBasis(self.basis.values.to(dtype=dtype))
        return DPSSChannel(self.coefficients.to(dtype=dtype), basis)

    def scaled(self, scale):
        divisor = scale[:, None, None]
        result = DPSSChannel(self.coefficients / divisor, self.basis,
                             spectrum=self.spectrum / divisor)
        result._locals = {width: values / divisor for width, values in self._locals.items()}
        return result

    def matvec(self, value):
        natural = torch.fft.ifftshift(value, dim=-1)
        filtered = torch.fft.ifft(self.spectrum * natural[:, None, :], dim=-1, norm="ortho")
        time = (self.basis.values.T[None] * filtered).sum(-2)
        return torch.fft.fftshift(torch.fft.fft(time, dim=-1, norm="ortho"), dim=-1)

    def adjoint(self, value):
        time = torch.fft.ifft(torch.fft.ifftshift(value, dim=-1), dim=-1, norm="ortho")
        transformed = torch.fft.fft(self.basis.values.T.conj()[None] * time[:, None, :],
                                     dim=-1, norm="ortho")
        return torch.fft.fftshift((self.spectrum.conj() * transformed).sum(-2), dim=-1)

    def local_entries(self, bandwidth):
        cached = self._locals.get(bandwidth)
        if cached is None:
            n = self.basis.size
            offsets = torch.arange(-bandwidth, bandwidth + 1, device=self.device)
            columns = torch.arange(n, device=self.device)[:, None] + offsets
            valid = (columns >= 0) & (columns < n)
            gathered = self.spectrum[:, :, self.basis.frequencies[columns.clamp(0, n - 1)]]
            cached = torch.einsum("qw,bqnw->bnw", self.basis.spectral[:, (-offsets) % n], gathered)
            cached = cached * valid
            self._locals[bandwidth] = cached
        return cached

    def frobenius_energy(self):
        if self._energy is None:
            self._energy = torch.einsum("bql,qr,brl->b", self.coefficients.conj(),
                                        self.basis.gram, self.coefficients).real.clamp_min(0)
        return self._energy

    def column_energy(self, row_weights=None):
        weights = (torch.ones(self.basis.size, dtype=self.coefficients.real.dtype, device=self.device)
                   if row_weights is None else row_weights)
        # The receiver covariance is shared, but support batched weights too.
        if weights.ndim == 2:
            return torch.cat([DPSSChannel(self.coefficients[i:i+1], self.basis,
                                           spectrum=self.spectrum[i:i+1]).column_energy(weights[i])
                              for i in range(len(self.coefficients))])
        if self._columns is not None and torch.equal(self._columns[0], weights):
            return self._columns[1]
        kernel = self.basis.weighted_kernel(weights)
        energy = torch.einsum("bqn,qrn,brn->bn", self.spectrum.conj(), kernel, self.spectrum).real
        energy = torch.fft.fftshift(energy, dim=-1).clamp_min(0)
        if not weights.requires_grad:
            self._columns = (weights.detach().clone(), energy)
        return energy

    def error_energy(self, reference):
        if self._reference is not None and self._reference[0] is reference:
            return self._reference[1]
        n, length = self.basis.size, self.coefficients.shape[-1]
        # Accumulate a potentially small difference of energies in float64.
        basis = self.basis.values.to(torch.complex128)
        coefficients = self.coefficients.to(torch.complex128)
        if isinstance(reference, DPSSReference):
            cross, reference_power = reference.projection, reference.energy64
        else:
            ref = reference.to(torch.complex128)
            circular = torch.fft.fft(torch.fft.ifft(torch.fft.ifftshift(ref, dim=(-2, -1)),
                                                   dim=-2, norm="ortho"), dim=-1, norm="ortho")
            rows = torch.arange(n, device=self.device)[:, None]
            columns = (rows - torch.arange(length, device=self.device)) % n
            cross = torch.einsum("nq,bnl->bql", basis.conj(), circular[:, rows, columns])
            reference_power = ref.abs().square().sum((-2, -1))
        raw = torch.einsum("bql,qr,brl->b", coefficients.conj(), basis.mH @ basis, coefficients).real
        overlap = (coefficients.conj() * cross).sum((-2, -1)).real
        energy = (raw + reference_power - 2 * overlap).clamp_min(0)
        energy = energy.to(self.coefficients.real.dtype)
        self._reference = (reference, energy)
        return energy

    def dense(self):
        """Diagnostic materialization only; inference never calls this method."""
        n = self.basis.size
        rows = torch.arange(n, device=self.device)[:, None]
        cols = torch.arange(n, device=self.device)[None, :]
        return torch.einsum("qkj,bqj->bkj", self.basis.spectral[:, (rows-cols) % n],
                            self.spectrum[:, :, self.basis.frequencies])


def base_local(base, bandwidth):
    if isinstance(base, DPSSReference):
        if base.local.shape[-1] != 2 * bandwidth + 1:
            raise ValueError("Reference statistics have a different correction bandwidth")
        return base.local
    if isinstance(base, DPSSChannel):
        return base.local_entries(bandwidth)
    from .operators import band_indices
    columns, valid = band_indices(base.shape[-1], bandwidth, base.device)
    rows = torch.arange(base.shape[-1], device=base.device)[:, None]
    return base[:, rows, columns] * valid


def base_energy(base):
    return base.frobenius_energy() if isinstance(base, DPSSChannel) else base.abs().square().sum((-2, -1))


def base_error(base, reference):
    return base.error_energy(reference) if isinstance(base, DPSSChannel) else (base-reference).abs().square().sum((-2, -1))
