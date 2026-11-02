"""Soft-decision data-aided DPSS-LMMSE channel re-estimation (DD-DPSS)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.algorithms.interface import EqualizerBatch, EqualizerResult
from src.algorithms.likelihood import batch_likelihood
from src.algorithms.MMSE.full_matrix import equalize_batch as full_lmmse


def controls(rank=32, prior_scale=0.03):
    if type(rank) is not int or rank < 1 or isinstance(prior_scale, bool) or not isinstance(prior_scale, (float, int)):
        raise ValueError("Invalid DD-CE rank or prior scale")
    if not np.isfinite(prior_scale) or prior_scale <= 0:
        raise ValueError("DD-CE prior scale must be finite and positive")
    return dict(rank=rank, prior_scale=float(prior_scale))


@dataclass(frozen=True)
class ChannelModes:
    time: Any       # nfft x rank, scaled by the square root of each covariance eigenvalue
    delay: Any      # rank x nfft, unit-norm zero-padded delay modes
    spectrum: Any   # frequency response of each delay mode, in stored FFT order

    def responses(self, symbols, xp):
        """Columns of this block's decision-directed observation matrix A."""
        fft = xp.fft
        filtered = fft.ifft(fft.ifftshift(symbols[:, None, :]*self.spectrum, (-1,)), None, -1, "ortho")
        received = self.time * filtered.swapaxes(-1, -2)
        return fft.fftshift(fft.fft(received, None, -2, "ortho"), (-2,))

    def channel_update(self, coefficients, backend):
        """Materialize the full ICI matrix of the fitted time-varying taps."""
        fft = (backend.torch or np).fft
        taps = (self.time*coefficients[:, None, :]) @ self.delay
        positions = backend.array(np.arange(self.time.shape[0]))
        rows = positions[:, None]
        circular = taps[:, rows, (rows-positions) % len(positions)]
        return fft.fftshift(fft.ifft(fft.fft(circular, None, -2, "ortho"), None, -1, "ortho"), (-2, -1))


def build_modes(basis, blocks, rank, backend):
    """Leading tap-covariance modes on this D window of the public receiver prior."""
    n = len(basis)
    q, r = np.linalg.qr(np.asarray(basis, dtype=np.complex128), mode="reduced")
    candidates = []
    for start, stop, temporal, delay in blocks:
        tpower, tvectors = np.linalg.eigh(r @ temporal @ r.conj().T)
        dpower, dvectors = np.linalg.eigh(delay)
        power = np.maximum(tpower, 0)[:, None]*np.maximum(dpower, 0)
        time = q @ tvectors
        for index in np.argsort(power, axis=None)[-rank:]:
            ti, di = np.unravel_index(index, power.shape)
            if power[ti, di] > 0:
                candidates.append((power[ti, di], time[:, ti], dvectors[:, di], start, stop))
    selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:rank]
    if not selected:
        raise ValueError("DD-CE requires a nonzero receiver prior")
    times = np.stack([np.sqrt(power)*v for power, v, _, _, _ in selected], axis=1)
    delays = np.zeros((len(selected), n), np.complex128)
    for i, (_, _, vector, start, stop) in enumerate(selected):
        delays[i, start:stop] = vector
    spectra = np.fft.fftshift(np.fft.fft(delays, axis=-1), axes=-1)
    return ChannelModes(backend.array(times), backend.array(delays), backend.array(spectra))


@dataclass(frozen=True)
class DDResult(EqualizerResult):
    corrected_channel: Any


def _soft_qpsk(batch, backend, whitener):
    """QPSK posterior means under the scalar soft-output LMMSE approximation."""
    xp = backend.torch or np
    gram, matched = batch_likelihood(batch, whitener, backend)
    identity = backend.array(np.eye(len(batch.data_indices), dtype=np.complex128))
    covariance = xp.linalg.inv(gram + identity)
    estimate = (covariance @ matched[..., None])[..., 0]
    variance = covariance.diagonal(0, -2, -1).real.clip(np.finfo(np.float64).eps, 1.)
    # Equal-energy QPSK cancels the common likelihood term. The gain cancels
    # before division, so an unobservable tone has zero mean as well.
    scaled = np.sqrt(2.) * estimate / variance
    return (xp.tanh(scaled.real) + 1j * xp.tanh(scaled.imag)) / np.sqrt(2.)


def equalize_batch(batch, backend, *, whitener, known, modes, prior_scale):
    """Infer from H0, the noisy received block, public tones and receiver prior."""
    xp = backend.torch or np
    soft = _soft_qpsk(batch, backend, whitener)
    symbols = backend.array(np.broadcast_to(known, (len(soft), len(known))).astype(np.complex128, copy=True))
    symbols[:, batch.data_indices] = soft
    residual = batch.raw_observations[..., None] - batch.channel @ symbols[..., None]
    design = whitener @ modes.responses(symbols, xp)
    adjoint = design.conj().swapaxes(-2, -1)
    ridge = backend.array(np.asarray(batch.variances)/prior_scale)[:, None, None]
    eye = backend.array(np.eye(design.shape[-1], dtype=np.complex128))
    weights = xp.linalg.solve(adjoint @ design + ridge*eye, adjoint @ (whitener @ residual))[..., 0]
    corrected = batch.channel + modes.channel_update(weights, backend)
    cancelled = batch.raw_observations - corrected @ backend.array(np.asarray(known, dtype=np.complex128))
    updated = EqualizerBatch(corrected, batch.raw_observations, cancelled, batch.variances,
                             batch.data_indices, batch.noise_covariance)
    final = full_lmmse(updated, backend, whitener=whitener)
    return DDResult(final.symbols, final.rejected_sites, corrected)
