"""One hard-decision, reduced-rank MAP channel update between two LMMSE solves."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.algorithms.interface import EqualizerBatch, EqualizerResult
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


def equalize_batch(batch, backend, *, whitener, known, modes, prior_scale):
    """Infer from H0, the noisy received block, public tones and receiver prior."""
    xp = backend.torch or np
    initial = full_lmmse(batch, backend, whitener=whitener).symbols
    hard = (np.where(initial.real < 0, -1, 1)+1j*np.where(initial.imag < 0, -1, 1))/np.sqrt(2)
    symbols = np.broadcast_to(known, (len(initial), len(known))).astype(np.complex128, copy=True)
    symbols[:, batch.data_indices] = hard
    symbols = backend.array(symbols)
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
