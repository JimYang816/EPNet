"""Reconstruct one label-free comparison batch on the requested device."""
from __future__ import annotations

import numpy as np

from src.algorithms.interface import EqualizerBatch


def prepare_batch(dataset, si, frames, d, backend):
    ds, b = dataset, backend
    h, n = ds._h, ds._config["nfft"]
    variances = h["samples/rf_noise_variance"][si, frames]
    y = (h["samples/clean_observations"][frames, d+1]
         +np.sqrt(variances[:, None])*h["samples/standard_noise"][frames, d+1])
    coefficients = b.array(h["samples/raw_coefficients"][si, frames].astype(complex))
    basis = b.array(ds._model.sampled_basis.reshape(4, n, -1)[d+1].astype(complex))
    taps = basis @ coefficients
    length = coefficients.shape[-1]
    if b.torch:
        t = b.torch
        rows = t.arange(n, device=b.device)[:, None]
        columns = (rows-t.arange(length, device=b.device)) % n
        circular = t.zeros((len(frames), n, n), dtype=t.complex128, device=b.device)
        circular[:, rows, columns] = taps
        raw = t.fft.fftshift(t.fft.ifft(t.fft.fft(circular, dim=-2, norm="ortho"), dim=-1, norm="ortho"), dim=(-2, -1))
    else:
        rows = np.arange(n)[:, None]
        columns = (rows-np.arange(length)) % n
        circular = np.zeros((len(frames), n, n), complex)
        circular[:, rows, columns] = taps
        raw = np.fft.fftshift(np.fft.ifft(np.fft.fft(circular, axis=-2, norm="ortho"), axis=-1, norm="ortho"), axes=(-2, -1))
    # Keep the complete raw matrix and never enter the historical _channels path.
    channel = raw
    if ds.input_view == "compensated":
        window = h["samples/window_index"][frames]
        unique, inverse = np.unique(window, return_inverse=True)
        reference = b.array(np.stack([h["windows/reference_csi"][int(wi), d] for wi in unique]).astype(complex))[inverse]
        alpha = float(h["compensation/alpha"][si])
        channel = reference+alpha*(raw-reference)
    y = b.array(y)
    residual = y-channel@b.array(ds._known[d+1])
    return EqualizerBatch(channel, y, residual, variances, ds._indices,
                          ds._shared["unit_rf_noise_covariance"][:])
