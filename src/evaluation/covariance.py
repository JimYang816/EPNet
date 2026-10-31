"""Train-only robust covariance, independent of detector iteration controls."""
from __future__ import annotations

import logging

import h5py
import numpy as np

from data_generation.src.transmitter.ofdm import qpsk
from src.common.numerics import hermitian

LOG = logging.getLogger(__name__)


def shrink_second_moment(samples):
    """Uncentered covariance with training-estimated diagonal shrinkage."""
    x = np.asarray(samples, np.complex128)
    if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Need at least two finite training residuals")
    count = len(x)
    covariance = hermitian(x.T@x.conj()/count)
    powers = abs(x)**2
    off_energy = max(float(np.sum(abs(covariance)**2)-np.sum(abs(np.diag(covariance))**2)), 0)
    fourth_off = float(np.mean(powers.sum(axis=1)**2-np.sum(powers**2, axis=1)))
    numerator = max((fourth_off-off_energy)/count, 0)
    shrinkage = float(np.clip(numerator/off_energy, 0, 1)) if off_energy else 1.0
    covariance *= 1-shrinkage
    covariance[np.diag_indices_from(covariance)] = powers.mean(axis=0)
    return covariance, shrinkage


def dpss_predict(coefficients, sampled_basis, symbols, backend):
    b = backend
    if b.torch:
        t = b.torch
        coeff, basis, x = b.array(coefficients), b.array(sampled_basis.astype(complex)), b.array(symbols)
        taps = t.einsum("dnq,bql->bdnl", basis, coeff)
        xt = t.fft.ifft(t.fft.ifftshift(x, dim=-1), norm="ortho")
        n, length = x.shape[-1], coeff.shape[-1]
        indices = (t.arange(n, device="cuda")[:, None]-t.arange(length, device="cuda")) % n
        yt = (taps*xt[..., indices]).sum(dim=-1)
        return b.numpy(t.fft.fftshift(t.fft.fft(yt, norm="ortho"), dim=-1))
    taps = np.einsum("dnq,bql->bdnl", sampled_basis, coefficients, optimize=True)
    xt = np.fft.ifft(np.fft.ifftshift(symbols, axes=-1), norm="ortho")
    n, length = symbols.shape[-1], coefficients.shape[-1]
    indices = (np.arange(n)[:, None]-np.arange(length)) % n
    yt = np.sum(taps*xt[..., indices], axis=-1)
    return np.fft.fftshift(np.fft.fft(yt, norm="ortho"), axes=-1)


def training_count(dataset):
    with h5py.File(dataset.path.parent/"train.h5", "r") as train:
        if (train.attrs.get("split") != "train" or not train.attrs.get("complete")
                or train.attrs.get("model_identity") != dataset._h.attrs["model_identity"]
                or not np.all(train["samples/complete"][:])):
            raise ValueError("Robust covariance requires complete matching training data")
        if set(train["windows/record"][:].tolist()) & set(dataset._h["windows/record"][:].tolist()):
            raise ValueError("Training/test record leakage")
        return len(train["samples/window_index"])


def calibrate(dataset, entry, backend, batch_size):
    """Commit one SNR covariance at a time to a separately keyed cache entry."""
    count = training_count(dataset)
    n = dataset._config["nfft"]
    with h5py.File(dataset.path.parent/"train.h5", "r") as train:
        basis = dataset._model.sampled_basis.reshape(4, n, -1)[[1, 2]]
        indices, known = dataset._indices, dataset._known[[1, 2]]
        for si, snr in enumerate(dataset.snr_db):
            if entry.complete[si]:
                continue
            LOG.info("%s training covariance SNR=%g, %d frames", dataset.channel, snr, count)
            residuals = np.empty((count, 2, n), complex)
            alpha = float(dataset._h["compensation/alpha"][si])
            for start in range(0, count, batch_size):
                stop = min(start+batch_size, count)
                x = np.broadcast_to(known, (stop-start, 2, n)).copy()
                x[..., indices] = qpsk(train["samples/bits"][start:stop])
                coefficients = train["samples/raw_coefficients"][si, start:stop].astype(complex)
                raw_y = dpss_predict(coefficients, basis, x, backend)
                clean = train["samples/clean_observations"][start:stop][:, [1, 2]]
                model_residual = train["samples/model_residual"][start:stop]
                residual = model_residual+alpha*(clean-model_residual-raw_y)
                scale = train["samples/rf_noise_variance"][si, start:stop]*10**(float(snr)/10)
                residuals[start:stop] = residual/np.sqrt(scale[:, None, None])
            covariances, shrinkage = zip(*(shrink_second_moment(residuals[:, d]) for d in range(2)))
            entry.commit((si,), error_covariance_unit=np.array(covariances), shrinkage=np.array(shrinkage))
