"""Selected-frequency full-frame joint LMMSE channel estimation without feedback."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.dpss import FULL_FRAME_DOMAIN
from data_generation.src.receiver.estimation import input_windows
from src.common.numerics import hermitian
from data_generation.src.pdd import SYMBOL_COUNT, DATA_SYMBOL_INDICES, TRAINING_SYMBOL_INDICES


def selected_time_covariance(kernel, rows):
    """S F kernel F^H S^H without whitening or retaining excluded FFT rows."""
    left = np.fft.fftshift(np.fft.fft(kernel, axis=0, norm="ortho"), axes=0)[rows]
    return np.fft.fftshift(np.fft.ifft(left, axis=1, norm="ortho"), axes=1)[:, rows]


def selected_components(prior, known, masks, backend):
    n, q, length = prior.model.c["nfft"], prior.model.count, prior.model.length
    known = np.asarray(known, np.complex128)
    if (prior.fit_domain != FULL_FRAME_DOMAIN or prior.empirical_weight != 0
            or prior.scale != 1 or prior.interference_scale != 1 or prior.feedback_iterations != 0
            or set(prior.source_records) & set(prior.model.c.get("test_records", []))):
        raise ValueError("PDDP requires an independent full-frame fixed prior with feedback disabled")
    if known.shape != (SYMBOL_COUNT, n) or np.any(known[list(DATA_SYMBOL_INDICES)][:, masks["data"]] != 0):
        raise ValueError("Unknown D1/D2 labels must be zero in public training")
    rows = [np.arange(n) if s in TRAINING_SYMBOL_INDICES else np.flatnonzero(masks["tracking"])
            for s in range(SYMBOL_COUNT)]
    offsets = np.cumsum([0]+[len(r) for r in rows])
    size = offsets[-1]
    covariance = backend.array(np.zeros((size, size), complex))
    cross = np.zeros((q, length, size), complex)
    v = prior.model.sampled_basis.reshape(SYMBOL_COUNT, n, q)
    x = input_windows(known, length)
    # E[x(n-l) x(m-k)^*] = ifft(data_mask)[(n-m)-(l-k)].
    correlation = np.fft.ifft(np.fft.ifftshift(masks["data"].astype(float)))
    difference = (np.arange(n)[:, None]-np.arange(n)) % n
    for select, temporal, delay in prior.separable_blocks():
        lt = linalg.cholesky(temporal, lower=True)
        ld = linalg.cholesky(delay, lower=True)
        width = delay.shape[0]
        # Project response factors FIRST. Only selected rows enter A Rc A^H.
        factors = []
        for s in range(SYMBOL_COUNT):
            vt, xd = v[s] @ lt, x[s, :, select] @ ld
            time = (vt[:, :, None]*xd[:, None, :]).reshape(n, q*width)
            frequency = np.fft.fftshift(np.fft.fft(time, axis=0, norm="ortho"), axes=0)
            factors.append(frequency[rows[s]])
        z = np.concatenate(factors)
        dz = backend.array(z)
        covariance += dz @ dz.conj().T
        cross[:, select] = np.einsum("qa,lb,mab->qlm", lt, ld, z.conj().reshape(size, q, width), optimize=True)
        del dz, z, factors
        # ICI from independent unknown QPSK in each D, including colored terms.
        rho = np.zeros(n, complex)
        indices = np.arange(width)
        np.add.at(rho, ((indices[:, None]-indices) % n).ravel(), delay.ravel())
        stationary = np.fft.ifft(np.fft.fft(rho)*np.fft.fft(correlation))[difference]
        for s in DATA_SYMBOL_INDICES:
            kernel = (v[s] @ temporal @ v[s].conj().T)*stationary
            interference = hermitian(selected_time_covariance(kernel, rows[s]))
            target = slice(offsets[s], offsets[s+1])
            covariance[target, target] += backend.array(interference)
    return hermitian(backend.numpy(covariance)), cross.reshape(q*length, size), rows, offsets


class JointEstimator:
    """One P1+D1+D2+P2 solve; both D share batch x Q x L coefficients."""

    def __init__(self, prior, known, masks, unit_noise_covariance, backend):
        self.prior, self.backend = prior, backend
        self.model_error_ratio = float(prior.model.c.get("csi_model_error_ratio", 0))
        if not 0 <= self.model_error_ratio <= 1:
            raise ValueError("CSI model error ratio must be finite and between zero and one")
        self.known = np.asarray(known).copy()
        self.masks = {k: np.asarray(v).copy() for k, v in masks.items()}
        covariance, cross, self.rows, self.offsets = selected_components(prior, known, masks, backend)
        n = prior.model.c["nfft"]
        noise = np.asarray(unit_noise_covariance, complex)
        if noise.shape != (n, n) or not np.all(np.isfinite(noise)) or not np.allclose(noise, noise.conj().T):
            raise ValueError("Invalid colored noise covariance")
        self.noise_gain = float(np.trace(noise).real/n)
        # Select S Rn S^H before whitening the full joint observation.
        blocks = [noise[np.ix_(r, r)] for r in self.rows]
        selected_noise = linalg.block_diag(*blocks)
        size = int(self.offsets[-1])
        f = linalg.cholesky(selected_noise, lower=True)
        w = linalg.solve_triangular(f, np.eye(size), lower=True, check_finite=False)
        dw = backend.array(w)
        dc = dw @ backend.array(covariance) @ dw.conj().T
        values, vectors = backend.eigh((dc+dc.conj().T)*.5)
        raw = backend.numpy(values)
        if raw.min() < -1e-8*max(raw.max(), 1e-30):
            raise RuntimeError("Selected observation covariance is not positive semidefinite")
        positive = np.maximum(raw, 0)
        self.ridge = prior.regularizer*float(positive.mean())
        self.values = backend.array(positive)
        self.projected = dw.conj().T @ vectors
        self.gain = backend.array(cross) @ self.projected
        self.diagnostics = [dict(observations=size, min_eigenvalue=float(raw.min()),
                                 max_eigenvalue=float(raw.max()), whitened_ridge=self.ridge)]

    def effective_noise_variances(self, observations, noise_variances):
        """Thermal RF variance plus a floor inferred only from public P1/P2."""
        y = np.asarray(observations, complex)
        if y.ndim == 2:
            y = y[None]
        if y.shape[1:] != self.known.shape:
            raise ValueError("Expected batch x 4 x nfft observations")
        variance = np.broadcast_to(np.asarray(noise_variances, float), (len(y),)).copy()
        if np.any(variance < 0) or not np.all(np.isfinite(variance)):
            raise ValueError("Invalid noise variance")
        if self.model_error_ratio == 0:
            return variance
        symbols = list(TRAINING_SYMBOL_INDICES)
        if self.noise_gain <= 0:
            raise ValueError("CSI covariance floor needs positive noise gain")
        training_power = np.mean(np.sum(abs(self.known[symbols])**2, axis=1))
        if training_power <= 0:
            raise ValueError("CSI covariance floor needs nonzero public training")
        data_power = np.mean(np.sum(abs(self.known[list(DATA_SYMBOL_INDICES)])**2, axis=1)) + np.count_nonzero(self.masks["data"])
        received_power = np.mean(abs(y[:, symbols])**2, axis=(1, 2))
        if not np.all(np.isfinite(received_power)):
            raise ValueError("Nonfinite public training power")
        signal_rf_power = np.maximum(received_power/self.noise_gain-variance, 0)
        return variance+self.model_error_ratio*(data_power/training_power)*signal_rf_power

    def estimate_coefficients(self, observations, known, masks, noise_variances):
        if not np.array_equal(known, self.known) or any(not np.array_equal(masks.get(k), v) for k, v in self.masks.items()):
            raise ValueError("Public training/masks mismatch")
        y = np.asarray(observations, complex)
        if y.ndim == 2:
            y = y[None]
        if y.shape[1:] != self.known.shape:
            raise ValueError("Expected batch x 4 x nfft observations")
        selected = np.concatenate([y[:, s, r] for s, r in enumerate(self.rows)], axis=1)
        if not np.all(np.isfinite(selected)):
            raise ValueError("Invalid selected observations")
        effective = self.effective_noise_variances(y, noise_variances)
        b = self.backend
        spectral = self.projected.conj().T @ b.array(selected.T)
        denominator = self.values[:, None]+b.array(effective)[None]+self.ridge
        coefficients = self.gain @ (spectral/denominator)
        return b.numpy(coefficients).T.reshape(len(y), self.prior.model.count, self.prior.model.length)

    def estimate(self, observations, known, masks, noise_variances):
        coefficients = self.estimate_coefficients(observations, known, masks, noise_variances)
        return np.stack([np.stack([taps_to_matrix(self.prior.model.reconstruct(c, s))
                                  for s in DATA_SYMBOL_INDICES]) for c in coefficients])
