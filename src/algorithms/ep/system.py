"""Full-ICI likelihood with the complete colored receiver-noise covariance."""
from __future__ import annotations

import numpy as np
from scipy import linalg

from src.common.numerics import DenseBackend


class NoiseWhitening:
    """A fixed, validated noise factor shared by changing channel systems."""
    def __init__(self, covariance, backend):
        r = np.array(covariance, dtype=complex, copy=True)
        if r.ndim != 2 or r.shape[0] != r.shape[1] or not np.all(np.isfinite(r)) or not np.allclose(r, r.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError("Noise covariance must be finite and Hermitian")
        self.lower = linalg.cholesky(r, lower=True)
        self.backend = backend
        self.operator = backend.array(linalg.solve_triangular(self.lower, np.eye(len(r)), lower=True, check_finite=False))
        r.flags.writeable = False
        self.covariance = r


class EqualizerSystem:
    def __init__(self, h, unit_noise_covariance, backend=None, *, prepared_noise=None):
        self.backend = backend or DenseBackend()
        h, r = np.asarray(h, np.complex128), np.asarray(unit_noise_covariance, np.complex128)
        if h.ndim != 2 or r.shape != (h.shape[0], h.shape[0]):
            raise ValueError("Inconsistent channel/noise dimensions")
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(r)):
            raise ValueError("Nonfinite channel/noise statistics")
        if not np.allclose(r, r.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError("Noise covariance must be Hermitian")
        if prepared_noise is not None:
            if prepared_noise.backend is not self.backend or not np.array_equal(prepared_noise.covariance, r):
                raise ValueError("Cached noise model mismatch")
            f = prepared_noise.lower
        else:
            f = linalg.cholesky(r, lower=True)
        white_h = linalg.solve_triangular(f, h, lower=True, check_finite=False)
        self.whitener = prepared_noise.operator if prepared_noise is not None else self.backend.array(linalg.solve_triangular(f, np.eye(len(f)), lower=True, check_finite=False))
        self.h = self.backend.array(white_h)
        self.gram = self.h.conj().T @ self.h
        self.dimension = h.shape[1]

    def likelihood(self, y, noise_variance):
        y = np.asarray(y, np.complex128)
        if y.shape != (self.h.shape[0],) or not np.all(np.isfinite(y)):
            raise ValueError("Invalid observation")
        if not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("Noise variance must be finite and positive")
        b = self.backend
        gram = self.gram/noise_variance
        matched = self.h.conj().T @ (self.whitener @ b.array(y))/noise_variance
        return gram, matched
