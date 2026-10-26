"""Receiver-model covariance proxy, applied without a dense coefficient covariance."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


class CovarianceProxy:
    """C = P - G diag(1 / (lambda + effective_rf_variance + ridge)) G^H.

    Unknown-data interference and the receiver's regularizer are included in
    the observation spectrum. This is a receiver-model proxy, not the true
    covariance of the reference-assisted channel error.
    """
    semantics = "receiver_model_covariance_proxy"

    def __init__(self, blocks, gain, values, ridge, variance, shape, scale=1.0):
        self.blocks, self.gain = blocks, gain
        self.values = np.asarray(values)
        self.denominator = self.values+float(variance)+float(ridge)
        self.coefficient_shape = tuple(shape)
        self.shape = (int(np.prod(shape)),)*2
        self.scale = float(scale)
        if (not np.all(np.isfinite(self.denominator)) or np.any(self.denominator <= 0)
                or not np.isfinite(self.scale) or self.scale < 0):
            raise ValueError("Invalid covariance proxy scale/denominator")

    def matmul(self, vector):
        x = np.asarray(vector, np.complex128)
        single = x.ndim == 1
        if single:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != self.shape[0]:
            raise ValueError("Expected coefficient dimension x RHS")
        q, length = self.coefficient_shape
        xx = x.reshape(q, length, -1)
        out = np.zeros_like(xx)
        for start, stop, temporal, delay in self.blocks:
            out[:, start:stop] = np.einsum("ab,bkc,lk->alc", temporal, xx[:, start:stop], delay, optimize=True)
        flat = out.reshape(x.shape)
        # gain may be an HDF5 dataset; stream spectral columns in bounded chunks.
        for start in range(0, len(self.values), 128):
            stop = min(start+128, len(self.values))
            g = np.asarray(self.gain[:, start:stop])
            flat -= g@((g.conj().T@x)/self.denominator[start:stop, None])
        flat *= self.scale
        return flat[:, 0] if single else flat

    def trace(self):
        value = sum(float(np.trace(t).real*np.trace(d).real) for _, _, t, d in self.blocks)
        for start in range(0, len(self.values), 128):
            stop = min(start+128, len(self.values))
            g = np.asarray(self.gain[:, start:stop])
            value -= float(np.sum(np.sum(abs(g)**2, axis=0)/self.denominator[start:stop]))
        if value < -1e-8*max(1, sum(np.trace(t).real*np.trace(d).real for _, _, t, d in self.blocks)):
            raise ValueError("Negative covariance proxy trace")
        return self.scale*max(value, 0.0)

    def modes(self, count, *, tolerance=1e-7):
        """Caller chooses the mode count; no bandwidth or layer setting is frozen."""
        n = self.shape[0]
        if not 0 < count < n:
            raise ValueError("Mode count must be between 1 and coefficient dimension - 1")
        operator = LinearOperator(self.shape, matvec=self.matmul, matmat=self.matmul, dtype=np.complex128)
        values, vectors = eigsh(operator, k=count, which="LA", tol=tolerance, v0=np.ones(n, complex))
        order = np.argsort(values)[::-1]
        return np.maximum(values[order], 0), vectors[:, order]


def blocks_from_prior(prior):
    return [(s.start, s.stop, t, d) for s, t, d in prior.separable_blocks()]
