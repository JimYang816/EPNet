"""Explicit complex128 CPU/CUDA backend; no implicit precision reduction."""
from __future__ import annotations

import numpy as np
from scipy import linalg


class DenseBackend:
    def __init__(self, device="cpu"):
        self.device = device
        self.torch = None
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            self.torch = torch

    def array(self, a):
        a = np.asarray(a)
        if self.torch:
            return self.torch.as_tensor(a, device="cuda")
        return a

    def numpy(self, a):
        return a.detach().cpu().numpy() if self.torch else np.asarray(a)

    def eigh(self, a):
        if self.torch:
            return self.torch.linalg.eigh(a)
        try:
            return linalg.eigh(a, check_finite=False, driver="evd")
        except linalg.LinAlgError:
            # Some Windows LAPACK builds fail divide-and-conquer on valid
            # rank-deficient complex covariances. EVR preserves the same solve.
            return linalg.eigh(a, check_finite=False, driver="evr")

    def solve(self, a, b):
        if self.torch:
            return self.torch.linalg.solve(a, b)
        return linalg.solve(a, b, assume_a="her", check_finite=False)

    def inverse_positive(self, a):
        if self.torch:
            f = self.torch.linalg.cholesky(a)
            return self.torch.cholesky_inverse(f)
        return linalg.cho_solve(linalg.cho_factor(a, lower=True, check_finite=False),
                                np.eye(len(a), dtype=np.complex128), check_finite=False)

    def synchronize(self):
        if self.torch:
            self.torch.cuda.synchronize()


def hermitian(a):
    return (a+a.conj().T)*0.5
