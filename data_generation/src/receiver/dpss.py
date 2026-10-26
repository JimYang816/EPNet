"""DPSS on the full 5632-sample P-D-D-P frame, including all four CPs."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg
from scipy.signal.windows import dpss
from data_generation.src.pdd import SYMBOL_COUNT

FULL_FRAME_DOMAIN = "full_frame_including_cp"


@dataclass
class DPSSModel:
    basis: np.ndarray
    bandwidth_hz: float
    tail_ratio: float
    length: int
    c: dict

    @property
    def count(self):
        return self.basis.shape[1]

    @property
    def sampled_basis(self):
        c = self.c
        positions = np.concatenate([s*(c["nfft"]+c["cp"])+c["cp"]-c["fft_advance"]+np.arange(c["nfft"]) for s in range(SYMBOL_COUNT)])
        return self.basis[positions]

    def coefficients(self, taps):
        """FFT-window least-squares projection, retained for model diagnostics."""
        g = np.asarray(taps)[..., :self.length].reshape(-1, SYMBOL_COUNT*self.c["nfft"], self.length)
        pinv = linalg.pinv(self.sampled_basis)
        return np.einsum("qn,mnl->mql", pinv, g, optimize=True)

    def coefficients_full_frame(self, taps):
        """Fit actual full-frame kernels, including CP; never extrapolate FFT taps."""
        g = np.asarray(taps)
        if (g.ndim < 2 or g.shape[-2] != len(self.basis)
                or g.shape[-1] < self.length or not np.all(np.isfinite(g))):
            raise ValueError("Full-frame taps must include every frame sample and the delay support")
        g = g[..., :self.length].reshape(-1, len(self.basis), self.length)
        return np.einsum("qn,mnl->mql", linalg.pinv(self.basis), g, optimize=True)

    def reconstruct(self, coefficients, symbol=1):
        n = self.c["nfft"]
        v = self.sampled_basis[symbol*n:(symbol+1)*n]
        return np.einsum("nq,...ql->...nl", v, coefficients, optimize=True)


def build_model(c):
    """Build the fixed absolute-Hz time basis, independent of native snapshot rate."""
    total = SYMBOL_COUNT*(c["nfft"]+c["cp"])
    bandwidth = float(c["dpss_half_bandwidth_hz"])
    if not np.isfinite(bandwidth) or not 0 < bandwidth < c["fs_baseband"]/2:
        raise ValueError("DPSS half bandwidth must be finite and within the baseband Nyquist limit")
    nw = total*bandwidth/c["fs_baseband"]
    basis, eigenvalues = dpss(total, nw, Kmax=min(total, int(np.ceil(2*nw))+24),
                             sym=True, norm=2, return_ratios=True)
    # The exact sum over all DPSS eigenvalues equals 2NW.
    tails = np.maximum(1-np.cumsum(eigenvalues)/(2*nw), 0)
    good = np.flatnonzero(tails <= c["dpss_tail_ratio"])
    if not len(good):
        raise RuntimeError("Insufficient DPSS eigenvalues for requested tail threshold")
    count = int(good[0]+1)
    if count != c["dpss_expected_count"]:
        raise ValueError("Fixed DPSS configuration generated an unexpected basis count")
    return DPSSModel(basis[:count].T.copy(), bandwidth, c["dpss_tail_ratio"], c["delay_length"], c)
