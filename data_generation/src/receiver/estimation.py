"""Frozen full-frame, delay-local channel statistics for P-D-D-P estimation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data_generation.src.transmitter.ofdm import ifft
from data_generation.src.receiver.dpss import FULL_FRAME_DOMAIN
from src.common.numerics import hermitian


def input_windows(known, length):
    n = known.shape[1]
    return ifft(known)[:, (np.arange(n)[:, None]-np.arange(length)) % n]


@dataclass
class FrozenPrior:
    model: object
    coefficients: np.ndarray
    temporal_covariance: np.ndarray
    delay_covariance: np.ndarray
    empirical_weight: float
    scale: float
    regularizer: float
    source_records: tuple
    fit_domain: str = FULL_FRAME_DOMAIN
    delay_block_size: int = 0
    interference_scale: float = 1.0
    feedback_iterations: int = 0

    @classmethod
    def fit_coefficients(cls, model, coefficients, weight, scale, source_records,
                         regularizer=1e-6, *, fit_domain=FULL_FRAME_DOMAIN, delay_block_size=0,
                         interference_scale=1.0):
        coeff = np.asarray(coefficients, np.complex128).copy()
        if (coeff.ndim != 3 or coeff.shape[1:] != (model.count, model.length)
                or not len(coeff) or not np.all(np.isfinite(coeff))
                or np.sum(abs(coeff)**2) <= 0):
            raise ValueError("Invalid prior coefficients")
        if fit_domain != FULL_FRAME_DOMAIN:
            raise ValueError("PDDP requires full-frame prior coefficients")
        if (not isinstance(delay_block_size, (int, np.integer))
                or delay_block_size < 0 or delay_block_size > model.length):
            raise ValueError("Invalid delay covariance block size")
        if not np.isfinite(interference_scale) or interference_scale < 0:
            raise ValueError("Invalid data interference scale")
        time = np.einsum("mql,mrl->qr", coeff, coeff.conj(), optimize=True)/len(coeff)
        delay = np.einsum("mql,mqk->lk", coeff, coeff.conj(), optimize=True)/len(coeff)
        trace = np.mean(np.sum(abs(coeff)**2, axis=(1, 2)))
        # Keep the separable and empirical second moments at the same trace.
        time += regularizer*np.trace(time).real/len(time)*np.eye(len(time))
        delay += regularizer*np.trace(delay).real/len(delay)*np.eye(len(delay))
        time /= np.trace(time).real
        delay *= trace/np.trace(delay).real
        return cls(model, coeff, hermitian(time), hermitian(delay), weight, scale,
                   regularizer, tuple(source_records), fit_domain, int(delay_block_size), float(interference_scale))

    def separable_blocks(self):
        """Retain each delay block's temporal law and within-block correlation."""
        if not self.delay_block_size or self.delay_block_size == self.model.length:
            yield slice(0, self.model.length), self.temporal_covariance, self.delay_covariance
            return
        for start in range(0, self.model.length, self.delay_block_size):
            select = slice(start, min(start+self.delay_block_size, self.model.length))
            coeff = self.coefficients[:, :, select]
            time = np.einsum("mql,mrl->qr", coeff, coeff.conj(), optimize=True)/len(coeff)
            delay = np.einsum("mql,mqk->lk", coeff, coeff.conj(), optimize=True)/len(coeff)
            energy = float(np.trace(delay).real)
            if energy == 0:
                continue
            time += self.regularizer*np.trace(time).real/len(time)*np.eye(len(time))
            delay += self.regularizer*energy/len(delay)*np.eye(len(delay))
            time /= np.trace(time).real
            delay *= energy/np.trace(delay).real
            yield select, hermitian(time), hermitian(delay)
