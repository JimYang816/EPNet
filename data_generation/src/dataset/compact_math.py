"""Lossless Fourier identities for compact CSI construction and diagnostics."""
from __future__ import annotations

import numpy as np

from data_generation.src.receiver.chain import taps_to_matrix


def matrix_to_taps(matrix):
    n = matrix.shape[-1]
    circular = np.fft.fft(np.fft.ifft(np.fft.ifftshift(matrix, axes=(-2, -1)), axis=-2,
                                       norm="ortho"), axis=-1, norm="ortho")
    rows = np.arange(n)[:, None]
    return circular[..., rows, (rows-np.arange(n)) % n]


def reconstruct(model, coefficients):
    return np.stack([taps_to_matrix(model.reconstruct(coefficients, symbol)) for symbol in (1, 2)])


def coefficient_nmse(model, coefficients, reference):
    """Exact full-matrix Frobenius NMSE via small Gram products, no band truncation."""
    coeff = np.asarray(coefficients, np.complex128)
    truth = matrix_to_taps(np.asarray(reference, np.complex128))
    result = []
    n, length = model.c["nfft"], model.length
    for d, symbol in enumerate((1, 2)):
        basis = model.sampled_basis[symbol*n:(symbol+1)*n]
        gram = basis.conj().T@basis
        cross = basis.conj().T@truth[d, :, :length]
        energy = np.sum(abs(truth[d])**2)
        raw_energy = np.einsum("...ql,qr,...rl->...", coeff.conj(), gram, coeff, optimize=True).real
        overlap = np.einsum("...ql,ql->...", coeff.conj(), cross, optimize=True).real
        if energy <= 0:
            raise ValueError("Zero-energy reference CSI")
        result.append(np.maximum((raw_energy+energy-2*overlap)/energy, 0))
    return np.stack(result, axis=-1)


def projection_power(reference, projected, known, data_indices):
    """Unknown data have unit symbol energy; known pilot contribution is coherent."""
    residual = np.asarray(reference)-np.asarray(projected)
    return np.sum(abs(residual[..., data_indices])**2, axis=-1)+abs(
        np.einsum("dij,dj->di", residual, known[[1, 2]], optimize=True))**2
