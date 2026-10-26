"""Full-frame kernels and delay-local moments used by the PDDP estimator."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from data_generation.src.receiver.chain import full_frame_kernels
from data_generation.src.receiver.dpss import DPSSModel, FULL_FRAME_DOMAIN
from data_generation.src.receiver.estimation import FrozenPrior


def test_full_frame_coefficient_recovery_and_cp_information():
    rng = np.random.default_rng(62)
    c = dict(nfft=16, cp=8, fft_advance=2)
    basis, _ = np.linalg.qr(rng.normal(size=(96, 4)))
    model = DPSSModel(basis, 1, .001, 5, c)
    coefficients = rng.normal(size=(3, 4, 5))+1j*rng.normal(size=(3, 4, 5))
    full = np.einsum("nq,mql->mnl", model.basis, coefficients)
    recovered = model.coefficients_full_frame(full)
    np.testing.assert_allclose(recovered, coefficients, atol=3e-15)
    sampled = np.einsum("nq,mql->mnl", model.sampled_basis, coefficients).reshape(3, 4, 16, 5)
    np.testing.assert_allclose(model.coefficients(sampled), coefficients, atol=4e-15)
    prior = FrozenPrior.fit_coefficients(model, recovered, 0, 1, [51])
    assert prior.fit_domain == FULL_FRAME_DOMAIN
    changed = full.copy()
    changed[:, 0, :] += 2
    assert np.linalg.norm(model.coefficients_full_frame(changed)-recovered) > .1
    with pytest.raises(ValueError, match="Full-frame"):
        model.coefficients_full_frame(sampled)
    with pytest.raises(ValueError, match="coefficients"):
        FrozenPrior.fit_coefficients(model, coefficients[:, :, :-1], 0, 1, [51])


@pytest.mark.parametrize("use_sparse", [False, True])
def test_full_frame_kernel_bounds_are_zero_not_wrapped(use_sparse):
    c = dict(nfft=4, cp=2, fft_advance=2, guard_samples=1)
    matrix = np.arange(676).reshape(26, 26)+1j
    op = sparse.csr_matrix(matrix) if use_sparse else matrix
    actual = full_frame_kernels(op, c, 8)
    assert actual.shape == (24, 8)
    np.testing.assert_array_equal(actual[0, 4:], 0)
    np.testing.assert_array_equal(actual[-1, 0], 0)
    assert actual[0, 3] == matrix[1, 0]
    assert actual[-1, 1] == matrix[24, 25]
    assert actual[0, 0] == matrix[1, 3]


def test_local_moments_keep_delay_dependent_dynamics():
    rng = np.random.default_rng(19)
    model = DPSSModel(np.eye(16, 2), 1, .001, 2, dict(nfft=2, cp=2, fft_advance=0))
    coefficients = np.zeros((10, 2, 2), complex)
    coefficients[:, 0, 0] = rng.normal(size=10)+1j*rng.normal(size=10)
    coefficients[:, 1, 1] = rng.normal(size=10)+1j*rng.normal(size=10)
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, [51], delay_block_size=1)
    blocks = list(prior.separable_blocks())
    assert blocks[0][1][0, 0].real > .99999
    assert blocks[1][1][1, 1].real > .99999
    with pytest.raises(ValueError, match="block size"):
        FrozenPrior.fit_coefficients(model, coefficients, 0, 1, [51], delay_block_size=-1)
