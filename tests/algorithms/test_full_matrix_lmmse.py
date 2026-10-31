"""Independent receive-domain analytical reference for batched matrix LMMSE."""
import numpy as np

from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.algorithms.MMSE.full_matrix import equalize_batch
from src.common.numerics import DenseBackend


def test_all_receive_rows_and_cancelled_observations_match_receive_domain_solution():
    rng = np.random.default_rng(808)
    channel = rng.normal(size=(2, 7, 7))+1j*rng.normal(size=(2, 7, 7))
    cancelled = rng.normal(size=(2, 7))+1j*rng.normal(size=(2, 7))
    raw = cancelled+50
    a = rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7))
    noise = a@a.conj().T+np.eye(7)
    indices, variances = np.array([1, 3, 6]), np.array([.3, .7])
    batch = EqualizerBatch(channel, raw, cancelled, variances, indices, noise)
    backend = DenseBackend()
    actual = equalize_batch(batch, backend, whitener=make_whitener(noise, backend))
    expected = []
    for full, y, variance in zip(channel, cancelled, variances):
        h = full[:, indices]
        expected.append(h.conj().T@np.linalg.solve(h@h.conj().T+variance*noise, y))
    np.testing.assert_allclose(actual.symbols, expected, atol=2e-14)
    np.testing.assert_array_equal(actual.rejected_sites, 0)
