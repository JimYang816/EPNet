"""The batch detector retains the standalone single-tap receiver contract."""
import numpy as np

from src.algorithms.interface import EqualizerBatch
from src.algorithms.MMSE.single_tap import equalize_batch, mmse
from src.common.numerics import DenseBackend


def test_batch_uses_raw_observations_and_matches_public_single_sample_interface():
    rng = np.random.default_rng(187)
    channel = rng.normal(size=(2, 5, 5))+1j*rng.normal(size=(2, 5, 5))
    raw = rng.normal(size=(2, 5))+1j*rng.normal(size=(2, 5))
    cancelled = raw+100  # Must have no influence on single-tap detection.
    indices = np.array([1, 3, 4])
    mask = np.isin(np.arange(5), indices)
    noise = np.diag([.4, .6, .8, 1, 1.2])
    variances = np.array([.2, .5])
    batch = EqualizerBatch(channel, raw, cancelled, variances, indices, noise)
    result = equalize_batch(batch, DenseBackend())
    expected = np.stack([mmse(h, y, mask, v*np.diag(noise)) for h, y, v in zip(channel, raw, variances)])
    np.testing.assert_allclose(result.symbols, expected, atol=1e-15)
    np.testing.assert_array_equal(result.rejected_sites, 0)
    assert not {"bits", "labels", "reference_csi", "true_error"} & set(vars(batch))


def test_zero_gain_and_zero_noise_produce_finite_zero_estimates():
    batch = EqualizerBatch(np.zeros((1, 3, 3), complex), np.ones((1, 3), complex),
                          np.zeros((1, 3), complex), np.zeros(1), np.arange(3), np.eye(3))
    np.testing.assert_array_equal(equalize_batch(batch, DenseBackend()).symbols, 0)
