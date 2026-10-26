import numpy as np
import pytest

from src.algorithms.MMSE import mmse
from src.algorithms.ep import EqualizerSystem, ep
from data_generation.src.transmitter.ofdm import qpsk, decisions


def test_ep_likelihood_with_colored_noise():
    rng = np.random.default_rng(2)
    h = rng.normal(size=(12, 7))+1j*rng.normal(size=(12, 7))
    a = rng.normal(size=(12, 12))+1j*rng.normal(size=(12, 12))
    r = a@a.conj().T+np.eye(12)
    y = rng.normal(size=12)+1j*rng.normal(size=12)
    gram, matched = EqualizerSystem(h, r).likelihood(y, .3)
    np.testing.assert_allclose(gram, h.conj().T@np.linalg.solve(r, h)/.3, atol=4e-14)
    np.testing.assert_allclose(matched, h.conj().T@np.linalg.solve(r, y)/.3, atol=4e-14)


@pytest.mark.parametrize("diagonal", [False, True])
def test_identity_diagonal_ep_and_mmse(diagonal):
    bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.uint8)
    x = qpsk(bits)
    h = np.diag(np.array([1, .7+.4j, -.9j, 1.3]) if diagonal else np.ones(4))
    y = h@x
    system = EqualizerSystem(h, np.eye(4))
    estimate, info = ep(system, y, .02)
    np.testing.assert_array_equal(decisions(estimate), bits)
    np.testing.assert_array_equal(decisions(mmse(h, y, np.ones(4, bool), .02)), bits)
    assert info["iterations"] <= 20


def test_zero_and_invalid_inputs():
    system = EqualizerSystem(np.zeros((5, 3), complex), np.eye(5))
    mean, info = ep(system, np.zeros(5), .1)
    np.testing.assert_array_equal(mean, 0)
    assert info["flat_or_invalid_cavities"] > 0
    assert np.all(np.isfinite(mean))
    assert info["iterations"] == 5
    assert info["converged"] and info["stop_reason"] == "relative_tolerance"
    assert info["relative_change"] == 0
    with pytest.raises(ValueError):
        system.likelihood(np.zeros(5), 0)
    with pytest.raises(ValueError):
        system.likelihood(np.full(5, np.nan), 1)
    with pytest.raises(np.linalg.LinAlgError):
        EqualizerSystem(np.ones((5, 3)), -np.eye(5))
