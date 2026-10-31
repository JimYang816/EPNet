import numpy as np

from src.algorithms.MMSE import lmmse
from src.algorithms.ep import EqualizerSystem
from src.algorithms.ep.real import paper_ep, real_likelihood
from src.common.numerics import DenseBackend
from data_generation.src.transmitter.ofdm import qpsk, decisions


def test_full_lmmse_and_real_likelihood_with_complex_correlated_noise():
    rng = np.random.default_rng(130)
    h = rng.normal(size=(7, 4))+1j*rng.normal(size=(7, 4))
    a = rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7))
    r = a@a.conj().T+np.eye(7)
    y = rng.normal(size=7)+1j*rng.normal(size=7)
    variance = .4
    system = EqualizerSystem(h, r)
    expected = h.conj().T@np.linalg.solve(h@h.conj().T+variance*r, y)
    np.testing.assert_allclose(lmmse(system, y, variance), expected, atol=2e-14)
    gram, matched = system.likelihood(y, variance)
    rg, rm = real_likelihood(gram, matched)
    hr = np.block([[h.real, -h.imag], [h.imag, h.real]])
    rr = variance*.5*np.block([[r.real, -r.imag], [r.imag, r.real]])
    yr = np.r_[y.real, y.imag]
    np.testing.assert_allclose(rg, hr.T@np.linalg.solve(rr, hr), atol=5e-14)
    np.testing.assert_allclose(rm, hr.T@np.linalg.solve(rr, yr), atol=5e-14)


def test_one_paper_iteration_against_explicit_discrete_moments():
    gram = np.array([[3.1, -.8], [-.8, 2.5]])
    matched = np.array([1.2, -.7])
    covariance = np.linalg.inv(gram+2*np.eye(2))
    mean = covariance@matched
    marginal = np.diag(covariance)
    cv = 1/(1/marginal-2)
    cm = cv*mean/marginal
    alphabet = np.array([-1, 1])/np.sqrt(2)
    weights = np.exp(-.5*(alphabet[None]-cm[:, None])**2/cv[:, None])
    weights /= weights.sum(axis=1, keepdims=True)
    tm = weights@alphabet
    tv = weights@(alphabet**2)-tm**2
    pv = 1/(1/tv-1/cv)
    pm = pv*(tm/tv-cm/cv)
    accept = pv >= 5e-7
    sv, sm = np.where(accept, .2*pv+.8*.5, .5), np.where(accept, .2*pm, 0)
    expected = np.linalg.solve(gram+np.diag(1/sv), matched+sm/sv)
    estimate, rejected = paper_ep(gram, matched, DenseBackend(), iterations=1)
    np.testing.assert_allclose(estimate, expected[0]+1j*expected[1], atol=2e-15)
    assert rejected == np.count_nonzero(~accept)


def test_batched_paper_ep_identity_and_invalid_site_rejection():
    bits = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], np.uint8)
    x = qpsk(bits)
    gram = np.broadcast_to(np.eye(2)/.2, (2, 2, 2))
    rg, rm = real_likelihood(gram, x/.2)
    estimate, rejected = paper_ep(rg, rm, DenseBackend())
    np.testing.assert_array_equal(decisions(estimate), bits)
    assert rejected.shape == (2,)
    zero, rejected = paper_ep(np.zeros((4, 4)), np.zeros(4), DenseBackend())
    np.testing.assert_array_equal(zero, 0)
    assert rejected == 40
