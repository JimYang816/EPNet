"""Analytical checks for the separately maintained robust detector core."""
import numpy as np

from src.algorithms.robust_ep.real import paper_ep
from src.common.numerics import DenseBackend


def test_robust_one_iteration_against_discrete_moments():
    gram = np.array([[3.1, -.8], [-.8, 2.5]])
    matched = np.array([1.2, -.7])
    covariance = np.linalg.inv(gram+2*np.eye(2))
    mean = covariance@matched
    cavity_variance = 1/(1/np.diag(covariance)-2)
    cavity_mean = cavity_variance*mean/np.diag(covariance)
    alphabet = np.array([-1, 1])/np.sqrt(2)
    weights = np.exp(-.5*(alphabet[None]-cavity_mean[:, None])**2/cavity_variance[:, None])
    weights /= weights.sum(axis=1, keepdims=True)
    tilted_mean = weights@alphabet
    tilted_variance = weights@(alphabet**2)-tilted_mean**2
    proposal_variance = 1/(1/tilted_variance-1/cavity_variance)
    proposal_mean = proposal_variance*(tilted_mean/tilted_variance-cavity_mean/cavity_variance)
    accepted = proposal_variance >= 5e-7
    variance = np.where(accepted, .2*proposal_variance+.8*.5, .5)
    site_mean = np.where(accepted, .2*proposal_mean, 0)
    expected = np.linalg.solve(gram+np.diag(1/variance), matched+site_mean/variance)
    actual, rejected = paper_ep(gram, matched, DenseBackend(), iterations=1)
    np.testing.assert_allclose(actual, expected[0]+1j*expected[1], atol=2e-15)
    assert rejected == np.count_nonzero(~accepted)


def test_robust_core_remains_independent_of_ordinary_ep(monkeypatch):
    import src.algorithms.ep.real as ordinary

    def fail(*args, **kwargs):
        raise AssertionError("Ordinary EP core was called")

    monkeypatch.setattr(ordinary, "paper_ep", fail)
    estimate, rejected = paper_ep(np.zeros((4, 4)), np.zeros(4), DenseBackend())
    np.testing.assert_array_equal(estimate, 0)
    assert rejected == 40
