import numpy as np
import pytest

from src.algorithms.ep import EqualizerSystem, ep
from src.common.numerics import DenseBackend


class PosteriorTrace(DenseBackend):
    def __init__(self, means=None):
        super().__init__()
        self.controlled = iter(means) if means is not None else None
        self.means = []

    def numpy(self, value):
        value = super().numpy(value)
        if value.ndim == 1:
            # Control posterior means independently of the EP moment updates.
            if self.controlled is not None:
                value = np.full(value.shape, next(self.controlled), dtype=complex)
            self.means.append(value.copy())
        return value


@pytest.mark.parametrize("means,budget,iterations,converged", [
    ([1, 2, 3, 4, 4.0002, 4.0003], 20, 5, True),  # Rounds 4 and 5 qualify.
    ([1]*6, 20, 5, True),  # An early streak still waits for round 5.
    ([1, 1, 1, 1, 2, 2, 2], 20, 6, True),  # Round 4 resets the streak.
    ([1, 2, 3, 4, 5, 6, 6], 6, 6, False),  # One qualifying round at the cap.
    ([0]*4, 3, 3, False),  # Budget shorter than the minimum.
    ([1]*6, 5, 5, True),  # Convergence can first occur at the cap.
])
def test_stopping_requires_minimum_and_consecutive_updated_means(means, budget, iterations, converged):
    backend = PosteriorTrace(means)
    system = EqualizerSystem(np.eye(1), np.eye(1), backend)
    output, info = ep(system, np.ones(1), .2, max_iterations=budget)
    assert info["iterations"] == iterations
    assert info["converged"] is converged
    assert info["stop_reason"] == ("relative_tolerance" if converged else "max_iterations")
    assert len(backend.means) == iterations+1
    np.testing.assert_array_equal(output, [means[iterations]])


@pytest.mark.parametrize("budget,reason", [(3, "max_iterations"), (20, "relative_tolerance")])
def test_relative_change_describes_returned_posterior(budget, reason):
    backend = PosteriorTrace()
    system = EqualizerSystem(np.eye(2), np.eye(2), backend)
    output, info = ep(system, np.array([1+1j, -1-1j])/np.sqrt(2), .02, max_iterations=budget)
    np.testing.assert_array_equal(output, backend.means[-1])
    expected = np.linalg.norm(output-backend.means[-2])/max(np.linalg.norm(output), 1e-15)
    assert info["relative_change"] == pytest.approx(expected, rel=1e-14, abs=1e-15)
    assert info["stop_reason"] == reason
    assert len(backend.means) == info["iterations"]+1


@pytest.mark.parametrize("controls", [
    dict(max_iterations=21), dict(max_iterations=1.5), dict(max_iterations=True),
    dict(min_iterations=0), dict(min_iterations=2.5), dict(min_iterations=True),
    dict(convergence_patience=0), dict(convergence_patience=1.5), dict(convergence_patience=True),
    dict(relative_tolerance=np.inf), dict(relative_tolerance=np.nan),
    dict(variance_floor=np.inf), dict(damping=np.nan),
])
def test_invalid_convergence_controls(controls):
    system = EqualizerSystem(np.eye(1), np.eye(1))
    with pytest.raises(ValueError, match="EP controls"):
        ep(system, np.ones(1), .2, **controls)


def test_nonfinite_updated_mean_is_rejected():
    system = EqualizerSystem(np.eye(1), np.eye(1), PosteriorTrace([1, np.nan]))
    with pytest.raises(FloatingPointError, match="nonfinite output"):
        ep(system, np.ones(1), .2, max_iterations=1)
