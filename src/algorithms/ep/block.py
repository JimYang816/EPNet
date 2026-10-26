"""Traditional parallel complex block EP with uniform unit-energy QPSK.

Natural-parameter damping and rejection of invalid updates follow Santos et al.,
EUSIPCO 2015 Algorithm 2, extended to proper complex Gaussian moments.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


CONSTELLATION = np.array([1+1j, 1-1j, -1-1j, -1+1j])/np.sqrt(2)


def ep(system, y, noise_variance, max_iterations=20, damping=0.5,
             relative_tolerance=1e-4, variance_floor=1e-8, *,
             min_iterations=5, convergence_patience=2):
    """Stop after consecutive small changes in the updated posterior mean.

    Iterations count completed site updates. The stability streak may start
    before min_iterations; budgets shorter than that minimum run to their cap.
    """
    counts = (max_iterations, min_iterations, convergence_patience)
    if (not all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)) for x in counts)
            or not (1 <= max_iterations <= 20 and min_iterations >= 1 and convergence_patience >= 1)
            or not np.all(np.isfinite((damping, relative_tolerance, variance_floor)))
            or not (0 < damping <= 1 and relative_tolerance > 0 and variance_floor > 0)):
        raise ValueError("Invalid EP controls")
    b, dimension = system.backend, system.dimension
    gram, matched = system.likelihood(y, noise_variance)
    precision = np.ones(dimension)
    natural_mean = np.zeros(dimension, np.complex128)
    covariance = b.inverse_positive(gram+b.array(np.diag(precision).astype(np.complex128)))
    mean = b.numpy(covariance @ matched)
    rejected, flat_cavities, converged = 0, 0, False
    stable_iterations = 0
    for iteration in range(1, max_iterations+1):
        previous = mean
        marginal = np.maximum(np.diag(b.numpy(covariance)).real, variance_floor)
        cavity_precision = 1/marginal-precision
        cavity_natural = mean/marginal-natural_mean
        valid = np.isfinite(cavity_precision) & (cavity_precision > 1e-14) & np.isfinite(cavity_natural)
        flat_cavities += int(np.sum(~valid))
        cavity_variance = np.ones(dimension)
        cavity_mean = np.zeros(dimension, np.complex128)
        cavity_variance[valid] = 1/cavity_precision[valid]
        cavity_mean[valid] = cavity_natural[valid]*cavity_variance[valid]
        logits = -abs(CONSTELLATION[None]-cavity_mean[:, None])**2/cavity_variance[:, None]
        probabilities = np.exp(logits-logsumexp(logits, axis=1)[:, None])
        tilted_mean = probabilities @ CONSTELLATION
        tilted_variance = np.maximum(np.sum(probabilities*abs(CONSTELLATION[None]-tilted_mean[:, None])**2, axis=1), variance_floor)
        proposed_precision = 1/tilted_variance-cavity_precision
        proposed_natural = tilted_mean/tilted_variance-cavity_natural
        damped_precision = (1-damping)*precision+damping*proposed_precision
        damped_natural = (1-damping)*natural_mean+damping*proposed_natural
        accept = valid & np.isfinite(damped_precision) & (damped_precision > 0) & np.isfinite(damped_natural)
        rejected += int(np.sum(valid & ~accept))
        precision[accept] = damped_precision[accept]
        natural_mean[accept] = damped_natural[accept]
        posterior_precision = gram+b.array(np.diag(precision).astype(np.complex128))
        posterior_natural = matched+b.array(natural_mean)
        if iteration == max_iterations:
            # The final mean needs no covariance for a subsequent site update.
            mean = b.numpy(b.solve(posterior_precision, posterior_natural))
        else:
            covariance = b.inverse_positive(posterior_precision)
            mean = b.numpy(covariance @ posterior_natural)
        if not np.all(np.isfinite(mean)):
            raise FloatingPointError("EP produced nonfinite output")
        relative_change = np.linalg.norm(mean-previous)/max(np.linalg.norm(mean), 1e-15)
        stable_iterations = stable_iterations+1 if relative_change <= relative_tolerance else 0
        if iteration >= min_iterations and stable_iterations >= convergence_patience:
            converged = True
            break
    return mean, dict(iterations=iteration, converged=converged, rejected_updates=rejected,
                      flat_or_invalid_cavities=flat_cavities, relative_change=float(relative_change),
                      stop_reason="relative_tolerance" if converged else "max_iterations")
