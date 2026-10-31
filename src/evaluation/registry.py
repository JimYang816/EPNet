"""Stable detector identifiers and explicit computation dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import importlib


@dataclass(frozen=True)
class Algorithm:
    identifier: str
    label: str
    module: str
    helpers: tuple[str, ...] = ()

    def equalize(self, *args, **kwargs):
        return importlib.import_module(self.module).equalize_batch(*args, **kwargs)


MATRIX = ("src.algorithms.likelihood:make_whitener", "src.algorithms.likelihood:likelihood_batch",
          "src.algorithms.likelihood:batch_likelihood")
REAL = ("src.algorithms.likelihood:backend_real_likelihood", "src.algorithms.likelihood:real_likelihood",
        "src.algorithms.likelihood:batch_real_likelihood")
ALGORITHMS = (
    Algorithm("single_tap_mmse", "Single-tap MMSE", "src.algorithms.MMSE.single_tap"),
    Algorithm("full_matrix_lmmse", "Full-matrix LMMSE", "src.algorithms.MMSE.full_matrix", MATRIX),
    Algorithm("ep", "EP", "src.algorithms.ep.real", MATRIX+REAL),
    Algorithm("robust_ep", "Robust EP", "src.algorithms.robust_ep.real", MATRIX+REAL),
)
CURVES = tuple(a.label for a in ALGORITHMS)
IDENTIFIERS = tuple(a.identifier for a in ALGORITHMS)


def resolve_parameters(algorithm_parameters=None, *, iterations=10, damping=.2, epsilon=5e-7):
    supplied = algorithm_parameters or {}
    if set(supplied)-set(IDENTIFIERS):
        raise ValueError(f"Unknown algorithms: {sorted(set(supplied)-set(IDENTIFIERS))}")
    result = {}
    for algorithm in ALGORITHMS:
        ep = algorithm.identifier in ("ep", "robust_ep")
        values = dict(iterations=iterations, damping=damping, epsilon=epsilon) if ep else {}
        override = supplied.get(algorithm.identifier, {})
        if not isinstance(override, dict) or set(override)-set(values):
            raise ValueError(f"Invalid parameters for {algorithm.identifier}")
        values.update(override)
        if ep:
            import math
            if (not isinstance(values["iterations"], int) or isinstance(values["iterations"], bool)
                    or values["iterations"] < 1 or not 0 < values["damping"] < 1
                    or not math.isfinite(values["epsilon"]) or values["epsilon"] <= 0):
                raise ValueError(f"Invalid parameters for {algorithm.identifier}")
        result[algorithm.identifier] = values
    return result


def resolve_force(force):
    selected = {force} if isinstance(force, str) else set(force)
    if selected-set(IDENTIFIERS)-{"all"}:
        raise ValueError(f"Unknown force algorithms: {sorted(selected-set(IDENTIFIERS)-{'all'})}")
    return set(IDENTIFIERS) if "all" in selected else selected
