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
ALGORITHMS = (
    Algorithm("single_tap_mmse", "Single-tap MMSE", "src.algorithms.MMSE.single_tap"),
    Algorithm("full_matrix_lmmse", "Full-matrix LMMSE", "src.algorithms.MMSE.full_matrix", MATRIX),
)
CURVES = tuple(a.label for a in ALGORITHMS)
IDENTIFIERS = tuple(a.identifier for a in ALGORITHMS)


def resolve_parameters(algorithm_parameters=None):
    supplied = algorithm_parameters or {}
    if set(supplied)-set(IDENTIFIERS):
        raise ValueError(f"Unknown algorithms: {sorted(set(supplied)-set(IDENTIFIERS))}")
    result = {}
    for algorithm in ALGORITHMS:
        override = supplied.get(algorithm.identifier, {})
        if not isinstance(override, dict) or override:
            raise ValueError(f"Invalid parameters for {algorithm.identifier}")
        result[algorithm.identifier] = {}
    return result


def resolve_force(force):
    selected = {force} if isinstance(force, str) else set(force)
    if selected-set(IDENTIFIERS)-{"all"}:
        raise ValueError(f"Unknown force algorithms: {sorted(selected-set(IDENTIFIERS)-{'all'})}")
    return set(IDENTIFIERS) if "all" in selected else selected
