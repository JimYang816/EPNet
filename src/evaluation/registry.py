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
    Algorithm("single_tap_mmse", "Single-tap LMMSE", "src.algorithms.MMSE.single_tap"),
    Algorithm("full_matrix_lmmse", "Full-matrix LMMSE", "src.algorithms.MMSE.full_matrix", MATRIX),
)
# Keep the original defaults for callers producing the historical two-MMSE
# report. The unified experiment explicitly opts into the additional comparators.
OPTIONAL_ALGORITHMS = (
    Algorithm("mmse_dfe", "MMSE-DFE", "src.algorithms.mmse_dfe.detector", MATRIX),
    Algorithm("mmse_pic", "MMSE-PIC", "src.algorithms.mmse_pic.detector", MATRIX),
)
AVAILABLE_ALGORITHMS = ALGORITHMS+OPTIONAL_ALGORITHMS
CURVES = tuple(a.label for a in ALGORITHMS)
IDENTIFIERS = tuple(a.identifier for a in AVAILABLE_ALGORITHMS)
SUPPORTED_CURVES = tuple(a.label for a in AVAILABLE_ALGORITHMS)+("UDNet", "OAMP-Net2", "ICR-MMSE-Net")


def normalize_report_labels(report):
    """Translate historical display names without changing numerical provenance."""
    aliases = {"Single-tap MMSE": ALGORITHMS[0].label}
    curves = {}
    for name, curve in report.get("curves", {}).items():
        label = aliases.get(name, name)
        if label in curves:
            raise ValueError(f"Duplicate report curve after label normalization: {label}")
        curves[label] = curve
    normalized = dict(report, curves=curves)
    if "comparison_algorithms" in report:
        normalized["comparison_algorithms"] = [aliases.get(name, name) for name in report["comparison_algorithms"]]
    return normalized


def resolve_parameters(algorithm_parameters=None):
    supplied = algorithm_parameters or {}
    if set(supplied)-set(IDENTIFIERS):
        raise ValueError(f"Unknown algorithms: {sorted(set(supplied)-set(IDENTIFIERS))}")
    result = {}
    for algorithm in AVAILABLE_ALGORITHMS:
        if algorithm in OPTIONAL_ALGORITHMS and algorithm.identifier not in supplied:
            continue
        override = supplied.get(algorithm.identifier, {})
        if algorithm.identifier == "mmse_pic":
            from src.algorithms.mmse_pic.detector import controls
            if not isinstance(override, dict):
                raise ValueError("Invalid parameters for mmse_pic")
            try:
                result[algorithm.identifier] = controls(**override)
            except TypeError as error:
                raise ValueError("Invalid parameters for mmse_pic") from error
        elif not isinstance(override, dict) or override:
            raise ValueError(f"Invalid parameters for {algorithm.identifier}")
        else:
            result[algorithm.identifier] = {}
    return result


def selected_algorithms(parameters):
    return tuple(a for a in AVAILABLE_ALGORITHMS if a.identifier in parameters)


def resolve_force(force):
    selected = {force} if isinstance(force, str) else set(force)
    if selected-set(IDENTIFIERS)-{"all"}:
        raise ValueError(f"Unknown force algorithms: {sorted(selected-set(IDENTIFIERS)-{'all'})}")
    return set(IDENTIFIERS) if "all" in selected else selected
