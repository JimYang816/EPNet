"""Efficiency control validation; six-process integration lives with unified evaluation."""
import pytest

pytest.importorskip("torch")

from src.evaluation.icr_efficiency import benchmark, _worker


@pytest.mark.parametrize("identifier", ("ep", "robust_ep"))
def test_retired_detectors_are_rejected_before_loading_inputs(identifier):
    with pytest.raises(ValueError, match="Invalid efficiency benchmark controls"):
        benchmark("missing.h5", "missing.json", "missing-icr.pt", "missing-ud.pt", algorithms=(identifier,))
    with pytest.raises(ValueError, match="Unknown efficiency algorithm"):
        _worker(dict(algorithm=identifier, device="cpu"))


def test_default_six_algorithm_comparison_requires_oamp_checkpoint():
    with pytest.raises(ValueError, match="Invalid efficiency benchmark controls"):
        benchmark("missing.h5", "missing.json", "missing-icr.pt", "missing-ud.pt")
