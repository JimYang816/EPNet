"""Reject apparent full training when the selected detector retained a frozen branch."""
import pytest
import torch

from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.algorithms.icr_mmse.checkpoint import cpu_state
from src.evaluation.icr_unified import detector_audit


def payload():
    torch.manual_seed(13)
    model = ICRMMSEnet(layers=2, bandwidth=2, detector_mode="budgeted", initial_soft_damping=.8)
    state = cpu_state(model)
    for i in range(2):
        for field in ("step_logit", "hidden.weight", "classifier.weight"):
            state[f"equalizers.{i}.{field}"] += .001
    return dict(seed=13, model_config=model.config, model=state, channel="NOF1",
                identity=dict(controls=dict(detector_parameters="all")))


def test_audit_verifies_each_component_and_preserves_rng():
    item = payload()
    before = torch.get_rng_state().clone()
    report = detector_audit(item)
    assert report["passed"]
    assert len(report["layers"]) == 2
    assert torch.equal(before, torch.get_rng_state())


def test_audit_rejects_cross_version_initialization_comparison():
    item = payload()
    item["identity"]["environment"] = dict(torch="different-version")
    with pytest.raises(ValueError, match="training Torch version"):
        detector_audit(item)


@pytest.mark.parametrize("failure", ("scalar_policy", "zero_head", "frozen_hidden", "frozen_physics", "final_only"))
def test_audit_rejects_unified_training_violation(failure):
    item = payload()
    if failure == "scalar_policy":
        item["identity"]["controls"]["detector_parameters"] = "scalars"
    elif failure == "final_only":
        item["identity"]["controls"]["stage_c_final_equalizer_only"] = True
    else:
        torch.manual_seed(13)
        initial = ICRMMSEnet(**item["model_config"]).state_dict()
        field = {"zero_head": "classifier.weight", "frozen_hidden": "hidden.weight",
                 "frozen_physics": "step_logit"}[failure]
        key = "equalizers.0."+field
        item["model"][key] = initial[key].clone()
    assert not detector_audit(item)["passed"]
