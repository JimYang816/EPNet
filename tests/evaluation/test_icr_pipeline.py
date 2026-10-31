"""Verify current channel training and shared-pretraining provenance."""
import json
from pathlib import Path

import pytest

from src.common.numerics import DenseBackend
from src.training.icr_workflow import DEFAULT_CONFIG, resolve_config, run_channel


@pytest.mark.parametrize("channel", ("NOF1", "NCS1"))
def test_current_pipeline_trains_both_branches_from_same_all_parameter_warmup(tmp_path, monkeypatch, channel):
    from src.training import icr_mmse
    calls = []
    def train(directory, run, backend, **kwargs):
        calls.append((Path(run), kwargs))
        return Path(run)/"icr_mmse.pt"
    monkeypatch.setattr(icr_mmse, "train", train)
    result = run_channel(DEFAULT_CONFIG, channel, DenseBackend(), output=tmp_path, resume=True, frame_limit=1)
    assert list(result) == ["warmup", "fixed", "joint"]
    assert [p.parent.name for p, _ in calls] == ["warmup", "fixed", "joint"]
    assert [k["stage_epochs"] for _, k in calls] == [[10, 0, 0], [10, 0, 10], [10, 10, 10]]
    assert [k["model_config"]["correction_mode"] for _, k in calls] == ["fixed", "fixed", "iterative"]
    assert "warmup_checkpoint" not in calls[0][1]
    for _, controls in calls:
        assert controls["detector_parameters"] == "all"
        assert controls["frame_limit"] == 1 and controls["resume"]
    for _, controls in calls[1:]:
        assert Path(controls["warmup_checkpoint"]) == tmp_path/"warmup"/channel.lower()/"warmup.pt"
    assert calls[0][1]["learning_rate"] == (.001 if channel == "NOF1" else .0003)
    assert calls[2][1]["learning_rate"] == .001
    assert calls[2][1]["lambda_h"] == (.01 if channel == "NOF1" else 1.)
    assert calls[0][1]["neural_learning_rate_scale"] == (.01 if channel == "NOF1" else 1.)


def test_channel_models_have_same_architecture_and_positive_neural_learning_rates():
    configs = [resolve_config(channel=c) for c in ("NOF1", "NCS1")]
    models = [dict(c["model"]) for c in configs]
    for m in models:
        m.pop("initial_csi_loading")
    assert models[0] == models[1]
    assert models[0]["detector_mode"] == "budgeted" and models[0]["layers"] == 8
    assert all(c["training"]["neural_learning_rate_scale"] > 0 for c in configs)


@pytest.mark.parametrize("mutation", ("scalar", "old_detector", "raw"))
def test_current_workflow_rejects_retired_training_modes_before_data_access(tmp_path, mutation):
    config = json.loads(DEFAULT_CONFIG.read_text())
    if mutation == "scalar":
        config["training"]["detector_parameters"] = "scalars"
    elif mutation == "old_detector":
        config["model"]["detector_mode"] = "unfolded"
    else:
        config["input_view"] = "raw"
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        resolve_config(path, channel="NOF1")
