"""A verified cache migration retains every learned and optimizer value."""
import json

import pytest
import torch

from scripts.run.icr_cache_upgrade import upgrade
from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE
from src.evaluation.fingerprints import source_fingerprint
from src.training.icr_mmse import TRAINING_CODE


def setup_migration(tmp_path):
    after = source_fingerprint(TRAINING_CODE)
    before = dict(after, **{"src.data.icr_training": "old-cache", "src.data.icr_dense_training": "old-input"})
    report = dict(passed=True, input_view="compensated", training_code_before=before,
        training_code_after=after, inference_code=source_fingerprint(INFERENCE_CODE),
        channels={"NOF1": {}, "NCS1": {}})
    report_path = tmp_path/"paired.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    run = tmp_path/"run"
    path = run/"nof1"/"selection"/"candidate"/"latest.pt"
    path.parent.mkdir(parents=True)
    state = dict(identity=dict(code=before, controls=dict(input_view="compensated", operator_backend="dense_device"),
        data={"train.h5": "unchanged"}, environment={"precision": "complex64/float32"}),
        model={"weight": torch.tensor([1.+2.j, 3.-4.j])},
        updater={"step": torch.tensor(17), "exp_avg": torch.tensor([.1, .2])},
        selected={"weight": torch.tensor([5.+6.j])}, rng={"cpu": torch.get_rng_state()},
        stage_index=1, epoch=7, history=[{"stage": "B", "epoch": 7}], finished=False)
    torch.save(state, path)
    return run, path, report_path, report, state


def test_cache_upgrade_preserves_training_state_and_backup(tmp_path):
    run, path, report_path, report, state = setup_migration(tmp_path)
    result = upgrade(run, report_path)
    assert len(result["checkpoints"]) == 1
    assert result["checkpoints"][0]["epoch"] == 7
    saved = torch.load(path, weights_only=True)
    for key in ("model", "updater", "selected", "rng"):
        for name, value in state[key].items():
            torch.testing.assert_close(saved[key][name], value, rtol=0, atol=0)
    for key in ("stage_index", "epoch", "history", "finished"):
        assert saved[key] == state[key]
    for key in ("controls", "data", "environment"):
        assert saved["identity"][key] == state["identity"][key]
    assert saved["identity"]["code"] == report["training_code_after"]
    assert saved["performance_migrations"][0]["report"] == str(report_path)
    backup = torch.load(result["checkpoints"][0]["backup"], weights_only=True)
    assert backup["identity"] == state["identity"]
    assert not upgrade(run, report_path)["checkpoints"]


@pytest.mark.parametrize("change", ("model", "raw", "unverified"))
def test_cache_upgrade_rejects_scientific_or_unverified_changes(tmp_path, change):
    run, path, report_path, report, state = setup_migration(tmp_path)
    if change == "model":
        state["identity"]["code"]["src.algorithms.icr_mmse.model"] = "different-model"
    elif change == "raw":
        state["identity"]["controls"]["input_view"] = "raw"
    else:
        report["passed"] = False
    torch.save(state, path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        upgrade(run, report_path)
    assert path.read_bytes() == previous
