"""Guard against stale model paths and incomplete full-experiment reports."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run.icr_experiment import LABELS, SNR, comparison_config, execute, validate_report


def test_new_joint_checkpoints_replace_historical_weights_and_missing_udnet(tmp_path):
    training = json.loads(Path("configs/training/icr_mmse.json").read_text())
    comparison = json.loads(Path("configs/evaluation/ber_comparison.json").read_text())
    baseline = tmp_path/"existing.pt"
    baseline.touch()
    comparison["checkpoints"]["NOF1"]["udnet"] = str(baseline)
    comparison["checkpoints"]["NCS1"]["udnet"] = str(tmp_path/"missing.pt")
    result = comparison_config(training, comparison, tmp_path)
    assert result["checkpoints"]["NOF1"]["udnet"] == str(baseline)
    assert result["checkpoints"]["NCS1"]["udnet"] == str(tmp_path/"udnet/ncs1/best.pt")
    for ch in training["datasets"]:
        assert result["checkpoints"][ch]["icr_mmse"] == str(tmp_path/"joint"/ch.lower()/"icr_mmse.pt")
    training["input_view"] = "raw"
    with pytest.raises(ValueError, match="compensated"):
        comparison_config(training, comparison, tmp_path)


def report():
    return dict(complete=True, input_view="compensated", channel="NOF1", snr_db=SNR,
                curves={name: dict(total_bits=[1228800]*7, errors=[12]*7, ber=[12/1228800]*7)
                        for name in LABELS}, nmse={})


@pytest.mark.parametrize("failure", ("smoke", "partial", "missing_algorithm", "raw", "snr"))
def test_digest_rejects_incomplete_or_inconsistent_reports(failure):
    item = deepcopy(report())
    if failure == "partial":
        item["curves"][LABELS[0]]["total_bits"][0] -= 1
    elif failure == "missing_algorithm":
        del item["curves"][LABELS[-1]]
    elif failure == "raw":
        item["input_view"] = "raw"
    elif failure == "snr":
        item["snr_db"] = [0]
    else:
        item["smoke"] = True
    with pytest.raises(ValueError):
        validate_report(item, "NOF1")


def test_complete_digest_uses_actual_error_counts():
    result = validate_report(report(), "NOF1")
    assert result["mean_ber"][LABELS[0]] == 12/1228800


def test_failed_phase_preserves_command_log_and_last_phase(tmp_path):
    import subprocess
    (tmp_path/"logs").mkdir()
    state = {}
    with pytest.raises(subprocess.CalledProcessError):
        execute(tmp_path, state, "failure", ["does_not_exist_for_test"])
    progress = json.loads((tmp_path/"progress.json").read_text())
    assert progress["phase"] == "failure" and not progress.get("completed_phases")
    assert "No module named" in (tmp_path/"logs/failure.log").read_text()
