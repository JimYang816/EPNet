"""CPU integration check for the existing-B -> C -> test comparison workflow."""
from copy import deepcopy
import json

import pytest
import torch

from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.icr_mmse import evaluate
from src.evaluation.icr_stage_c import comparison_rows, run_channel
from src.training.icr_mmse import train
from tests.training.test_icr_mmse_training import experiment


def test_stage_c_control_inherits_source_and_produces_comparison(experiment):
    directory, root = experiment
    with CompactDataset(directory/"train.h5") as dataset:
        channel = dataset.channel
    source_root = root/"original"
    source = source_root/channel.lower()/"seed_17"
    controls = dict(model_config=dict(layers=2, bandwidth=2, noise_mode="diagonal"),
                    stage_epochs=(1, 1, 1), batch_size=1, frame_limit=1, seed=17,
                    prefetch_batches=0, reference_cache_mib=2)
    original = train(directory, source, DenseBackend(), **controls)
    report = evaluate(directory/"test.h5", original, source/"evaluation", DenseBackend(),
                      batch_size=1, bootstrap_samples=20)
    write_json(source_root/"resolved_config.json", dict(input_view="compensated",
        datasets={channel: str(directory)}, selection=dict(seeds=[17]), training=dict(seed=17),
        evaluation=dict(batch_size=1, bootstrap_samples=20)))
    result = run_channel(source_root, root/"control", channel, DenseBackend(), allow_smoke=True)
    run = root/"control"/channel.lower()
    continuation = json.loads((run/"continuation.json").read_text())
    assert continuation["training_options"]["learning_rate"] == 1e-4
    assert continuation["training_options"]["stage_epochs"] == [1, 1, 1]
    assert continuation["training_options"]["batch_size"] == 1
    assert result["model_config"] == report["model_config"]
    assert result["stage_c_epochs"] == 1
    checkpoint = torch.load(run/"icr_mmse.pt", weights_only=True)
    assert [row["stage"] for row in checkpoint["history"]] == ["C"]
    assert len(result["rows"]) == 2
    for name in ("comparison.json", "comparison.csv", "ber_snr.png", "nmse_snr.png"):
        assert (run/"comparison"/name).stat().st_size > 0
    self_comparison = comparison_rows(report, report)
    assert all(row["ber_delta"] == 0 and row["nmse_delta_db"] == 0 for row in self_comparison)
    wrong = deepcopy(report)
    wrong["data_identity"]["test.h5"]["sha256"] = "different-test"
    with pytest.raises(ValueError, match="data_identity"):
        comparison_rows(report, wrong)
    wrong = deepcopy(report)
    wrong["per_snr"][0]["total_bits"] += 1
    with pytest.raises(ValueError, match="coverage differs"):
        comparison_rows(report, wrong)
    bounded = run_channel(source_root, root/"bounded", channel, DenseBackend(), allow_smoke=True,
        correction_learning_rate=1e-5, nmse_max_relative_degradation=0., previous_root=root/"control")
    assert bounded["correction_learning_rate"] == 1e-5
    assert bounded["validation_nmse_max_relative_degradation"] == 0.
    assert bounded["previous_selection"] == result["candidate_selection"]
    assert bounded["stage_c_selection"]["selected_validation"]["corrected_nmse"] <= bounded["stage_c_selection"]["nmse_limit"]
    assert all("previous_ber" in row and "previous_nmse_db" in row for row in bounded["rows"])
    options = json.loads((root/"bounded"/channel.lower()/"continuation.json").read_text())["training_options"]
    assert options["stage_c_correction_learning_rate"] == 1e-5
    assert options["stage_epochs"] == [1, 1, 1]
