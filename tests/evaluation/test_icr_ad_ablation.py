"""CPU integration: four models, matched budgets, paired data and recovery."""
from copy import deepcopy
import json

import pytest
import torch

from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.icr_ad_ablation import run_quartet, save_comparison, compare_reports
from src.training.icr_mmse import train
from tests.training.test_icr_mmse_training import experiment


def test_quartet_trains_then_tests_with_matched_updates_and_can_resume(experiment):
    directory, root = experiment
    with CompactDataset(directory/"train.h5") as dataset:
        channel = dataset.channel
    config = dict(input_view="compensated", datasets={channel: str(directory)},
        model=dict(bandwidth=2, window_size=8, center_size=4, noise_mode="diagonal"),
        channel_options={channel: dict(layers=2, lambda_h=.1)},
        training=dict(stage_epochs=[1, 1, 1], batch_size=1, reference_cache_mib=2,
            operator_backend="dense_device", sample_order="snr_interleaved", per_snr_acceptance={},
            stage_c_learning_rate=1e-4, stage_c_correction_learning_rate=1e-5),
        evaluation=dict(batch_size=1, bootstrap_samples=20))
    phases = []
    output = root/"ablation"
    result = run_quartet(config, channel, 17, output, DenseBackend(),
                         phase_callback=phases.append, allow_smoke=True)
    assert phases == ["A_preparation", "shared_warmup_training", "B_training", "C_training", "D_training",
                      "A_test_evaluation", "B_test_evaluation", "C_test_evaluation", "D_test_evaluation"]
    run = output/channel.lower()/"seed_17"
    audit = json.loads((run/"training_audit.json").read_text())
    assert audit["detector_training_epochs"] == dict(A=0, B=2, C=0, D=2)
    assert audit["correction_training_epochs"] == dict(A=0, B=0, C=2, D=2)
    assert audit["identical_B_D_warmup"] and audit["identical_C_D_initial_correction"]
    assert audit["test_access"] is False
    assert result["performance"]["A"]["parameters"] == 0
    assert result["performance"]["B"]["parameters"] < result["performance"]["D"]["parameters"]
    save_comparison([result], output)
    for name in ("ber_snr.png", "nmse_snr.png", "comparison.csv", "seed_summary.json"):
        assert (output/name).stat().st_size > 0
    reports = {name: json.loads((run/name/"evaluation/metrics.json").read_text()) for name in "ABCD"}
    for name in "AB":
        for row in reports[name]["per_snr"]:
            assert row["mean_nmse_linear"][-1] == pytest.approx(row["mean_nmse_linear"][0])
    changed = deepcopy(reports)
    changed["C"]["per_snr"][0]["by_record"][0]["bits"] += 1
    with pytest.raises(ValueError, match="coverage differs"):
        compare_reports(changed)
    before = {name: torch.load(run/name/"icr_mmse.pt", weights_only=True)["model"] for name in "ABCD"}
    resumed = run_quartet(config, channel, 17, output, DenseBackend(), resume=True, allow_smoke=True)
    assert result["rows"] == resumed["rows"]
    for name in "ABCD":
        after = torch.load(run/name/"icr_mmse.pt", weights_only=True)["model"]
        assert all(torch.equal(value, after[key]) for key, value in before[name].items())


def test_exact_detector_joint_training_recovers_mid_stage(experiment):
    directory, root = experiment
    options = dict(model_config=dict(layers=2, bandwidth=2, detector_mode="lmmse", noise_mode="diagonal"),
        stage_epochs=(0, 1, 2), batch_size=1, frame_limit=1, seed=19,
        operator_backend="dense_device", sample_order="snr_interleaved", reference_cache_mib=2,
        stage_c_learning_rate=1e-4, stage_c_correction_learning_rate=1e-5)
    full = train(directory, root/"exact_full", DenseBackend(), **options)
    train(directory, root/"exact_resume", DenseBackend(), stop_after_epochs=2, **options)
    resumed = train(directory, root/"exact_resume", DenseBackend(), resume=True, **options)
    a, b = [torch.load(path, weights_only=True) for path in (full, resumed)]
    assert [row["stage"] for row in a["history"]] == ["B", "C", "C"]
    assert a["history"] == b["history"]
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
