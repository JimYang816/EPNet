"""Small CPU paired run: shared warmup, equal order, validation then test, plots."""
from copy import deepcopy
import json

import pytest

from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.icr_gate_controls import run_pair, save_comparison, comparison_rows
from tests.training.test_icr_mmse_training import experiment


def test_paired_gate_run_selects_both_models_before_test_and_preserves_conditions(experiment):
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
    result = run_pair(config, channel, 17, root/"pair", DenseBackend(), phase_callback=phases.append, allow_smoke=True)
    assert phases == ["gate_training", "no_gate_training", "gate_test_evaluation", "no_gate_test_evaluation"]
    assert result["identical_warmup_weights"] and result["identical_sampling_schedule"]
    save_comparison([result], root/"comparison")
    for name in ("ber_snr.png", "nmse_snr.png", "seed_summary.json", "comparison.csv"):
        assert (root/"comparison"/name).stat().st_size > 0
    reports = [json.loads((root/"pair"/channel.lower()/"seed_17"/variant/"evaluation/metrics.json").read_text())
               for variant in ("gate", "no_gate")]
    changed = deepcopy(reports[1])
    changed["model_config"]["layers"] += 1
    with pytest.raises(ValueError, match="Only use_gate"):
        comparison_rows(reports[0], changed)
    changed = deepcopy(reports[1])
    changed["per_snr"][0]["by_record"][0]["bits"] += 1
    with pytest.raises(ValueError, match="record sample counts"):
        comparison_rows(reports[0], changed)
