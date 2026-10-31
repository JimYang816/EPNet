import numpy as np
import pytest

from src.evaluation.icr_mmse import RecordMetrics, process_peak_bytes, record_interval
from src.evaluation.icr_selection import sequential_select


def test_bootstrap_counts_physical_records_and_preserves_repeats():
    one = record_interval([3], [100])
    assert one["records"] == 1 and one["lower"] is None
    metrics = RecordMetrics(2)
    for _ in range(10):
        metrics.add(0, 17, [2, 1], 20, [.8, .4, .3])
    metrics.add(0, 33, [3, 4], 20, [.4, .3, .5])
    row = metrics.report([15], bootstrap_samples=64)[0]
    assert row["records"] == 2 and row["blocks"] == 11
    np.testing.assert_allclose(row["ber"], [23/220, 14/220])
    np.testing.assert_allclose(row["mean_nmse_linear"], [8.4/11, 4.3/11, 3.5/11])
    assert row["feedback_ber_deterioration_fraction"] == [1/11]
    assert row["csi_deterioration_fraction"] == [0., 1/11]
    assert row["ber_intervals"][-1]["records"] == 2
    assert process_peak_bytes() > 0


def test_sequential_selection_uses_validation_ber_and_csi_constraint():
    calls = []
    def assess(model, weight, name):
        calls.append((dict(model), weight, name))
        # Lowest BER at weight=.01 is disqualified by worse CSI.
        return dict(ber=weight+.001*model["bandwidth"]+.00001*model["layers"],
                    csi_improved=weight >= .1, corrected_nmse=.2)
    selected = sequential_select(dict(layers=8, bandwidth=16), .1, assess)
    assert selected["lambda_h"] == .1
    assert selected["model_config"] == dict(layers=4, bandwidth=16)
    assert [stage["stage"] for stage in selected["stages"]] == ["lambda_h", "layers"]
    assert [len(stage["candidates"]) for stage in selected["stages"]] == [3, 4]
    assert all(model["bandwidth"] == 16 for model, _, _ in calls)
    assert not selected["test_used"]
    assert all(weight == .1 for _, weight, name in calls if not name.startswith("lambda"))


def test_selection_cannot_fall_through_to_test_when_csi_never_improves():
    with pytest.raises(RuntimeError, match="test scoring is prohibited"):
        sequential_select(dict(layers=8, bandwidth=16), .1,
            lambda *_: dict(ber=.001, csi_improved=False, corrected_nmse=.3))


@pytest.mark.parametrize("precision", ("complex64", "complex128"))
def test_coverage_full_matrix_scoring_and_compensated_mmse_baselines(tmp_path, monkeypatch, precision):
    import torch
    from threadpoolctl import threadpool_limits
    from data_generation.tests.test_compact import fixture
    from data_generation.src.dataset.compact_export import generate
    from src.algorithms.icr_mmse.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, atomic_save, cpu_state
    from src.algorithms.icr_mmse.model import ICRMMSEnet
    from src.common.numerics import DenseBackend
    from src.data.compact import CompactDataset
    from src.evaluation.fingerprints import source_fingerprint
    from src.evaluation import icr_mmse as scoring
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(1):
            c, prior, chain, archive = fixture()
            c["learning_dataset"]["repeats"]["train"] = 4
            directory, generation = tmp_path/"data", tmp_path/"generation"
            generation.mkdir()
            generate(directory, generation, c, archive, chain, prior, DenseBackend(),
                     dict(commit="test", archive_sha256="test"))
            model = ICRMMSEnet(layers=2, bandwidth=2, window_size=6, center_size=3, hidden_size=4,
                              precision=precision)
            checkpoint = tmp_path/"icr.pt"
            with CompactDataset(directory/"train.h5") as dataset:
                atomic_save(checkpoint, dict(schema=SCHEMA, input_view="compensated", model_config=model.config,
                    feedback_layout=FEEDBACK_LAYOUT,
                    inference_code=source_fingerprint(INFERENCE_CODE), model=cpu_state(model),
                    training_scope="one_channel_all_snr_D1_D2", channel=dataset.channel,
                    data_schema=str(dataset._h.attrs["schema"]), model_identity=str(dataset._h.attrs["model_identity"]),
                    data_indices=dataset._indices.tolist(), snr_db=dataset.snr_db.tolist()))
            opened = []
            def tracked(path, **kwargs):
                from pathlib import Path
                opened.append(Path(path).name)
                return CompactDataset(path, **kwargs)
            monkeypatch.setattr(scoring, "CompactDataset", tracked)
            original_batches, original_supervision = scoring.input_batches, scoring.supervision
            expected_complex = torch.complex64 if precision == "complex64" else torch.complex128
            expected_real = torch.float32 if precision == "complex64" else torch.float64
            observed = []
            def tracked_batches(*args, **kwargs):
                assert kwargs["precision"] == precision
                for inputs, keys in original_batches(*args, **kwargs):
                    assert inputs["initial_channel"].dtype == expected_complex
                    assert inputs["rf_variance"].dtype == expected_real
                    observed.append(inputs["initial_channel"].dtype)
                    yield inputs, keys
            def tracked_supervision(*args, **kwargs):
                assert kwargs["precision"] == precision
                targets = original_supervision(*args, **kwargs)
                assert targets["reference_channel"].dtype == expected_complex
                return targets
            monkeypatch.setattr(scoring, "input_batches", tracked_batches)
            monkeypatch.setattr(scoring, "supervision", tracked_supervision)
            covered = scoring.coverage(directory, tmp_path/"coverage", bandwidths=(0, 100),
                                       batch_size=2, precision=precision)
            assert covered["precision"] == precision
            assert opened == ["train.h5", "validation.h5"]
            for split in covered["splits"].values():
                for row in split["per_snr"]:
                    assert 0 <= row["fractions"][0] <= 1
                    assert row["fractions"][1] == pytest.approx(1.)
            accelerated = scoring.coverage(directory, tmp_path/"coverage_fast", bandwidths=(0, 100),
                batch_size=2, precision=precision, operator_backend="dense_device", reference_cache_mib=2)
            for split in ("train", "validation"):
                for before, after in zip(covered["splits"][split]["per_snr"],
                                         accelerated["splits"][split]["per_snr"], strict=True):
                    assert before["record_count"] == after["record_count"]
                    np.testing.assert_allclose(before["fractions"], after["fractions"], rtol=2e-6, atol=1e-8)
                    assert before["error_energy"] == pytest.approx(after["error_energy"], rel=2e-6)
            report = scoring.evaluate(directory/"test.h5", checkpoint, tmp_path/"evaluation", DenseBackend(),
                batch_size=2, include_baselines=True, bootstrap_samples=16,
                cache_dir=tmp_path/"cache")
            assert set(report["curves"]) == {"Single-tap MMSE", "Full-matrix LMMSE", "ICR-MMSE-Net"}
            assert report["input_view"] == "compensated"
            assert observed
            assert report["effective_model_config"]["precision"] == precision
            assert report["numerical_environment"]["precision"] == precision+"/"+str(expected_real).split(".")[-1]
            assert report["comparison_precision"]["traditional_baselines"] == "complex128/float64"
            for row in report["per_snr"]:
                assert row["records"] == 1
                np.testing.assert_allclose(row["mean_nmse_linear"], row["mean_nmse_linear"][0], rtol=1e-12)
                assert row["ber_intervals"][0]["lower"] is None
            assert report["performance"]["preprocessing_and_all_iterations_seconds"] > 0
            assert (tmp_path/"evaluation"/"ber_snr.png").exists()
            assert (tmp_path/"evaluation"/"nmse_snr.png").exists()
            assert "research_target" not in report
            from src.evaluation.icr_selection import match_once_budget
            probe = match_once_budget(directory, checkpoint, DenseBackend(), repeats=(1, 2),
                                      frame_limit=1, timing_repeats=1)
            assert probe["split"] == "validation" and probe["correction_events"] == 1
            assert probe["precision"] == precision
            assert len(probe["trials"]) == 2 and probe["ratio"] > 0
            with pytest.raises(ValueError, match="identical full frame counts"):
                scoring.evaluate(directory/"test.h5", checkpoint, tmp_path/"partial", DenseBackend(),
                                 include_baselines=True, frame_limit=1)
    finally:
        torch.set_num_threads(prior_threads)
