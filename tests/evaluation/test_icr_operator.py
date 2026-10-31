"""Same-checkpoint integration, input leakage guards and isolated memory capture."""
import json

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.icr_mmse.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, cpu_state
from src.algorithms.icr_mmse.dpss import DPSSChannel
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.experiment import file_sha256
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches
from src.evaluation.fingerprints import source_fingerprint
from src.evaluation.icr_operator import compare_operators
from src.training.icr_mmse import run_epoch


@pytest.fixture
def experiment(tmp_path):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        c, prior, chain, archive = fixture()
        c["snr_db"] = [0, 10]
        data, generation = tmp_path/"data", tmp_path/"generation"
        generation.mkdir()
        generate(data, generation, c, archive, chain, prior, DenseBackend(), {"commit": "operator fixture"})
        torch.manual_seed(25)
        model = ICRMMSEnet(layers=2, bandwidth=2, window_size=6, center_size=3,
                           noise_mode="diagonal", precision="complex128")
        with torch.no_grad():
            model.corrector.head.weight[:10].normal_(std=.005)
            for layer in model.equalizers:
                layer.classifier.weight.normal_(std=.002)
        checkpoint = tmp_path/"model.pt"
        with CompactDataset(data/"train.h5", input_view="raw") as dataset:
            torch.save(dict(schema=SCHEMA, feedback_layout=FEEDBACK_LAYOUT, input_view="raw",
                model_config=model.config, model=cpu_state(model), inference_code=source_fingerprint(INFERENCE_CODE),
                training_scope="one_channel_all_snr_D1_D2", channel=dataset.channel,
                model_identity=str(dataset._h.attrs["model_identity"]), data_schema=str(dataset._h.attrs["schema"]),
                data_indices=dataset._indices.tolist(), snr_db=dataset.snr_db.tolist()), checkpoint)
        yield data, checkpoint, model


def test_reader_matches_saved_raw_and_never_opens_truth_or_dense_reconstruction(experiment, monkeypatch):
    directory, _, _ = experiment
    original = h5py.Dataset.__getitem__
    with CompactDataset(directory/"validation.h5", input_view="raw") as ds:
        expected = {(si, d): ds.raw_model_input(si, 0) for si in range(2) for d in range(2)}
        def guard(dataset, selection):
            if any(word in dataset.name for word in ("reference", "projection", "compensation", "bits", "actual_snr")):
                raise AssertionError("Forbidden DPSS input: " + dataset.name)
            return original(dataset, selection)
        monkeypatch.setattr(h5py.Dataset, "__getitem__", guard)
        monkeypatch.setattr(ds, "raw_model_input", lambda *a: pytest.fail("Dense reconstruction requested"))
        for inputs, keys in input_batches(ds, operator_backend="dpss", precision="complex128", frame_limit=1):
            raw = inputs["initial_channel"]
            assert isinstance(raw, DPSSChannel)
            for i, (si, _, d) in enumerate(keys):
                np.testing.assert_allclose(raw.dense()[i].numpy(), expected[si, d]["csi"][d], atol=2e-14)
                np.testing.assert_array_equal(inputs["observations"][i].numpy(), expected[si, d]["observations"][d+1])


def test_dpss_backend_trains_through_current_stage_c(experiment):
    directory, _, model = experiment
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = cpu_state(model)
    with CompactDataset(directory/"train.h5", input_view="raw") as ds:
        metrics = run_epoch(model, ds, "cpu", stage="C", frame_limit=1,
                            operator_backend="dpss", optimizer=optimizer)
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_full_comparison_uses_identical_weights_and_isolated_processes(experiment, tmp_path):
    data, checkpoint, _ = experiment
    before = file_sha256(checkpoint)
    result = compare_operators(data/"test.h5", checkpoint, tmp_path/"comparison", batch_size=1,
                               frame_limit=2, timing_repeats=2, threads=1)
    assert file_sha256(checkpoint) == before
    assert result["checkpoint_sha256"] == before
    assert result["equivalence"]["passed"]
    assert result["final_bit_disagreements"] == 0
    assert result["maximum_probability_difference"] < 1e-10
    assert result["differences"]["ber"] == 0
    assert abs(result["differences"]["nmse_db"]) < 1e-9
    a, b = [result["modes"][mode]["performance"] for mode in ("dense", "dpss")]
    assert a["process_id"] != b["process_id"]
    assert a["blocks"] == b["blocks"] == result["blocks"] == 8
    assert not a["test_truth_reads"] and not b["test_truth_reads"]
    for row in (a, b):
        assert row["peak_cuda_allocated_bytes"] is None
        assert row["peak_process_rss_bytes"] > 0
        assert row["end_to_end_seconds_per_block"] > 0
    run = tmp_path/"comparison"
    for name in ("comparison.json", "equivalence.json", "ber_snr.png", "nmse_snr.png", "runtime_memory.png"):
        assert (run/name).exists()
    assert json.loads((run/"comparison.json").read_text(encoding="utf-8"))["complete"]


def test_native_drift_diagnostic_never_hides_strict_failure_or_skips_double_control(experiment, tmp_path, monkeypatch):
    from src.evaluation import icr_operator
    data, checkpoint, _ = experiment
    reached = []
    double_ok = False

    def worker(kind, options):
        reached.append((kind, options.get("validation_precision")))
        if kind == "gradients":
            return dict(passed=double_ok if options.get("validation_precision") else False)
        raise LookupError("accuracy diagnostic reached")

    monkeypatch.setattr(icr_operator, "_subprocess", worker)
    with pytest.raises(RuntimeError, match="output/gradient equivalence failed"):
        compare_operators(data/"test.h5", checkpoint, tmp_path/"strict")
    assert reached == [("gradients", None)]
    reached.clear()
    with pytest.raises(RuntimeError, match="double-precision control failed"):
        compare_operators(data/"test.h5", checkpoint, tmp_path/"double_failed", diagnose_native_drift=True)
    assert reached == [("gradients", None), ("gradients", "complex128")]
    double_ok = True
    with pytest.raises(LookupError, match="accuracy diagnostic reached"):
        compare_operators(data/"test.h5", checkpoint, tmp_path/"diagnostic", diagnose_native_drift=True)
    assert not json.loads((tmp_path/"diagnostic/equivalence.json").read_text())["passed"]
    assert json.loads((tmp_path/"diagnostic/equivalence_complex128.json").read_text())["passed"]
