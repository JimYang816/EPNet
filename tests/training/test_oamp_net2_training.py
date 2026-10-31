"""Fixed-split training, exact resume, inference boundaries and six-way reports."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

torch = pytest.importorskip("torch")

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.likelihood import make_whitener
from src.algorithms.oamp_net2.checkpoint import load_model, validate_dataset
from src.algorithms.oamp_net2.inference import equalize_batch
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.oamp_net2_cache import build_features
from src.evaluation import comparison
from src.evaluation.fingerprints import DataIdentity, source_fingerprint
from src.evaluation.inputs import prepare_batch
from src.evaluation.oamp_net2 import evaluate
from src.evaluation.registry import resolve_parameters
from src.training.oamp_net2 import train


@pytest.fixture
def experiment(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with threadpool_limits(2):
            c, prior, chain, archive = fixture()
            c["snr_db"] = [0, 10]
            c["learning_dataset"]["repeats"]["train"] = 4
            directory, generation = tmp_path/"data", tmp_path/"generation"
            generation.mkdir()
            generate(directory, generation, c, archive, chain, prior, DenseBackend(), {"commit": "synthetic"})
            yield directory, tmp_path
    finally:
        torch.set_num_threads(old_threads)


def test_training_does_not_open_test_data_and_resume_is_exact(experiment, monkeypatch):
    directory, root = experiment
    backend = DenseBackend()
    options = dict(epochs=3, patience=20, batch_size=4, frame_limit=2, model_config={"layers": 3})
    file_class = h5py.File
    def protected(path, *args, **kwargs):
        if Path(path).name == "test.h5":
            raise AssertionError("Test split opened during training")
        return file_class(path, *args, **kwargs)
    with monkeypatch.context() as scope:
        scope.setattr(h5py, "File", protected)
        full = train(directory, root/"full", backend, **options)
        train(directory, root/"resumed", backend, stop_after_epochs=1, **options)
        with pytest.raises(ValueError, match="identity changed"):
            train(directory, root/"resumed", backend, resume=True, **dict(options, learning_rate=.002))
        resumed = train(directory, root/"resumed", backend, resume=True, **options)
    a = torch.load(full.parent/"latest.pt", weights_only=True)
    b = torch.load(resumed.parent/"latest.pt", weights_only=True)
    assert a["finished"] and b["finished"] and a["parameters"] == 12
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert [row["validation_loss"] for row in a["history"]] == [row["validation_loss"] for row in b["history"]]
    assert [row["validation_by_snr"] for row in a["history"]] == [row["validation_by_snr"] for row in b["history"]]
    assert set(a["identity"]["data"]) == {"train.h5", "validation.h5", "model.h5"}
    assert a["training_scope"] == "one_channel_all_snr_D1_D2" and a["input_view"] == "compensated"
    assert a["training_blocks"] == 2*2*2
    load_model(resumed)
    raw = dict(a, input_view="raw")
    torch.save(raw, root/"raw.pt")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_model(root/"raw.pt")
    with pytest.raises(ValueError, match="controls"):
        train(directory, root/"bad-view", backend, input_view="raw")
    cache_path = next(path for path in (root/"full/features").glob("train_*") if path.is_dir())
    cached = np.load(cache_path/"projected_matched.npy")
    expected = cached.flat[0]
    cached.flat[0] += 100
    np.save(cache_path/"projected_matched.npy", cached)
    with CompactDataset(directory/"train.h5") as ds:
        guard = DataIdentity({name: directory/name for name in ("train.h5", "model.h5")})
        restored = build_features(ds, root/"full/features", backend, guard, frame_limit=2)
        assert restored.arrays["projected_matched"].flat[0] == pytest.approx(expected)
        restored.close()


def test_label_free_inference_complete_evaluation_and_cache_reuse(experiment, monkeypatch):
    directory, root = experiment
    backend = DenseBackend()
    checkpoint = train(directory, root/"model", backend, epochs=1, batch_size=4,
                       frame_limit=2, model_config={"layers": 3})
    model, payload = load_model(checkpoint)
    with CompactDataset(directory/"test.h5") as ds:
        validate_dataset(payload, ds)
        white = make_whitener(ds._shared["unit_rf_noise_covariance"][:], backend)
        original = h5py.Group.__getitem__
        def protected(group, name):
            if any(label in str(name) for label in ("bits", "classes", "labels")):
                raise AssertionError("Label accessed during inference")
            return original(group, name)
        with monkeypatch.context() as scope:
            scope.setattr(h5py.Group, "__getitem__", protected)
            scope.setattr(CompactDataset, "labels", lambda *a: pytest.fail("Labels entered inference"))
            scope.setattr(CompactDataset, "audit", lambda *a: pytest.fail("Audit entered inference"))
            batch = prepare_batch(ds, 0, [0, 1], 0, backend)
            result = equalize_batch(batch, backend, model=model, whitener=white)
            assert result.symbols.shape == (2, len(ds._indices)) and np.isfinite(result.symbols).all()
    options = dict(batch_size=2, cache_dir=root/"cache", algorithm_parameters={"mmse_pic": {"iterations": 3}})
    report = evaluate(directory/"test.h5", checkpoint, root/"evaluation", backend, **options)
    assert set(report["curves"]) == {"Single-tap MMSE", "Full-matrix LMMSE", "MMSE-PIC", "OAMP-Net2"}
    assert report["bits_per_snr"] == 64
    assert report["oamp_net2"]["sha256"]
    assert (root/"evaluation/ber_snr.png").is_file() and (root/"evaluation/nmse_snr.png").is_file()
    with monkeypatch.context() as scope:
        scope.setattr("src.evaluation.oamp_net2.equalize_batch", lambda *a, **kw: pytest.fail("Cached OAMP-Net2 recomputed"))
        again = evaluate(directory/"test.h5", checkpoint, root/"again", backend, **dict(options, batch_size=1))
    assert again["curves"]["OAMP-Net2"]["provenance"]["reused"]
    assert again["curves"]["OAMP-Net2"]["errors"] == report["curves"]["OAMP-Net2"]["errors"]
    with pytest.raises(ValueError, match="compensated"):
        evaluate(directory/"test.h5", checkpoint, root/"invalid", backend, input_view="raw")
    with pytest.raises(ValueError, match="test split"):
        evaluate(directory/"validation.h5", checkpoint, root/"invalid-split", backend)


def test_pic_addition_and_parameter_change_preserve_existing_mmse_caches(experiment):
    directory, root = experiment
    options = dict(batch_size=2, cache_dir=root/"cache")
    original = comparison.evaluate(directory/"test.h5", root/"old", DenseBackend(), **options)
    extended = comparison.evaluate(directory/"test.h5", root/"new", DenseBackend(),
        algorithm_parameters={"mmse_pic": {"iterations": 2}}, **options)
    for name in original["curves"]:
        assert extended["curves"][name]["provenance"]["reused"]
        assert extended["curves"][name]["provenance"]["cache_key"] == original["curves"][name]["provenance"]["cache_key"]
    changed = comparison.evaluate(directory/"test.h5", root/"changed", DenseBackend(),
        algorithm_parameters={"mmse_pic": {"iterations": 4}}, **options)
    assert all(changed["curves"][name]["provenance"]["reused"] for name in original["curves"])
    assert changed["curves"]["MMSE-PIC"]["provenance"]["cache_key"] != extended["curves"]["MMSE-PIC"]["provenance"]["cache_key"]
    with pytest.raises(ValueError, match="Invalid parameters"):
        resolve_parameters({"mmse_pic": {"unknown": 1}})


def test_unified_six_algorithms_equal_test_counts_and_offline_replot(experiment):
    from src.algorithms.icr_mmse.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, atomic_save, cpu_state
    from src.algorithms.icr_mmse.model import ICRMMSEnet
    from src.training.udnet import train as train_udnet
    from src.evaluation.unified import evaluate as unified, ORDER
    from src.evaluation.comparison_plots import replot
    directory, root = experiment
    backend = DenseBackend()
    oamp = train(directory, root/"oamp", backend, epochs=1, batch_size=4, frame_limit=2, model_config={"layers": 3})
    udnet = train_udnet(directory, root/"udnet", backend, epochs=1, batch_size=4, frame_limit=2)
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=6, center_size=3, hidden_size=4, precision="complex128")
    icr = root/"icr.pt"
    with CompactDataset(directory/"train.h5") as dataset:
        atomic_save(icr, dict(schema=SCHEMA, input_view="compensated", model_config=model.config,
            feedback_layout=FEEDBACK_LAYOUT, inference_code=source_fingerprint(INFERENCE_CODE),
            model=cpu_state(model), training_scope="one_channel_all_snr_D1_D2", channel=dataset.channel,
            data_schema=str(dataset._h.attrs["schema"]), model_identity=str(dataset._h.attrs["model_identity"]),
            data_indices=dataset._indices.tolist(), snr_db=dataset.snr_db.tolist()))
    checkpoints = dict(udnet=udnet, oamp_net2=oamp, icr_mmse=icr)
    report = unified(directory/"test.h5", root/"unified", backend, checkpoints=checkpoints,
        batch_size=2, cache_dir=root/"cache", algorithm_parameters={"mmse_pic": {"iterations": 3}}, bootstrap_samples=16)
    assert tuple(report["curves"]) == ORDER
    assert all(curve["total_bits"] == [64, 64] for curve in report["curves"].values())
    assert report["input_view"] == "compensated"
    assert len(report["nmse"]["corrected_mean_nmse_db"]) == 2
    assert (root/"unified/report.html").is_file()
    assert replot(root/"unified")["curves"] == report["curves"]
    with h5py.File(directory/"test.h5", "r+") as test:
        test["samples/bits"][:] = 1-test["samples/bits"][:]
    (directory/"train.h5").rename(directory/"unavailable-training.h5")
    from src.evaluation.icr_efficiency import benchmark, AVAILABLE_IDENTIFIERS
    efficiency = benchmark(directory/"test.h5", root/"unified/metrics.json", icr, udnet,
        oamp_net2_checkpoint=oamp, batch_size=2, frame_limit=1, threads=2, output=root/"efficiency.json")
    rows = efficiency["algorithms"]
    assert {row["algorithm"] for row in rows} == set(AVAILABLE_IDENTIFIERS)
    assert len({row["process_id"] for row in rows}) == 6
    assert all(row["blocks"] == 4 for row in rows) and not efficiency["test_labels_read"]
    assert all(row["setup_seconds"] > 0 and row["reconstruction_and_all_iterations_seconds"] > 0
               and row["peak_process_rss_bytes"] > 0 for row in rows)
    assert all(row["parameters"] == 0 for row in rows[:3])
    assert all(row["parameters"] > 0 for row in rows[3:])
    assert efficiency["reference_assisted_csi_construction"]
    assert efficiency["initial_provenance_validation_seconds"] > 0
    assert "train.h5" not in efficiency["data_identity"]
    for row in rows:
        assert row["setup_and_prediction_seconds"] == pytest.approx(
            row["setup_seconds"] + row["reconstruction_and_all_iterations_seconds"])
    assert json.loads((root/"efficiency.json").read_text(encoding="utf-8"))["input_view"] == "compensated"
    assert next(row for row in rows if row["algorithm"] == "mmse_pic")["detector_controls"]["iterations"] == 3
    assert next(row for row in rows if row["algorithm"] == "oamp_net2")["parameters"] == 12
    with pytest.raises(FileNotFoundError, match="Missing oamp_net2"):
        unified(directory/"test.h5", root/"missing", backend, checkpoints={"oamp_net2": root/"missing.pt"})
    assert not (root/"missing/metrics.json").exists()
    wrong = dict(report, input_view="raw")
    (root/"wrong.json").write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="compensated CSI comparison report"):
        benchmark(directory/"test.h5", root/"wrong.json", icr, udnet, algorithms=("full_matrix_lmmse",))


def test_record_leakage_and_smoke_checkpoint_are_rejected(experiment):
    directory, root = experiment
    checkpoint = train(directory, root/"model", DenseBackend(), epochs=1, batch_size=4, frame_limit=2)
    _, payload = load_model(checkpoint)
    with CompactDataset(directory/"test.h5") as dataset:
        dataset._config["smoke"] = False
        with pytest.raises(ValueError, match="Smoke"):
            validate_dataset(payload, dataset)
    with h5py.File(directory/"train.h5", "r") as training, h5py.File(directory/"validation.h5", "r+") as validation:
        validation["windows/record"][0] = training["windows/record"][0]
    with pytest.raises(ValueError, match="disjoint"):
        train(directory, root/"leak", DenseBackend(), epochs=1, batch_size=4)
