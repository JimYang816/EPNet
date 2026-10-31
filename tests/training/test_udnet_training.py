"""Synthetic compact-v2 end-to-end training, recovery and cache acceptance."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.udnet.checkpoint import load_model
from src.algorithms.udnet.inference import equalize_batch
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.udnet import LocalPreprocessor, WindowPlan
from src.data.udnet_cache import build_features
from src.evaluation.fingerprints import DataIdentity
from src.evaluation.inputs import prepare_batch
from src.evaluation.udnet import evaluate
from src.training.udnet import train
from src.training.udnet_diagnostics import ValidationMetrics
from src.training.udnet_supervision import multilayer_cross_entropy


@pytest.fixture
def experiment(tmp_path):
    torch.set_num_threads(2)
    with threadpool_limits(2):
        c, prior, chain, archive = fixture()
        c["snr_db"] = [0, 10]
        c["learning_dataset"]["repeats"]["train"] = 4
        directory, generation = tmp_path/"data", tmp_path/"generation"
        generation.mkdir()
        generate(directory, generation, c, archive, chain, prior, DenseBackend(), {"commit": "synthetic"})
        yield directory, tmp_path


def test_training_resume_is_exact_and_rejects_changed_identity(experiment):
    directory, root = experiment
    backend = DenseBackend()
    options = dict(epochs=3, patience=20, batch_size=4, frame_limit=2)
    full = train(directory, root/"full", backend, **options)
    partial = train(directory, root/"resumed", backend, stop_after_epochs=1, **options)
    state = torch.load(partial.parent/"latest.pt", weights_only=True)
    assert state["epoch"] == 1 and not state["finished"]
    with pytest.raises(ValueError, match="identity changed"):
        train(directory, root/"resumed", backend, resume=True, **dict(options, learning_rate=.002))
    resumed = train(directory, root/"resumed", backend, resume=True, **options)
    a = torch.load(full.parent/"latest.pt", weights_only=True)
    b = torch.load(resumed.parent/"latest.pt", weights_only=True)
    assert a["finished"] and b["finished"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    assert [h["validation_loss"] for h in a["history"]] == [h["validation_loss"] for h in b["history"]]
    assert [h["validation_metrics"] for h in a["history"]] == [h["validation_metrics"] for h in b["history"]]
    assert a["history"][0]["training_health"]["zero_gradient_layers"] == []
    assert a["preprocessing"]["power_normalization"] == "per_window_mean_active_gram_diagonal"
    assert a["input_view"] == a["identity"]["controls"]["input_view"] == "compensated"
    # Old, unnormalized checkpoints must never enter normalized inference.
    legacy = dict(a, preprocessing={k: v for k, v in a["preprocessing"].items() if k != "power_normalization"})
    torch.save(legacy, root/"legacy.pt")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_model(root/"legacy.pt")
    assert "test.h5" not in a["identity"]["data"]
    assert a["batch_size"] == 4
    assert set(a["random_state"]) == {"torch", "cuda", "numpy", "python", "shuffle"}
    load_model(resumed)
    # A stale feature is rebuilt; the content-bound manifest detects corruption.
    cache_path = next(p for p in (root/"resumed/features").glob("train_*") if p.is_dir())
    value = np.load(cache_path/"matched.npy")
    value.flat[0] += 100
    np.save(cache_path/"matched.npy", value)
    with CompactDataset(directory/"train.h5") as ds:
        guard = DataIdentity({name: directory/name for name in ("train.h5", "model.h5")})
        cache = build_features(ds, root/"resumed/features", backend, guard, frame_limit=2)
        assert cache.manifest["identity"]["input_view"] == "compensated"
        assert cache.arrays["matched"].flat[0] != value.flat[0]
    with h5py.File(directory/"train.h5", "r+") as file:
        file["samples/standard_noise"][0, 0, 0] += .01
    with pytest.raises(ValueError, match="identity changed"):
        train(directory, root/"resumed", backend, resume=True, **options)


def test_independent_evaluation_and_inference_never_read_labels(experiment, monkeypatch):
    directory, root = experiment
    backend = DenseBackend()
    best = train(directory, root/"model", backend, epochs=1, batch_size=4, frame_limit=2)
    model, _ = load_model(best)
    with CompactDataset(directory/"test.h5") as ds:
        prepared = LocalPreprocessor(WindowPlan.build(ds._config["nfft"], ds._indices),
                                     ds._shared["unit_rf_noise_covariance"][:])
        original = h5py.Group.__getitem__
        def guard(group, name):
            if any(key in str(name) for key in ("bits", "classes", "labels")):
                raise AssertionError("Label accessed during UDNet inference")
            return original(group, name)
        with monkeypatch.context() as scoped:
            scoped.setattr(h5py.Group, "__getitem__", guard)
            batch = prepare_batch(ds, 0, [0], 0, backend)
            result = equalize_batch(batch, backend, model=model, preprocessor=prepared)
            assert result.symbols.shape == (1, len(ds._indices))
    # Both independently validated MMSE baselines and shared NMSE are present.
    report = evaluate(directory/"test.h5", best, root/"report", backend, cache_dir=root/"cache")
    assert set(report["curves"]) == {"Single-tap MMSE", "Full-matrix LMMSE", "UDNet"}
    assert report["curves"]["UDNet"]["total_symbols"] == [report["bits_per_snr"]//2]*2
    for name in ("ber_snr.png", "nmse_snr.png", "ser_snr.png", "training_curves.png", "report.html"):
        assert (root/"report"/name).exists()
    again = evaluate(directory/"test.h5", best, root/"again", backend, cache_dir=root/"cache")
    assert all(curve["provenance"]["reused"] for curve in again["curves"].values())
    assert again["nmse"]["provenance"]["reused"]
    assert json.loads((root/"again/metrics.json").read_text(encoding="utf-8"))["complete"]


def test_early_stopping_and_code_identity_rejection(experiment, monkeypatch):
    import importlib
    training = importlib.import_module("src.training.udnet")
    directory, root = experiment
    # Fixed validation loss: initial best, then two non-improving epochs.
    original_epoch = training.run_epoch
    def constant_loss(*args, **kwargs):
        if kwargs.get("metrics") is not None:
            original_epoch(*args, **kwargs)
        return 2.
    monkeypatch.setattr(training, "run_epoch", constant_loss)
    best = train(directory, root/"early", DenseBackend(), epochs=5, patience=2, batch_size=4, frame_limit=1)
    state = torch.load(best.parent/"latest.pt", weights_only=True)
    assert state["epoch"] == 3 and state["bad_epochs"] == 2 and state["finished"]
    assert torch.load(best, weights_only=True)["epoch"] == 1
    original = training.source_fingerprint
    monkeypatch.setattr(training, "source_fingerprint", lambda references: dict(original(references), revision="changed"))
    with pytest.raises(ValueError, match="identity changed"):
        train(directory, root/"early", DenseBackend(), resume=True, epochs=5, patience=2, batch_size=4, frame_limit=1)


def test_validation_metrics_weight_masks_and_snr_groups_across_batches():
    torch.manual_seed(83)
    probabilities = torch.randn(4, 22, 32, 4).softmax(-1)
    labels = torch.randint(0, 4, (4, 32))
    mask = torch.zeros((4, 32), dtype=torch.bool)
    for i in range(4):
        mask[i, 8:10+i] = True
    labels = labels.masked_fill(~mask, -1)
    metrics = ValidationMetrics([0, 25], 4, 22, "cpu")
    for start, stop in [(0, 3), (3, 4)]:
        metrics.observe(np.arange(start, stop), probabilities[start:stop],
                        dict(labels=labels[start:stop], data_mask=mask[start:stop]))
    result = metrics.result()
    expected = multilayer_cross_entropy(probabilities, labels, mask)
    assert sum(result["layer_ce"]) == pytest.approx(float(expected), rel=1e-6)
    for i, row in enumerate(result["per_snr"]):
        sl = slice(2*i, 2*i+2)
        assert row["loss"] == pytest.approx(float(multilayer_cross_entropy(probabilities[sl], labels[sl], mask[sl])), rel=1e-6)
        classes = probabilities[sl, -1].argmax(-1)
        errors = (classes//2 != labels[sl]//2).long()+(classes%2 != labels[sl]%2).long()
        assert row["central_ber"] == pytest.approx(float((errors*mask[sl]).sum()/(2*mask[sl].sum())))
