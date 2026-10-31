"""Small CPU checks for stage freezing, raw-view checkpoints and exact recovery."""
from __future__ import annotations

import h5py
import pytest
torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.icr_mmse.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, cpu_state, load_model
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.numerics import DenseBackend
from src.evaluation.fingerprints import source_fingerprint
from src.training.icr_mmse import configure_stage, train


@pytest.fixture
def experiment(tmp_path):
    torch.set_num_threads(2)
    with threadpool_limits(2):
        c, prior, chain, archive = fixture()
        c["snr_db"] = [0, 10]
        directory, generation = tmp_path / "data", tmp_path / "generation"
        generation.mkdir()
        generate(directory, generation, c, archive, chain, prior, DenseBackend(), {"commit": "synthetic"})
        yield directory, tmp_path


def test_stage_freezing():
    model = ICRMMSEnet(layers=2, bandwidth=2)
    for stage in "ABC":
        configure_stage(model, stage)
        for name, value in model.named_parameters():
            correction = name.startswith("corrector.") or name == "eta_logits"
            assert value.requires_grad == (stage == "C" or (correction if stage == "B" else not correction))


@pytest.mark.parametrize("resume_stop", (1, 2, 3))
def test_three_stages_resume_and_never_open_test(experiment, monkeypatch, resume_stop):
    directory, root = experiment
    original_file = h5py.File

    def guarded_file(path, *args, **kwargs):
        if str(path).replace("\\", "/").endswith("/test.h5"):
            raise AssertionError("Test data must not enter training")
        return original_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", guarded_file)
    controls = dict(model_config=dict(layers=2, bandwidth=2), stage_epochs=(1, 2, 1),
                    batch_size=1, frame_limit=1, seed=9)
    complete = train(directory, root / "full", DenseBackend(), **controls)
    train(directory, root / "resumed", DenseBackend(), stop_after_epochs=resume_stop, **controls)
    with pytest.raises(ValueError, match="identity changed"):
        train(directory, root / "resumed", DenseBackend(), resume=True, **dict(controls, lambda_h=1.))
    resumed = train(directory, root / "resumed", DenseBackend(), resume=True, **controls)
    a = torch.load(complete, weights_only=True)
    b = torch.load(resumed, weights_only=True)
    assert [h["stage"] for h in a["history"]] == ["A", "B", "B", "C"]
    assert a["input_view"] == "compensated"
    assert a["model_config"]["precision"] == "complex64"
    assert a["feedback_layout"] == FEEDBACK_LAYOUT
    assert all(value.dtype == (torch.complex64 if value.is_complex() else torch.float32)
               for value in a["model"].values())
    assert "test.h5" not in a["identity"]["data"]
    assert a["history"] == b["history"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    warmup, warmup_metadata = load_model(complete.parent / "warmup.pt")
    assert warmup.config["correction_mode"] == "fixed"
    assert not warmup_metadata["inference_correction_enabled"]
    load_model(resumed)
    assert train(directory, root / "resumed", DenseBackend(), resume=True, **controls) == resumed


def test_warmup_reuse_validates_records_and_keeps_fixed_inference(experiment):
    directory, root = experiment
    controls = dict(model_config=dict(layers=2, bandwidth=2), stage_epochs=(1, 1, 1),
                    batch_size=1, frame_limit=1, seed=12)
    train(directory, root / "source", DenseBackend(), stop_after_epochs=1, **controls)
    warmup_path = root / "source" / "warmup.pt"
    saved = torch.load(warmup_path, weights_only=True)
    # Stage-A predictions stay fixed even if its recorded prospective B mode
    # requests multiple repairs. Loading must reset both switches together.
    saved["model_config"].update(correction_mode="once", correction_passes=3)
    alternate = root / "once_warmup.pt"
    torch.save(saved, alternate)
    fixed, _ = load_model(alternate)
    assert fixed.config["correction_mode"] == "fixed"
    assert fixed.config["correction_passes"] == 1
    train(directory, root / "reused", DenseBackend(), warmup_checkpoint=alternate,
          stop_after_epochs=1, **controls)
    copied = torch.load(root / "reused" / "warmup.pt", weights_only=True)
    assert not copied["inference_correction_enabled"]
    for name, value in saved["model"].items():
        if name.startswith("equalizers."):
            torch.testing.assert_close(value, copied["model"][name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="precision differs"):
        train(directory, root / "wrong_precision", DenseBackend(), warmup_checkpoint=alternate,
              **dict(controls, model_config=dict(layers=2, bandwidth=2, precision="complex128")))
    saved["identity"]["data"]["train.h5"]["sha256"] = "different-records"
    wrong = root / "wrong_warmup.pt"
    torch.save(saved, wrong)
    with pytest.raises(ValueError, match="different records/data"):
        train(directory, root / "wrong", DenseBackend(), warmup_checkpoint=wrong, **controls)


def test_checkpoints_reject_old_schema_or_wrong_compact_feature_layout(tmp_path):
    model = ICRMMSEnet(layers=2, bandwidth=16)
    payload = dict(schema=SCHEMA, input_view="compensated", feedback_layout=FEEDBACK_LAYOUT,
        training_scope="one_channel_all_snr_D1_D2", inference_code=source_fingerprint(INFERENCE_CODE),
        model_config=model.config, model=cpu_state(model))
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    restored, _ = load_model(path)
    assert restored.corrector.conv1.in_channels == 467
    for changes in (dict(schema="watermark-icr-mmse-net-raw-v1"),
                    dict(feedback_layout=None), dict(feedback_layout="aligned-band17-noise3")):
        legacy = dict(payload, **changes)
        legacy["model"] = dict(payload["model"])
        legacy["model"]["corrector.conv1.weight"] = torch.zeros(32, 564, 3)
        torch.save(legacy, path)
        with pytest.raises(ValueError, match="feedback layout"):
            load_model(path)


@pytest.mark.parametrize("prefetch", (0, 2))
@pytest.mark.parametrize("candidate_backend", ("dpss", "dense_device"))
def test_compact_pipeline_preserves_inputs_metrics_and_updates(experiment, prefetch, candidate_backend):
    from copy import deepcopy
    from src.data.compact import CompactDataset
    from src.data.icr_training import ReferenceCache
    from src.training.icr_mmse import run_epoch
    directory, _ = experiment
    torch.manual_seed(471)
    original = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4, noise_mode="diagonal")
    with torch.no_grad():
        original.corrector.head.weight[:10].normal_(std=.01)
    with CompactDataset(directory/"train.h5", input_view="raw" if candidate_backend == "dpss" else "compensated") as dataset:
        for stage in "ABC":
            models = [deepcopy(original), deepcopy(original)]
            metrics = []
            for model, backend in zip(models, ("dense", candidate_backend)):
                configure_stage(model, stage)
                updater = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                metrics.append(run_epoch(model, dataset, "cpu", stage=stage, batch_size=2,
                    frame_limit=2, optimizer=updater, operator_backend=backend,
                    prefetch_batches=prefetch, reference_cache=ReferenceCache(2**20)))
            assert metrics[0]["errors"] == metrics[1]["errors"]
            assert metrics[0]["per_snr"] == metrics[1]["per_snr"]
            for name in ("loss", "initial_nmse", "corrected_nmse"):
                if metrics[0][name] is not None:
                    assert metrics[0][name] == pytest.approx(metrics[1][name], rel=5e-4, abs=2e-6)
            for first, second in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(first, second, rtol=5e-4, atol=2e-5)


def test_prefetch_propagates_failure_and_closes_worker(experiment, monkeypatch):
    from threading import enumerate as threads
    from src.data.compact import CompactDataset
    from src.data.icr_training import training_batches
    directory, _ = experiment
    with CompactDataset(directory/"train.h5") as dataset:
        monkeypatch.setattr(CompactDataset, "labels", lambda *a: (_ for _ in ()).throw(ValueError("injected")))
        with pytest.raises(ValueError, match="injected"):
            with training_batches(dataset, prefetch_batches=2, bandwidth=2) as batches:
                next(batches)
    assert not any(t.name == "icr-cpu-prefetch" for t in threads())


def test_pinned_transfer_materializes_expanded_known_symbols(experiment, monkeypatch):
    from src.data.compact import CompactDataset
    from src.data import icr_training
    from src.data.icr_mmse import input_batches
    directory, _ = experiment
    checked = []

    def pinned(value):
        # CUDA pin_memory on torch 2.5 rejects overlapping expand views.
        assert not any(size > 1 and stride == 0 for size, stride in zip(value.shape, value.stride()))
        checked.append(value.shape)
        return value

    monkeypatch.setattr(torch.Tensor, "pin_memory", pinned)
    with CompactDataset(directory/"train.h5") as dataset:
        item, keys = next(input_batches(dataset, batch_size=1, frame_limit=1))
        # The smoke fixture has only one training frame. Repeat its raw input
        # to reproduce the actual CUDA batch-two expanded-pilot layout.
        for name in ("observations", "rf_variance", "initial_channel"):
            item[name] = item[name].repeat((2,)+(1,)*(item[name].ndim-1))
        item["known"] = item["known"].expand(2, -1)
        assert item["known"].stride(0) == 0
        monkeypatch.setattr(icr_training, "input_batches", lambda *a, **k: iter([(item, keys*2)]))
        iterator = icr_training._cpu_batches(dataset, include_reference=False, compact_reference=False,
            bandwidth=2, cache=icr_training.ReferenceCache(), pin_memory=True, batch_size=2, frame_limit=2,
            precision="complex64", operator_backend="dense", snr_indices=[0], frame_indices=None)
        inputs, _, keys = next(iterator)
        assert len(keys) == 2 and inputs["known"].is_contiguous()
        iterator.close()
    assert checked
