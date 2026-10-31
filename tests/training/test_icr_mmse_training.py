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


def test_scalar_detector_control_preserves_the_csi_training_schedule():
    model = ICRMMSEnet(layers=2, bandwidth=2, detector_mode="budgeted", initial_soft_damping=.8)
    for stage in "ABC":
        configure_stage(model, stage, detector_parameters="scalars")
        for name, value in model.named_parameters():
            correction = name.startswith("corrector.") or name == "eta_logits"
            neural = name.startswith("equalizers.") and (".hidden." in name or ".classifier." in name)
            expected = not neural and (stage == "C" or (correction if stage == "B" else not correction))
            assert value.requires_grad == expected


def test_two_rate_optimizer_keeps_both_detector_components_trainable():
    from src.training.icr_mmse import _optimizer
    model = ICRMMSEnet(layers=2, bandwidth=2, detector_mode="budgeted", initial_soft_damping=.8)
    for stage in "ABC":
        configure_stage(model, stage, detector_parameters="all")
        optimizer = _optimizer(model, stage, .001, .0001, .01)
        groups = {group["name"]: group for group in optimizer.param_groups}
        actual = [id(p) for group in groups.values() for p in group["params"]]
        assert len(actual) == len(set(actual))
        assert set(actual) == {id(p) for p in model.parameters() if p.requires_grad}
        if stage != "B":
            assert groups["physical"]["lr"] == .001
            assert groups["neural"]["lr"] == .00001
            for name, p in model.named_parameters():
                if name.startswith("equalizers."):
                    assert p.requires_grad
                    group = "neural" if ".hidden." in name or ".classifier." in name else "physical"
                    assert id(p) in {id(value) for value in groups[group]["params"]}
        if stage != "A":
            assert groups["correction"]["lr"] == (.0001 if stage == "C" else .001)


def test_two_rate_learning_updates_hidden_head_and_physics(experiment):
    from src.data.compact import CompactDataset
    from src.training.icr_mmse import _optimizer, run_epoch
    directory, _ = experiment
    model = ICRMMSEnet(layers=2, bandwidth=2, detector_mode="budgeted", initial_soft_damping=.8)
    configure_stage(model, "A", detector_parameters="all")
    initial = cpu_state(model)
    optimizer = _optimizer(model, "A", .001, neural_learning_rate_scale=.01)
    with CompactDataset(directory/"train.h5", input_view="compensated") as dataset:
        for _ in range(2):
            run_epoch(model, dataset, "cpu", stage="A", batch_size=1, frame_limit=1,
                      optimizer=optimizer, loss_options=dict(objective="bit_ce", final_weight=.9))
    for i in range(2):
        for component in ("hidden.weight", "classifier.weight", "step_logit"):
            key = f"equalizers.{i}.{component}"
            assert not torch.equal(model.state_dict()[key], initial[key]), key


def test_two_rate_training_resume_keeps_optimizer_identity(experiment):
    directory, root = experiment
    options = dict(model_config=dict(layers=2, bandwidth=2, detector_mode="budgeted", initial_soft_damping=.8),
                   stage_epochs=(1, 0, 1), batch_size=1, frame_limit=1, seed=31,
                   neural_learning_rate_scale=.01, loss_options=dict(objective="bit_ce", final_weight=.9))
    full = train(directory, root/"two_rate_full", DenseBackend(), **options)
    train(directory, root/"two_rate_resume", DenseBackend(), stop_after_epochs=1, **options)
    with pytest.raises(ValueError, match="identity changed"):
        train(directory, root/"two_rate_resume", DenseBackend(), resume=True,
              **dict(options, neural_learning_rate_scale=.1))
    resumed = train(directory, root/"two_rate_resume", DenseBackend(), resume=True, **options)
    a, b = (torch.load(path, weights_only=True) for path in (full, resumed))
    assert a["identity"]["controls"]["neural_learning_rate_scale"] == .01
    assert a["history"] == b["history"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)


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


@pytest.mark.parametrize("grouped_and_bounded", (False, True))
def test_stage_b_continuation_runs_only_c_and_resumes_exactly(experiment, monkeypatch, grouped_and_bounded):
    from src.common.experiment import file_sha256
    from src.training import icr_mmse as training

    directory, root = experiment
    controls = dict(model_config=dict(layers=2, bandwidth=2, noise_mode="diagonal"),
                    stage_epochs=(1, 1, 2), batch_size=1, frame_limit=1, seed=17)
    train(directory, root/"source", DenseBackend(), stop_after_epochs=2, **controls)
    source = root/"source/correction_pretrained.pt"
    before_hash = file_sha256(source)
    initial = torch.load(source, weights_only=True)["model"]
    actual_epoch = training.run_epoch
    calls = []

    def checked_epoch(model, dataset, device, **kwargs):
        assert kwargs["stage"] == "C"
        assert all(parameter.requires_grad for parameter in model.parameters())
        if not calls:
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        calls.append(kwargs["optimizer"] if "optimizer" in kwargs else None)
        if kwargs.get("optimizer") is not None:
            groups = kwargs["optimizer"].param_groups
            if grouped_and_bounded:
                rates = {id(parameter): group["lr"] for group in groups for parameter in group["params"]}
                assert sum(len(group["params"]) for group in groups) == len(rates)
                assert len(rates) == len(list(model.parameters()))
                for name, parameter in model.named_parameters():
                    correction = name.startswith("corrector.") or name == "eta_logits"
                    assert rates[id(parameter)] == (1e-5 if correction else 1e-4)
            else:
                assert {g["lr"] for g in groups} == {1e-4}
        return actual_epoch(model, dataset, device, **kwargs)

    actual_file = h5py.File

    def no_test(path, *args, **kwargs):
        assert not str(path).replace("\\", "/").endswith("/test.h5")
        return actual_file(path, *args, **kwargs)

    monkeypatch.setattr(training, "run_epoch", checked_epoch)
    monkeypatch.setattr(h5py, "File", no_test)
    options = dict(controls, stage_b_checkpoint=source, learning_rate=1e-4)
    if grouped_and_bounded:
        options.update(stage_c_correction_learning_rate=1e-5, stage_c_nmse_max_relative_degradation=0.)
    full = train(directory, root/"continuation", DenseBackend(), **options)
    train(directory, root/"resumed_c", DenseBackend(), stop_after_epochs=1, **options)
    with pytest.raises(ValueError, match="identity changed"):
        train(directory, root/"resumed_c", DenseBackend(), resume=True,
              **dict(options, learning_rate=1e-3))
    resumed = train(directory, root/"resumed_c", DenseBackend(), resume=True, **options)
    a, b = (torch.load(path, weights_only=True) for path in (full, resumed))
    assert [row["stage"] for row in a["history"]] == ["C", "C"]
    assert a["history"] == b["history"]
    if grouped_and_bounded:
        assert a["stage_c_selection"] == b["stage_c_selection"]
        selection = a["stage_c_selection"]
        assert selection["nmse_limit"] == selection["reference_validation"]["corrected_nmse"]
        assert selection["selected_validation"]["corrected_nmse"] <= selection["nmse_limit"]
        with pytest.raises(ValueError, match="identity changed"):
            train(directory, root/"resumed_c", DenseBackend(), resume=True,
                  **dict(options, stage_c_nmse_max_relative_degradation=.01))
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    assert a["identity"]["controls"]["learning_rate"] == 1e-4
    assert a["identity"]["data"]["stage_b_checkpoint"]["sha256"] == before_hash
    assert file_sha256(source) == before_hash
    assert not (full.parent/"warmup.pt").exists()
    assert not (full.parent/"correction_pretrained.pt").exists()


def test_stage_c_selection_uses_stage_b_bound_instead_of_only_initial_csi():
    from src.training.icr_mmse import _selection_key
    baseline = dict(initial_nmse=1., corrected_nmse=.2, ber=.1)
    bad = dict(initial_nmse=1., corrected_nmse=.21, ber=.01)
    good = dict(initial_nmse=1., corrected_nmse=.19, ber=.09)
    assert _selection_key("C", bad) < _selection_key("C", baseline)
    key = lambda metrics: _selection_key("C", metrics, stage_c_nmse_limit=.2)
    assert key(good) < key(baseline) < key(bad)
    assert key(dict(good, corrected_nmse=.2))[0] == 0
    assert key(dict(good, corrected_nmse=.2000001))[0] == 1


def test_stage_c_rejects_ber_gain_with_nmse_regression_and_restores_b(experiment, monkeypatch):
    from src.training import icr_mmse as training
    directory, root = experiment
    controls = dict(model_config=dict(layers=2, bandwidth=2), stage_epochs=(1, 1, 2),
                    batch_size=1, frame_limit=1, seed=17)
    train(directory, root/"source", DenseBackend(), stop_after_epochs=2, **controls)
    source = root/"source/correction_pretrained.pt"
    original = torch.load(source, weights_only=True)["model"]
    actual_epoch = training.run_epoch
    validation_calls = 0

    def prescribed_validation(model, dataset, device, **kwargs):
        nonlocal validation_calls
        metrics = actual_epoch(model, dataset, device, **kwargs)
        if kwargs.get("optimizer") is None:
            metrics.update(initial_nmse=1., corrected_nmse=.2 if validation_calls == 0 else .3,
                           ber=.1 if validation_calls == 0 else .01)
            validation_calls += 1
        return metrics

    monkeypatch.setattr(training, "run_epoch", prescribed_validation)
    path = train(directory, root/"bounded", DenseBackend(), **controls, stage_b_checkpoint=source,
        learning_rate=1e-4, stage_c_correction_learning_rate=1e-5, stage_c_nmse_max_relative_degradation=0.)
    result = torch.load(path, weights_only=True)
    assert result["stage_c_selection"]["selected_epoch"] == 0
    assert all(not entry["nmse_bound_eligible"] for entry in result["history"])
    for name, value in result["model"].items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_stage_b_continuation_rejects_wrong_source(experiment):
    directory, root = experiment
    controls = dict(model_config=dict(layers=2, bandwidth=2), stage_epochs=(1, 1, 1),
                    batch_size=1, frame_limit=1, seed=17)
    train(directory, root/"source", DenseBackend(), stop_after_epochs=2, **controls)
    source = root/"source/correction_pretrained.pt"
    with pytest.raises(ValueError, match="stage-B correction"):
        train(directory, root/"wrong_stage", DenseBackend(),
              stage_b_checkpoint=root/"source/warmup.pt", **controls)
    with pytest.raises(ValueError, match="architecture/configuration"):
        train(directory, root/"wrong_config", DenseBackend(), stage_b_checkpoint=source,
              **dict(controls, model_config=dict(layers=2, bandwidth=3)))
    saved = torch.load(source, weights_only=True)
    saved["identity"]["data"]["train.h5"]["sha256"] = "different-data"
    wrong = root/"wrong_stage_b.pt"
    torch.save(saved, wrong)
    with pytest.raises(ValueError, match="different records/data"):
        train(directory, root/"wrong_data", DenseBackend(), stage_b_checkpoint=wrong, **controls)


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
            for a, b in zip(metrics[0]["per_snr"], metrics[1]["per_snr"]):
                for key in a:
                    if "nmse" in key and a[key] is not None:
                        assert a[key] == pytest.approx(b[key], rel=5e-4, abs=2e-6)
                    else:
                        assert a[key] == b[key]
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
