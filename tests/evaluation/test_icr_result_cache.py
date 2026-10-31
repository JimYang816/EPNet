"""ICR report reuse must survive unrelated work and reject stale/corrupt inputs."""
import json
from pathlib import Path

import h5py
import pytest
import torch
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.icr_mmse.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, atomic_save, cpu_state
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation import icr_cache, icr_mmse
from src.evaluation.fingerprints import source_fingerprint


@pytest.fixture
def experiment(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(1):
            c, prior, chain, archive = fixture()
            c["snr_db"] = [0, 10]
            c["learning_dataset"]["repeats"]["train"] = 4
            directory, generation = tmp_path/"data", tmp_path/"generation"
            generation.mkdir()
            generate(directory, generation, c, archive, chain, prior, DenseBackend(), {"commit": "test"})
            model = ICRMMSEnet(layers=2, bandwidth=2, window_size=6, center_size=3, hidden_size=4,
                              detector_mode="budgeted")
            with CompactDataset(directory/"train.h5") as ds:
                checkpoint = tmp_path/"study/joint"/ds.channel.lower()/"icr_mmse.pt"
                atomic_save(checkpoint, dict(schema=SCHEMA, input_view="compensated", model_config=model.config,
                    feedback_layout=FEEDBACK_LAYOUT, inference_code=source_fingerprint(INFERENCE_CODE),
                    model=cpu_state(model), training_scope="one_channel_all_snr_D1_D2", channel=ds.channel,
                    data_schema=str(ds._h.attrs["schema"]), model_identity=str(ds._h.attrs["model_identity"]),
                    data_indices=ds._indices.tolist(), snr_db=ds.snr_db.tolist()))
            yield directory/"test.h5", checkpoint, tmp_path
    finally:
        torch.set_num_threads(threads)


def test_preexisting_report_seeds_cache_without_forward_or_training(experiment, monkeypatch):
    dataset, checkpoint, root = experiment
    old = root/"study/comparison"/checkpoint.parent.name/"icr_mmse"
    original = icr_mmse.evaluate(dataset, checkpoint, old, DenseBackend(), bootstrap_samples=16)
    before = (old/"metrics.json").read_bytes()
    monkeypatch.setattr(icr_mmse, "evaluate", lambda *a, **k: pytest.fail("Unchanged ICR recomputed"))
    monkeypatch.setattr("src.training.icr_mmse.train", lambda *a, **k: pytest.fail("Evaluation trained a model"))
    moved = root/"moved-checkpoint.pt"
    moved.write_bytes(checkpoint.read_bytes())
    for run, batch_size, weight in ((root/"imported", 1, checkpoint), (root/"different-output", 4, moved)):
        report = icr_cache.evaluate(dataset, weight, run, DenseBackend(), cache_dir=root/"cache",
                                    batch_size=batch_size, bootstrap_samples=16)
        assert report["evaluation_cache"]["reused"]
        assert report["curves"] == original["curves"]
        assert report["per_snr"] == original["per_snr"]
        assert report["checkpoint"] == str(weight.resolve())
        assert report["evaluation_cache"]["measurement_source_report"] == str((old/"metrics.json").resolve())
        assert (run/"ber_snr.png").exists() and (run/"nmse_snr.png").exists()
    assert (old/"metrics.json").read_bytes() == before


@pytest.mark.parametrize("change", ("checkpoint", "test_data", "corrupt_cache", "force"))
def test_changed_inputs_or_force_do_not_reuse_stale_report(experiment, change):
    dataset, checkpoint, root = experiment
    options = dict(cache_dir=root/"cache", bootstrap_samples=16)
    original = icr_cache.evaluate(dataset, checkpoint, root/"first", DenseBackend(), **options)
    if change == "checkpoint":
        payload = torch.load(checkpoint, weights_only=True)
        payload["model"]["equalizers.0.step_logit"] += .01
        atomic_save(checkpoint, payload)
    elif change == "test_data":
        with h5py.File(dataset, "r+") as handle:
            handle["samples/bits"][:] = 1-handle["samples/bits"][:]
    elif change == "corrupt_cache":
        Path(original["evaluation_cache"]["cache_file"]).write_text("{}")
    report = icr_cache.evaluate(dataset, checkpoint, root/"second", DenseBackend(), force=change == "force", **options)
    assert not report["evaluation_cache"]["reused"]
    if change == "checkpoint":
        assert report["checkpoint_sha256"] != original["checkpoint_sha256"]
    if change == "test_data":
        for before, after in zip(original["per_snr"], report["per_snr"]):
            assert before["bit_errors"][-1]+after["bit_errors"][-1] == before["total_bits"]


@pytest.mark.parametrize("change", ("partial", "code", "raw"))
def test_invalid_historical_report_is_not_imported(experiment, change):
    dataset, checkpoint, root = experiment
    old = root/"study/comparison"/checkpoint.parent.name/"icr_mmse"
    candidate = icr_mmse.evaluate(dataset, checkpoint, old, DenseBackend(), bootstrap_samples=16)
    if change == "partial":
        candidate["per_snr"][0]["total_bits"] -= 1
    elif change == "code":
        candidate["code"] = {}
    else:
        candidate["input_view"] = "raw"
    (old/"metrics.json").write_text(json.dumps(candidate))
    report = icr_cache.evaluate(dataset, checkpoint, root/"new", DenseBackend(), cache_dir=root/"cache",
                                bootstrap_samples=16)
    assert not report["evaluation_cache"]["reused"]
