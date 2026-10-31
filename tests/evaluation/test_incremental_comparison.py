"""Selective execution, source invalidation and failure-safe publication."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation import comparison, fingerprints, nmse
from src.evaluation.comparison_plots import replot
from src.evaluation.registry import ALGORITHMS, resolve_parameters
from src.evaluation.result_cache import CacheEntry, FileLock


@pytest.fixture
def experiment(tmp_path):
    with threadpool_limits(2):
        c, prior, chain, archive = fixture()
        c["learning_dataset"]["repeats"]["train"] = 4
        data, generation = tmp_path/"data", tmp_path/"generation"
        generation.mkdir()
        generate(data, generation, c, archive, chain, prior, DenseBackend(), {"commit": "fixture"})

        def run(name="run", **kwargs):
            return comparison.evaluate(data/"test.h5", tmp_path/name, DenseBackend(), batch_size=kwargs.pop("batch_size", 2),
                                       iterations=2, cache_dir=tmp_path/"cache", **kwargs)

        yield SimpleNamespace(root=tmp_path, data=data, run=run)


def watch(monkeypatch):
    calls = {a.identifier: 0 for a in ALGORITHMS}
    for algorithm in ALGORITHMS:
        module = importlib.import_module(algorithm.module)
        original = module.equalize_batch

        def wrapper(*args, _original=original, _identifier=algorithm.identifier, **kwargs):
            calls[_identifier] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, "equalize_batch", wrapper)
    return calls


def changed(calls):
    return {name for name, count in calls.items() if count}


def forbid(*args, **kwargs):
    raise AssertionError("A cached computation was executed")


def source_edit(monkeypatch, experiment, reference, before, after):
    """Supply an edited source file without changing the repository under test."""
    module, _, _ = reference.partition(":")
    original_find = importlib.util.find_spec
    original = Path(original_find(module).origin).read_text(encoding="utf-8-sig")
    assert before in original
    path = experiment.root/"edited.py"
    path.write_text(original.replace(before, after, 1), encoding="utf-8")

    def find(name, *args, **kwargs):
        return SimpleNamespace(origin=str(path)) if name == module else original_find(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find)


def test_cross_directory_and_batch_thread_changes_reuse_without_computation(experiment, monkeypatch):
    first = experiment.run()
    calls = watch(monkeypatch)
    monkeypatch.setattr(comparison, "calibrate", forbid)
    monkeypatch.setattr(comparison, "prepare_batch", forbid)
    with threadpool_limits(1):
        second = experiment.run("another", batch_size=1)
    assert not changed(calls)
    for name, curve in first["curves"].items():
        updated = second["curves"][name]
        assert updated["provenance"]["reused"]
        assert updated["provenance"]["generations"] == curve["provenance"]["generations"]
        assert updated["errors"] == curve["errors"]
    assert (experiment.root/"another/nmse_snr.png").is_file()


@pytest.mark.parametrize("identifier", ["ep", "robust_ep"])
def test_independent_parameters_do_not_recalibrate_or_recompute_other_algorithms(experiment, monkeypatch, identifier):
    first = experiment.run()
    calls = watch(monkeypatch)
    monkeypatch.setattr(comparison, "calibrate", forbid)
    updated = experiment.run("changed", algorithm_parameters={identifier: {"iterations": 3}})
    assert changed(calls) == {identifier}
    assert updated["robust_covariance"]["provenance"]["cache_key"] == first["robust_covariance"]["provenance"]["cache_key"]
    other = "robust_ep" if identifier == "ep" else "ep"
    assert updated["parameters"][other]["iterations"] == 2


@pytest.mark.parametrize("identifier", [a.identifier for a in ALGORITHMS])
def test_independent_source_edits_invalidate_only_the_owner(experiment, monkeypatch, identifier):
    experiment.run()
    calls = watch(monkeypatch)
    algorithm = next(a for a in ALGORITHMS if a.identifier == identifier)
    source_edit(monkeypatch, experiment, algorithm.module, "def equalize_batch(", "COMPUTATION_REVISION = 2\n\ndef equalize_batch(")
    monkeypatch.setattr(comparison, "calibrate", forbid)
    experiment.run("edited")
    assert changed(calls) == {identifier}


@pytest.mark.parametrize("reference,before,after,expected", [
    ("src.algorithms.likelihood:likelihood_batch", "h = whitener @", "h = 1.0*whitener @", {"full_matrix_lmmse", "ep", "robust_ep"}),
    ("src.algorithms.likelihood:real_likelihood", "return (2*np.concatenate", "return (2.0*np.concatenate", {"ep", "robust_ep"}),
    ("src.evaluation.inputs", "channel = reference+alpha*", "channel = reference+1.0*alpha*", {a.identifier for a in ALGORITHMS}),
    ("src.evaluation.covariance", "covariance *= 1-shrinkage", "covariance *= 1.0-shrinkage", {"robust_ep"}),
])
def test_shared_computation_invalidates_only_actual_dependents(experiment, monkeypatch, reference, before, after, expected):
    experiment.run()
    calls = watch(monkeypatch)
    source_edit(monkeypatch, experiment, reference, before, after)
    experiment.run("shared-edit")
    assert changed(calls) == expected


def test_comments_docstrings_and_formatting_do_not_invalidate(experiment, monkeypatch):
    experiment.run()
    calls = watch(monkeypatch)
    source_edit(monkeypatch, experiment, "src.algorithms.ep.real", "import numpy as np",
                "# Revised explanation only.\n\nimport numpy    as    np")
    experiment.run("comment-edit")
    assert not changed(calls)
    assert fingerprints.normalized_source('"old"\ndef f():\n    "doc"\n    return 1\n') == fingerprints.normalized_source(
        '# text\n"new"\ndef f( ):\n    "changed"\n    return (1)\n')


def test_plotting_source_is_excluded_from_computation_identity(experiment, monkeypatch):
    experiment.run()
    calls = watch(monkeypatch)
    source_edit(monkeypatch, experiment, "src.evaluation.comparison_plots", '"#667085"', '"#667086"')
    experiment.run("plot-edit")
    assert not changed(calls)


def test_moving_identical_data_preserves_cache_keys(experiment, monkeypatch):
    original = experiment.run()
    moved = experiment.root/"moved"
    experiment.data.rename(moved)
    calls = watch(monkeypatch)
    updated = comparison.evaluate(moved/"test.h5", experiment.root/"moved-run", DenseBackend(),
                                  batch_size=2, iterations=2, cache_dir=experiment.root/"cache")
    assert not changed(calls)
    assert updated["data_identity"] == original["data_identity"]


@pytest.mark.parametrize("split,expected", [("train", {"robust_ep"}), ("test", {a.identifier for a in ALGORITHMS})])
def test_content_hash_detects_in_place_data_changes(experiment, monkeypatch, split, expected):
    first = experiment.run()
    calls = watch(monkeypatch)
    with h5py.File(experiment.data/f"{split}.h5", "r+") as h:
        h["samples/bits"][0, 0, 0, 0] ^= np.uint8(1)
    updated = experiment.run("data-edit")
    assert changed(calls) == expected
    assert updated["data_identity"][f"{split}.h5"]["sha256"] != first["data_identity"][f"{split}.h5"]["sha256"]
    if split == "train":
        assert updated["nmse"]["provenance"]["reused"]


def test_numerical_environment_change_invalidates_results(experiment, monkeypatch):
    experiment.run()
    calls = watch(monkeypatch)
    original = fingerprints.numerical_environment
    monkeypatch.setattr(fingerprints, "numerical_environment", lambda backend: dict(original(backend), revision="changed"))
    experiment.run("environment-edit")
    assert changed(calls) == {a.identifier for a in ALGORITHMS}


@pytest.mark.parametrize("forced", [("ep",), ("single_tap_mmse", "robust_ep"), ("all",)])
def test_force_still_publishes_all_four_curves(experiment, monkeypatch, forced):
    experiment.run()
    calls = watch(monkeypatch)
    monkeypatch.setattr(comparison, "calibrate", forbid)
    updated = experiment.run("forced", force=forced)
    assert changed(calls) == ({a.identifier for a in ALGORITHMS} if "all" in forced else set(forced))
    assert len(updated["curves"]) == 4


def test_failed_forced_attempt_preserves_report_and_resumes_only_missing_blocks(experiment, monkeypatch):
    original = experiment.run()
    saved = {name: (experiment.root/"run"/name).read_bytes() for name in ("metrics.json", "ber_snr.png", "nmse_snr.png")}
    ep_module = importlib.import_module("src.algorithms.ep.real")
    equalize = ep_module.equalize_batch
    attempts = []

    def fail_second(batch, *args, **kwargs):
        attempts.append(len(batch.variances))
        if len(attempts) == 2:
            raise RuntimeError("deliberate detector failure")
        return equalize(batch, *args, **kwargs)

    with monkeypatch.context() as broken:
        broken.setattr(ep_module, "equalize_batch", fail_second)
        with pytest.raises(RuntimeError, match="deliberate"):
            experiment.run(force=["ep"])
    for name, content in saved.items():
        assert (experiment.root/"run"/name).read_bytes() == content
    calls = watch(monkeypatch)
    resumed = experiment.run()
    assert changed(calls) == {"ep"}
    ep = resumed["curves"]["EP"]
    assert ep["provenance"]["computed_blocks"] == 26
    assert ep["provenance"]["reused_blocks"] == 2
    assert sum(g["blocks"] for g in ep["provenance"]["generations"]) == 28
    for name in original["curves"]:
        assert resumed["curves"][name]["errors"] == original["curves"][name]["errors"]


def test_corrupt_completed_block_is_recomputed_without_repeating_valid_blocks(experiment, monkeypatch):
    original = experiment.run()
    path = original["curves"]["EP"]["provenance"]["cache_file"]
    with h5py.File(path, "r+") as h:
        h["bit_errors"][0, 0, 0] += 1
    calls = watch(monkeypatch)
    updated = experiment.run("repaired")
    assert changed(calls) == {"ep"}
    ep = updated["curves"]["EP"]
    assert ep["provenance"]["computed_blocks"] == 1
    assert ep["provenance"]["reused_blocks"] == 27
    assert ep["errors"] == original["curves"]["EP"]["errors"]


def test_unreadable_cache_is_a_miss_and_lock_prevents_concurrent_writers(experiment, monkeypatch):
    original = experiment.run()
    path = Path(original["curves"]["Single-tap MMSE"]["provenance"]["cache_file"])
    path.write_bytes(b"broken HDF5")
    calls = watch(monkeypatch)
    updated = experiment.run("repaired")
    assert changed(calls) == {"single_tap_mmse"}
    assert updated["curves"]["Single-tap MMSE"]["errors"] == original["curves"]["Single-tap MMSE"]["errors"]
    lock_path = path.parent/"entry.lock"
    with FileLock(lock_path):
        with pytest.raises(RuntimeError, match="in use"):
            with FileLock(lock_path):
                pass
    with FileLock(lock_path):
        pass


def test_plot_only_needs_neither_data_nor_backend_and_preserves_provenance(experiment, monkeypatch):
    original = experiment.run()
    experiment.data.rename(experiment.root/"unavailable-data")
    calls = watch(monkeypatch)
    monkeypatch.setattr(comparison, "calibrate", forbid)
    assert replot(experiment.root/"run") == original
    command = importlib.import_module("scripts.evaluate.compare")
    config = experiment.root/"plot-config.json"
    config.write_text(json.dumps(dict(datasets={"NCS1": "missing.h5"}, output=str(experiment.root))), encoding="utf-8")
    (experiment.root/"run").rename(experiment.root/"ncs1")
    monkeypatch.setattr(command, "DenseBackend", forbid)
    monkeypatch.setattr("sys.argv", ["compare", "--config", str(config), "--device", "cuda", "--plot-only"])
    command.main()
    assert not changed(calls)
    assert json.loads((experiment.root/"ncs1/metrics.json").read_text(encoding="utf-8")) == original


def test_nmse_matches_direct_full_csi_linear_average(experiment):
    report = experiment.run()
    with CompactDataset(experiment.data/"test.h5") as dataset:
        expected = [10*np.log10(np.mean([dataset.audit(si, fi)["true_nmse_linear"] for fi in range(dataset.frame_count)]))
                    for si in range(len(dataset.snr_db))]
    np.testing.assert_allclose(report["nmse"]["mean_nmse_db"], expected, atol=2e-12)


def test_matching_acceptance_statistics_are_reused(experiment, monkeypatch):
    files = fingerprints.DataIdentity({name: experiment.data/name for name in ("test.h5", "model.h5")}).files
    with CompactDataset(experiment.data/"test.h5") as dataset:
        values = [nmse.mean_nmse_db(dataset, si) for si in range(len(dataset.snr_db))]
        manifest = dict(complete=True, protocol=dataset._h.attrs["schema"], channel=dataset.channel,
                        config_hash=dataset._h.attrs["config_hash"], snr_db=dataset.snr_db.tolist(), files=files,
                        splits={"test": dict(mean_nmse_db=values, checked_frames=dataset.frame_count)})
    (experiment.data/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(nmse, "mean_nmse_db", forbid)
    report = experiment.run()
    assert report["nmse"]["mean_nmse_db"] == values


def test_parameter_validation_and_legacy_precedence():
    values = resolve_parameters({"ep": {"iterations": 4}}, iterations=3, damping=.4)
    assert values["ep"] == dict(iterations=4, damping=.4, epsilon=5e-7)
    assert values["robust_ep"] == dict(iterations=3, damping=.4, epsilon=5e-7)
    with pytest.raises(ValueError, match="Unknown algorithms"):
        resolve_parameters({"unknown": {}})
    with pytest.raises(ValueError, match="Invalid parameters"):
        resolve_parameters({"ep": {"iterations": 0}})


def test_cli_rejects_force_combined_with_plot_only(monkeypatch):
    command = importlib.import_module("scripts.evaluate.compare")
    monkeypatch.setattr("sys.argv", ["compare", "--plot-only", "--force", "ep"])
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2


def test_legacy_checkpoint_is_preserved_and_not_imported(experiment, monkeypatch):
    run = experiment.root/"run"
    run.mkdir()
    legacy = run/"checkpoint.h5"
    legacy.write_bytes(b"historical checkpoint with no source fingerprint")
    calls = watch(monkeypatch)
    experiment.run()
    assert changed(calls) == {a.identifier for a in ALGORITHMS}
    assert legacy.read_bytes() == b"historical checkpoint with no source fingerprint"


def test_report_publication_failure_rolls_back_all_artifacts(experiment, monkeypatch):
    experiment.run()
    run = experiment.root/"run"
    saved = {name: (run/name).read_bytes() for name in ("metrics.json", "ber_snr.png", "nmse_snr.png")}
    replace = Path.replace

    def fail_nmse(self, target):
        if self.name == "nmse_snr.png" and self.parent.name.startswith(".comparison-"):
            raise OSError("deliberate publication failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_nmse)
    with pytest.raises(OSError, match="publication"):
        experiment.run(force=["ep"])
    for name, content in saved.items():
        assert (run/name).read_bytes() == content


def test_mid_run_data_mutation_discards_the_new_cache_attempt(tmp_path):
    changed_data = False

    def guard():
        if changed_data:
            raise RuntimeError("Dataset changed during evaluation")

    entry = CacheEntry(tmp_path, "unit", {"data": "before"}, {"counts": ((2,), "int64")},
                       (2,), {"id": "session"}, data_guard=guard)
    with pytest.raises(RuntimeError, match="Dataset changed"):
        with entry:
            entry.commit((0,), counts=1)
            changed_data = True
    assert not entry.work.exists()
    assert not entry.path.exists()
    changed_data = False
    with entry:
        assert not entry.complete.any()
