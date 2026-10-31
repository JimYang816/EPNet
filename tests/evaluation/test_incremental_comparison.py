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
from src.evaluation.registry import ALGORITHMS, resolve_parameters, resolve_force
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
                                       cache_dir=tmp_path/"cache", **kwargs)

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


def test_comparison_needs_neither_retired_detectors_nor_training_data(experiment, monkeypatch):
    original_import = importlib.import_module
    def current_import(name, *args, **kwargs):
        if name in ("src.algorithms.ep.real", "src.algorithms.robust_ep.real", "src.evaluation.covariance"):
            pytest.fail("Retired implementation imported: " + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", current_import)
    (experiment.data/"train.h5").rename(experiment.data/"unavailable-training.h5")
    report = experiment.run()
    assert set(report["curves"]) == {"Single-tap LMMSE", "Full-matrix LMMSE"}
    assert set(report["data_identity"]) == {"test.h5", "model.h5"}
    assert "robust_covariance" not in report and "ep" not in report
    assert not (experiment.root/"cache/covariance").exists()


@pytest.mark.parametrize("identifier", [a.identifier for a in ALGORITHMS])
def test_independent_source_edits_invalidate_only_the_owner(experiment, monkeypatch, identifier):
    experiment.run()
    calls = watch(monkeypatch)
    algorithm = next(a for a in ALGORITHMS if a.identifier == identifier)
    source_edit(monkeypatch, experiment, algorithm.module, "def equalize_batch(", "COMPUTATION_REVISION = 2\n\ndef equalize_batch(")
    experiment.run("edited")
    assert changed(calls) == {identifier}


@pytest.mark.parametrize("reference,before,after,expected", [
    ("src.algorithms.likelihood:likelihood_batch", "h = whitener @", "h = 1.0*whitener @", {"full_matrix_lmmse"}),
    ("src.evaluation.inputs", "channel = reference+alpha*", "channel = reference+1.0*alpha*", {a.identifier for a in ALGORITHMS}),
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
    source_edit(monkeypatch, experiment, "src.algorithms.MMSE.full_matrix", "import numpy as np",
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
                                  batch_size=2, cache_dir=experiment.root/"cache")
    assert not changed(calls)
    assert updated["data_identity"] == original["data_identity"]


@pytest.mark.parametrize("split,expected", [("train", set()), ("test", {a.identifier for a in ALGORITHMS})])
def test_content_hash_detects_in_place_data_changes(experiment, monkeypatch, split, expected):
    first = experiment.run()
    calls = watch(monkeypatch)
    with h5py.File(experiment.data/f"{split}.h5", "r+") as h:
        h["samples/bits"][0, 0, 0, 0] ^= np.uint8(1)
    updated = experiment.run("data-edit")
    assert changed(calls) == expected
    if split == "train":
        assert "train.h5" not in updated["data_identity"]
        assert updated["nmse"]["provenance"]["reused"]
    else:
        assert updated["data_identity"][f"{split}.h5"]["sha256"] != first["data_identity"][f"{split}.h5"]["sha256"]


def test_numerical_environment_change_invalidates_results(experiment, monkeypatch):
    experiment.run()
    calls = watch(monkeypatch)
    original = fingerprints.numerical_environment
    monkeypatch.setattr(fingerprints, "numerical_environment", lambda backend: dict(original(backend), revision="changed"))
    experiment.run("environment-edit")
    assert changed(calls) == {a.identifier for a in ALGORITHMS}


@pytest.mark.parametrize("forced", [("full_matrix_lmmse",), ("single_tap_mmse", "full_matrix_lmmse"), ("all",)])
def test_force_still_publishes_both_mmse_curves(experiment, monkeypatch, forced):
    experiment.run()
    calls = watch(monkeypatch)
    updated = experiment.run("forced", force=forced)
    assert changed(calls) == ({a.identifier for a in ALGORITHMS} if "all" in forced else set(forced))
    assert len(updated["curves"]) == 2


def test_failed_forced_attempt_preserves_report_and_resumes_only_missing_blocks(experiment, monkeypatch):
    original = experiment.run()
    saved = {name: (experiment.root/"run"/name).read_bytes() for name in ("metrics.json", "ber_snr.png", "nmse_snr.png")}
    matrix_module = importlib.import_module("src.algorithms.MMSE.full_matrix")
    equalize = matrix_module.equalize_batch
    attempts = []

    def fail_second(batch, *args, **kwargs):
        attempts.append(len(batch.variances))
        if len(attempts) == 2:
            raise RuntimeError("deliberate detector failure")
        return equalize(batch, *args, **kwargs)

    with monkeypatch.context() as broken:
        broken.setattr(matrix_module, "equalize_batch", fail_second)
        with pytest.raises(RuntimeError, match="deliberate"):
            experiment.run(force=["full_matrix_lmmse"])
    for name, content in saved.items():
        assert (experiment.root/"run"/name).read_bytes() == content
    calls = watch(monkeypatch)
    resumed = experiment.run()
    assert changed(calls) == {"full_matrix_lmmse"}
    matrix = resumed["curves"]["Full-matrix LMMSE"]
    assert matrix["provenance"]["computed_blocks"] == 26
    assert matrix["provenance"]["reused_blocks"] == 2
    assert sum(g["blocks"] for g in matrix["provenance"]["generations"]) == 28
    for name in original["curves"]:
        assert resumed["curves"][name]["errors"] == original["curves"][name]["errors"]


def test_corrupt_completed_block_is_recomputed_without_repeating_valid_blocks(experiment, monkeypatch):
    original = experiment.run()
    path = original["curves"]["Full-matrix LMMSE"]["provenance"]["cache_file"]
    with h5py.File(path, "r+") as h:
        h["bit_errors"][0, 0, 0] += 1
    calls = watch(monkeypatch)
    updated = experiment.run("repaired")
    assert changed(calls) == {"full_matrix_lmmse"}
    matrix = updated["curves"]["Full-matrix LMMSE"]
    assert matrix["provenance"]["computed_blocks"] == 1
    assert matrix["provenance"]["reused_blocks"] == 27
    assert matrix["errors"] == original["curves"]["Full-matrix LMMSE"]["errors"]


def test_unreadable_cache_is_a_miss_and_lock_prevents_concurrent_writers(experiment, monkeypatch):
    original = experiment.run()
    path = Path(original["curves"]["Single-tap LMMSE"]["provenance"]["cache_file"])
    path.write_bytes(b"broken HDF5")
    calls = watch(monkeypatch)
    updated = experiment.run("repaired")
    assert changed(calls) == {"single_tap_mmse"}
    assert updated["curves"]["Single-tap LMMSE"]["errors"] == original["curves"]["Single-tap LMMSE"]["errors"]
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


def test_legacy_single_tap_name_replots_without_inference_or_lost_provenance(experiment, monkeypatch):
    original = experiment.run()
    historical = dict(original, schema="watermark-unified-comparison-v1")
    historical["curves"] = {("Single-tap MMSE" if name == "Single-tap LMMSE" else name): curve
                            for name, curve in original["curves"].items()}
    historical["comparison_algorithms"] = list(historical["curves"])
    run = experiment.root/"run"
    (run/"metrics.json").write_text(json.dumps(historical), encoding="utf-8")
    (run/"report.html").write_text("Single-tap MMSE", encoding="utf-8")
    experiment.data.rename(experiment.root/"unavailable-data")
    monkeypatch.setattr(comparison, "prepare_batch", forbid)
    calls = watch(monkeypatch)
    updated = replot(run)
    assert not changed(calls)
    assert updated["curves"] == original["curves"]
    assert updated["comparison_algorithms"] == list(original["curves"])
    for name in ("report_generation", "data_identity", "parameters", "nmse"):
        assert updated[name] == original[name]
    html = (run/"report.html").read_text(encoding="utf-8")
    assert "Single-tap LMMSE" in html and "Single-tap MMSE" not in html
    historical["curves"]["Single-tap LMMSE"] = original["curves"]["Single-tap LMMSE"]
    (run/"metrics.json").write_text(json.dumps(historical), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate report curve"):
        replot(run)


def test_plot_only_removes_retired_curves_from_historical_reports(experiment, monkeypatch):
    original = experiment.run()
    historical = dict(original)
    historical["curves"] = dict(original["curves"], EP=original["curves"]["Full-matrix LMMSE"],
                                **{"Robust EP": original["curves"]["Full-matrix LMMSE"]})
    historical["parameters"] = dict(original["parameters"], ep={"iterations": 10}, robust_ep={"iterations": 10})
    historical.update(robust_covariance={"provenance": "unavailable"}, ep="historical detector",
                      paper_doi="historical EP paper", rejected_sites=[0, 0])
    (experiment.root/"run/metrics.json").write_text(json.dumps(historical), encoding="utf-8")
    experiment.data.rename(experiment.root/"unavailable-data")
    monkeypatch.setattr(comparison, "prepare_batch", forbid)
    updated = replot(experiment.root/"run")
    assert set(updated["curves"]) == {"Single-tap LMMSE", "Full-matrix LMMSE"}
    assert updated["parameters"] == original["parameters"]
    assert updated["report_generation"] == original["report_generation"]
    assert updated["nmse"] == original["nmse"]
    assert not {"robust_covariance", "ep", "paper_doi", "rejected_sites"}.intersection(updated)
    for name in updated["curves"]:
        assert updated["curves"][name] == original["curves"][name]


def test_nmse_matches_direct_full_csi_linear_average(experiment):
    report = experiment.run()
    with CompactDataset(experiment.data/"test.h5") as dataset:
        expected = [10*np.log10(np.mean([np.sum(abs(dataset.model_input(si, fi)["csi"]-dataset.reference_csi(fi))**2, axis=(1, 2))
                                       /np.sum(abs(dataset.reference_csi(fi))**2, axis=(1, 2)) for fi in range(dataset.frame_count)]))
                    for si in range(len(dataset.snr_db))]
    np.testing.assert_allclose(report["nmse"]["mean_nmse_db"], expected, atol=2e-12)


def test_matching_acceptance_statistics_are_reused(experiment, monkeypatch):
    files = fingerprints.DataIdentity({name: experiment.data/name for name in ("test.h5", "model.h5")}).files
    with CompactDataset(experiment.data/"test.h5") as dataset:
        values = [nmse.mean_nmse_db(dataset, si) for si in range(len(dataset.snr_db))]
        manifest = dict(complete=True, input_view=dataset.input_view, protocol=dataset._h.attrs["schema"], channel=dataset.channel,
                        config_hash=dataset._h.attrs["config_hash"], snr_db=dataset.snr_db.tolist(), files=files,
                        splits={"test": dict(mean_nmse_db=values, checked_frames=dataset.frame_count)})
    (experiment.data/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(nmse, "mean_nmse_db", forbid)
    report = experiment.run()
    assert report["nmse"]["mean_nmse_db"] == values


def test_mmse_parameter_validation():
    assert resolve_parameters() == {"single_tap_mmse": {}, "full_matrix_lmmse": {}}
    with pytest.raises(ValueError, match="Unknown algorithms"):
        resolve_parameters({"unknown": {}})
    with pytest.raises(ValueError, match="Invalid parameters"):
        resolve_parameters({"full_matrix_lmmse": {"iterations": 1}})


@pytest.mark.parametrize("identifier", ("ep", "robust_ep"))
def test_retired_algorithms_cannot_be_selected(identifier):
    with pytest.raises(ValueError, match="Unknown algorithms"):
        resolve_parameters({identifier: {}})
    with pytest.raises(ValueError, match="Unknown force algorithms"):
        resolve_force((identifier,))


def test_cli_rejects_force_combined_with_plot_only(monkeypatch):
    command = importlib.import_module("scripts.evaluate.compare")
    monkeypatch.setattr("sys.argv", ["compare", "--plot-only", "--force", "full_matrix_lmmse"])
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
        experiment.run(force=["full_matrix_lmmse"])
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
