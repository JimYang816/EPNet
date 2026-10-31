"""Exercise the complete experiment orchestration without training large models."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.common.numerics import DenseBackend
from src.evaluation import icr_selection


@pytest.mark.parametrize("channel", ("NOF1", "NCS1"))
@pytest.mark.parametrize("seed_mode", ("default", "multiple", "omitted"))
def test_default_pipeline_selects_before_test_and_shares_seed_warmup(tmp_path, monkeypatch, channel, seed_mode):
    from src.evaluation import icr_mmse as scoring
    from src.training import icr_mmse as training
    from src.training import udnet
    config = json.loads((Path(__file__).resolve().parents[2]/"configs/training/icr_mmse.json").read_text(encoding="utf-8"))
    assert config["selection"]["seeds"] == [20261028]
    if seed_mode == "multiple":
        config["selection"]["seeds"] = [20261028, 20261029, 20261030]
    elif seed_mode == "omitted":
        config["selection"].pop("seeds")
        config["training"]["seed"] = 12345
    expected_seeds = config["selection"].get("seeds", [config["training"]["seed"]])
    events = []

    def coverage(directory, run, **kwargs):
        events.append(dict(kind="coverage", kwargs=kwargs))
        return {}

    def train(directory, run, backend, **kwargs):
        run = Path(run)
        events.append(dict(kind="train", run=run, kwargs=deepcopy(kwargs)))
        run.mkdir(parents=True, exist_ok=True)
        (run/"warmup.pt").touch()
        checkpoint = run/"icr_mmse.pt"
        checkpoint.touch()
        return checkpoint

    def train_udnet(directory, run, backend, **kwargs):
        run = Path(run)
        events.append(dict(kind="udnet", run=run, kwargs=deepcopy(kwargs)))
        run.mkdir(parents=True, exist_ok=True)
        checkpoint = run/"best.pt"
        checkpoint.touch()
        return checkpoint

    def evaluate(dataset, checkpoint, run, backend, **kwargs):
        events.append(dict(kind="evaluate", split=Path(dataset).stem, checkpoint=Path(checkpoint),
                           run=Path(run), kwargs=deepcopy(kwargs)))
        return dict(selection=dict(ber=.1, csi_improved=True, initial_nmse=1., corrected_nmse=.5),
                    curves={"ICR-MMSE-Net":dict(ber=[.1]*7)}, performance=dict(seconds_per_block=1.))

    def match_budget(directory, checkpoint, backend, **kwargs):
        events.append(dict(kind="budget", checkpoint=Path(checkpoint), kwargs=kwargs))
        return dict(equalizer_repeats=2, matched=True, ratio=1.)

    monkeypatch.setattr(scoring, "coverage", coverage)
    monkeypatch.setattr(scoring, "evaluate", evaluate)
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(udnet, "train", train_udnet)
    monkeypatch.setattr(icr_selection, "match_once_budget", match_budget)
    run = tmp_path/"experiment"
    # --resume can first reach a new UDNet directory after interrupted selection.
    repeats = icr_selection.run_channel(config["datasets"][channel], run, DenseBackend(), config, resume=True)
    assert events[0]["kind"] == "coverage"
    assert events[0]["kwargs"]["precision"] == config["model"].get("precision", "complex64")
    assert events[0]["kwargs"]["bandwidths"] == (16,)
    assert all(event["kwargs"]["model_config"]["bandwidth"] == 16
               for event in events if event["kind"] == "train")
    searches = [event for event in events if event["kind"] == "train" and "selection" in event["run"].parts]
    assert len(searches) == 7
    assert all(event["kwargs"]["model_config"]["precision"] == config["model"].get("precision", "complex64")
               for event in searches)
    assert [event["run"].name for event in searches[:3]] == ["lambda_h_0p01", "lambda_h_0p1", "lambda_h_1p0"]
    val_events = [event for event in events if event["kind"] == "evaluate" and event["split"] == "validation"]
    assert len(val_events) == 7 and all(event["kwargs"]["require_test"] is False for event in val_events)
    udnet_event = next(event for event in events if event["kind"] == "udnet")
    assert udnet_event["kwargs"]["input_view"] == "compensated"
    assert udnet_event["kwargs"]["resume"] is False
    first_test = next(i for i, event in enumerate(events) if event["kind"] == "evaluate" and event["split"] == "test")
    assert all(events.index(event) < first_test for event in val_events)
    assert events.index(udnet_event) < first_test
    assert [repeat["seed"] for repeat in repeats] == expected_seeds
    main_runs = [event["run"].name for event in events if event["kind"] == "train"
                 and event["run"].parent == run]
    assert main_runs == ["seed_" + str(seed) for seed in expected_seeds]
    for seed in expected_seeds:
        seed_run = run/("seed_"+str(seed))
        ablations = [event for event in events if event["kind"] == "train"
                     and event["run"].parent == seed_run/"ablations"]
        assert {event["run"].name for event in ablations} == set(config["ablations"])
        assert all(event["kwargs"]["warmup_checkpoint"] == seed_run/"warmup.pt" for event in ablations)
        assert all(event["kwargs"]["seed"] == seed for event in ablations)
        by_name = {event["run"].name:event["kwargs"]["model_config"] for event in ablations}
        assert by_name["once"]["correction_mode"] == "once"
        assert by_name["once_budget_matched"]["correction_passes"] == 1
        assert by_name["once_budget_matched"]["equalizer_repeats"] == 2
        assert by_name["no_uncertainty"]["use_uncertainty"] is False
        assert by_name["no_gate"]["use_gate"] is False
        main = next(event for event in events if event["kind"] == "evaluate" and event["run"] == seed_run/"evaluation")
        assert main["kwargs"]["include_baselines"] is True
        assert main["kwargs"]["udnet_checkpoint"] == run/"udnet_compensated"/"best.pt"
        assert any(event["kind"] == "evaluate" and event["checkpoint"] == seed_run/"warmup.pt"
                   for event in events)
    saved = json.loads((run/"seed_repeats.json").read_text(encoding="utf-8"))
    assert saved["complete"] and len(saved["repeats"]) == len(expected_seeds)
    assert saved["seeds"] == expected_seeds
    assert saved["selected"]["test_used"] is False


def test_pipeline_rejects_obsolete_bandwidth_search(tmp_path):
    config = dict(model=dict(bandwidth=16), training={}, selection=dict(bandwidth=[8, 16, 32]))
    with pytest.raises(ValueError, match="remove selection.bandwidth"):
        icr_selection.run_channel(tmp_path/"data", tmp_path/"run", DenseBackend(), config)


@pytest.mark.parametrize("seeds", ([], [1, 1], ["bad"], [True], 1))
def test_pipeline_rejects_invalid_seeds_before_data_access(tmp_path, seeds):
    config = dict(model=dict(bandwidth=16), training={}, selection=dict(seeds=seeds))
    with pytest.raises(ValueError, match="distinct integer random seeds"):
        icr_selection.run_channel(tmp_path/"data", tmp_path/"run", DenseBackend(), config)


def test_udnet_resume_uses_recovery_checkpoint_only_when_present(tmp_path, monkeypatch):
    from src.training import udnet
    observed = []
    def train(directory, run, backend, **kwargs):
        observed.append(kwargs)
        return Path(run)/"best.pt"
    monkeypatch.setattr(udnet, "train", train)
    icr_selection._train_dataset_udnet(tmp_path/"data", tmp_path, DenseBackend(), {}, resume=True)
    assert observed[-1]["resume"] is False
    (tmp_path/"latest.pt").touch()
    icr_selection._train_dataset_udnet(tmp_path/"data", tmp_path, DenseBackend(), {}, resume=True)
    assert observed[-1]["resume"] is True
