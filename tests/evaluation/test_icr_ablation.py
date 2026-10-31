"""Check formal coverage, paired comparisons, and actual linear-feedback fitting."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from scripts.run.icr_ablation import linear_options
from src.algorithms.icr_mmse.checkpoint import load_model
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.numerics import DenseBackend
from src.evaluation.icr_ablation import GROUPS, SNR, save_report, summarize
from src.training.icr_mmse import train
from src.training.icr_workflow import resolve_config


def reports():
    result = {}
    for index, group in enumerate(GROUPS):
        layers = 1 if group == "A" else 8
        config = ICRMMSEnet(layers=8, detector_mode="lmmse" if group in ("A", "C") else "budgeted",
                           correction_mode="fixed" if group in ("A", "B") else "iterative").config
        errors = 12-index
        nmse = [.1]+([.1]*layers if group in ("A", "B") else [.05]*layers)
        rows = [dict(snr_db=snr, blocks=1200, total_bits=1228800, records=2,
            bit_errors=[2*errors]*layers, mean_nmse_linear=nmse,
            by_record=[dict(record=r, bits=614400, blocks=600, bit_errors=[errors]*layers)
                       for r in (1, 2)]) for snr in SNR]
        result[group] = dict(complete=True, smoke=False, channel="NOF1", split="test",
            input_view="compensated", snr_db=SNR, effective_model_config=config, per_snr=rows,
            checkpoint=f"{group}.pt", checkpoint_sha256=group, data_identity={"test.h5": "same"},
            performance={"seconds_per_block": .01})
    return result


def test_formal_report_keeps_true_counts_paired_records_and_both_figures(tmp_path):
    report = summarize(reports())
    assert report["curves"]["A"]["mean_ber"] == 24/1228800
    interval = report["paired_ber_difference"]["D_minus_B"]["overall"]
    assert interval["records"] == 2
    assert interval["estimate"] == -4/1228800
    assert report["curves"]["A"]["mean_nmse_db"] == pytest.approx([-10.]*7)
    save_report(report, tmp_path)
    for name in ("comparison.json", "comparison.csv", "ber_snr.png", "nmse_snr.png"):
        assert (tmp_path/name).stat().st_size > 0


@pytest.mark.parametrize("mutation", ("missing", "partial", "raw", "depth", "records", "data", "fixed_csi"))
def test_formal_summary_rejects_invalid_scientific_comparisons(mutation):
    items = reports()
    if mutation == "missing":
        items.pop("C")
    elif mutation == "partial":
        items["D"]["per_snr"][0]["total_bits"] -= 1024
    elif mutation == "raw":
        items["D"]["input_view"] = "raw"
    elif mutation == "depth":
        items["D"]["effective_model_config"]["layers"] = 22
    elif mutation == "records":
        items["C"]["per_snr"][0]["by_record"][0]["record"] = 99
    elif mutation == "data":
        items["C"]["data_identity"]["test.h5"] = "different"
    else:
        items["B"]["per_snr"][0]["mean_nmse_linear"][-1] = .05
    with pytest.raises(ValueError):
        summarize(items)


def test_linear_ablation_uses_current_correction_budget_without_fake_detector_training():
    resolved = resolve_config(channel="NOF1")
    before = deepcopy(resolved)
    a, c = (linear_options(resolved, group) for group in ("A", "C"))
    assert resolved == before
    assert a["stage_epochs"] == [0, 0, 0] and c["stage_epochs"] == [0, 10, 10]
    assert "warmup_checkpoint" not in c
    assert c["model_config"]["layers"] == 8
    assert c["model_config"]["detector_mode"] == "lmmse"
    assert c["model_config"]["correction_mode"] == "iterative"
    assert c["stage_c_correction_learning_rate"] == before["training"]["stage_c_correction_learning_rate"]


def test_cpu_linear_training_resumes_and_never_opens_test(tmp_path, monkeypatch):
    import h5py
    from data_generation.tests.test_compact import fixture
    from data_generation.src.dataset.compact_export import generate
    from src.evaluation.icr_mmse import evaluate
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(1):
            config, prior, chain, archive = fixture()
            config["snr_db"] = [0, 10]
            directory, generation = tmp_path/"data", tmp_path/"generation"
            generation.mkdir()
            generate(directory, generation, config, archive, chain, prior, DenseBackend(), {"commit": "synthetic"})
            resolved = dict(model=dict(layers=2, bandwidth=2, detector_mode="budgeted", window_size=6,
                                       center_size=3, hidden_size=4),
                            training=dict(stage_epochs=[1, 1, 1], batch_size=1, frame_limit=1,
                                          learning_rate=.001, lambda_h=.1, seed=31,
                                          loss_options=dict(objective="bit_ce", final_weight=.9)))
            original_file = h5py.File
            opened = []
            def no_test(path, *args, **kwargs):
                from pathlib import Path
                opened.append(Path(path).name)
                assert Path(path).name != "test.h5"
                return original_file(path, *args, **kwargs)
            with monkeypatch.context() as guarded:
                guarded.setattr(h5py, "File", no_test)
                a = train(directory, tmp_path/"A", DenseBackend(), **linear_options(resolved, "A"))
                options = linear_options(resolved, "C")
                train(directory, tmp_path/"C", DenseBackend(), stop_after_epochs=1, **options)
                c = train(directory, tmp_path/"C", DenseBackend(), resume=True, **options)
            assert set(opened) == {"train.h5", "validation.h5", "model.h5"}
            model, payload = load_model(c)
            _, baseline = load_model(a)
            assert payload["complete"] and len(payload["history"]) == 2
            assert not any(name.startswith("equalizers.") for name in payload["model"])
            assert any(not torch.equal(value, baseline["model"][name])
                       for name, value in payload["model"].items() if name.startswith("corrector."))
            for group, checkpoint, expected_layers in (("A", a, 1), ("C", c, 2)):
                report = evaluate(directory/"test.h5", checkpoint, tmp_path/f"eval_{group}", DenseBackend(),
                                  frame_limit=1, batch_size=1, bootstrap_samples=8)
                assert len(report["per_snr"][0]["bit_errors"]) == expected_layers
                assert np.isfinite(report["selection"]["ber"])
    finally:
        torch.set_num_threads(original_threads)
