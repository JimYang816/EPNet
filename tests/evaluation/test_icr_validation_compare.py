import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset, data_system
from src.evaluation.fingerprints import DataIdentity
from src.evaluation import icr_validation_compare as comparison


@pytest.fixture
def candidate(tmp_path):
    with threadpool_limits(1):
        config, prior, chain, archive = fixture()
        data, generation = tmp_path / "data", tmp_path / "generation"
        generation.mkdir()
        generate(data, generation, config, archive, chain, prior, DenseBackend(),
                 dict(commit="test", archive_sha256="test"))
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"frozen candidate for comparison test")
    identity = DataIdentity({"validation.h5": data / "validation.h5", "model.h5": data / "model.h5"})
    with CompactDataset(data / "validation.h5") as dataset:
        count, snr = dataset.frame_count, dataset.snr_db.tolist()
        bits = 4 * count * len(dataset._indices)
        channel = dataset.channel
    report = dict(complete=True, split="validation", input_view="compensated", channel=channel, smoke=True,
        checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), snr_db=snr,
        data_identity=identity.files, model_config=dict(precision="complex64", noise_mode="diagonal"),
        curves={"ICR-MMSE-Net":dict(errors=[0]*len(snr), total_bits=[bits]*len(snr), ber=[0.]*len(snr))},
        per_snr=[dict(snr_db=value, blocks=2*count, total_bits=bits, bit_errors=[0], ber=[0.],
                      mean_nmse_linear=[1., .5], mean_nmse_db=[0., 10*np.log10(.5)]) for value in snr])
    path = tmp_path / "icr_metrics.json"
    write_json(path, report)
    return data, path, report


def test_validation_comparison_uses_full_raw_lmmse_and_all_frames(candidate, tmp_path, monkeypatch):
    data, path, report = candidate
    opened = []
    def reader(path, **kwargs):
        opened.append(Path(path).name)
        return CompactDataset(path, **kwargs)
    monkeypatch.setattr(comparison, "CompactDataset", reader)
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical compensation must not enter raw comparison")
    monkeypatch.setattr(CompactDataset, "raw_model_input", forbidden)
    with threadpool_limits(1):
        actual = comparison.evaluate(data / "validation.h5", path, tmp_path / "comparison", batch_size=3)
        expected_errors = []
        with CompactDataset(data / "validation.h5", input_view="compensated") as dataset:
            for si in range(len(dataset.snr_db)):
                errors = 0
                for frame in range(dataset.frame_count):
                    inputs = dataset.model_input(si, frame)
                    residual, channels = data_system(inputs)
                    noise = inputs["noise"]["rf_variance"] * inputs["noise"]["unit_rx_covariance"]
                    for block in range(2):
                        h = channels[block]
                        weighted_h = np.linalg.solve(noise, h)
                        weighted_y = np.linalg.solve(noise, residual[block])
                        symbols = np.linalg.solve(h.conj().T @ weighted_h + np.eye(h.shape[-1]),
                                                  h.conj().T @ weighted_y)
                        truth = dataset.labels(frame)["symbols"][block]
                        errors += np.count_nonzero((symbols.real < 0) != (truth.real < 0))
                        errors += np.count_nonzero((symbols.imag < 0) != (truth.imag < 0))
                expected_errors.append(errors)
    assert opened == ["validation.h5"]
    assert actual["curves"]["Full-matrix LMMSE"]["errors"] == expected_errors
    assert actual["curves"]["ICR-MMSE-Net"] == report["curves"]["ICR-MMSE-Net"]
    assert actual["curves"]["Full-matrix LMMSE"]["total_bits"] == report["curves"]["ICR-MMSE-Net"]["total_bits"]
    assert actual["icr_report_reused"] and not actual["test_access"] and not actual["fitting"]
    assert actual["complete"] and actual["split"] == "validation"
    assert len(actual["lmmse_block_errors"]) == len(report["snr_db"])
    for name in ("metrics.json", "ber_snr.png", "ber_snr.svg", "nmse_snr.png"):
        assert (tmp_path / "comparison" / name).is_file()


@pytest.mark.parametrize("change", ("view", "split", "data", "counts", "checkpoint"))
def test_comparison_rejects_mismatched_candidate_before_prediction(candidate, tmp_path, monkeypatch, change):
    data, path, report = candidate
    if change == "view":
        report["input_view"] = "raw"
    elif change == "split":
        report["split"] = "test"
    elif change == "data":
        report["data_identity"]["validation.h5"]["sha256"] = "wrong"
    elif change == "counts":
        report["curves"]["ICR-MMSE-Net"]["total_bits"][0] -= 1
    else:
        report["checkpoint_sha256"] = "wrong"
    write_json(path, report)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid comparison reached prediction")
    monkeypatch.setattr(comparison, "equalize_batch", forbidden)
    with pytest.raises(ValueError):
        comparison.evaluate(data / "validation.h5", path, tmp_path / "invalid")
