"""Independent direct solves, oracle isolation and frozen-candidate identity."""
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.icr_mmse.checkpoint import (SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE,
                                               atomic_save, cpu_state, load_model)
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches
from src.evaluation.fingerprints import DataIdentity, source_fingerprint
from src.evaluation.inputs import prepare_batch
from src.evaluation import icr_csi_lmmse as comparison


@pytest.fixture
def candidate(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        config, prior, chain, archive = fixture()
        config["snr_db"] = [0, 15, 30]
        data, generation = tmp_path/"data", tmp_path/"generation"
        generation.mkdir()
        generate(data, generation, config, archive, chain, prior, DenseBackend(),
                 dict(commit="test", archive_sha256="test"))
        torch.manual_seed(17)
        model = ICRMMSEnet(layers=2, bandwidth=2, window_size=6, center_size=3,
                          hidden_size=4, noise_mode="diagonal")
        with torch.no_grad():
            model.corrector.head.bias[:5].fill_(.12)
            model.corrector.head.bias[5:10].fill_(-.08)
        identity = DataIdentity({"validation.h5": data/"validation.h5", "model.h5": data/"model.h5"})
        with CompactDataset(data/"validation.h5") as dataset:
            count, snr = dataset.frame_count, dataset.snr_db.tolist()
            bits = 4*count*len(dataset._indices)
            payload = dict(schema=SCHEMA, input_view="compensated", model_config=model.config,
                feedback_layout=FEEDBACK_LAYOUT, inference_code=source_fingerprint(INFERENCE_CODE),
                model=cpu_state(model), training_scope="one_channel_all_snr_D1_D2", channel=dataset.channel,
                data_schema=str(dataset._h.attrs["schema"]), model_identity=str(dataset._h.attrs["model_identity"]),
                data_indices=dataset._indices.tolist(), snr_db=snr, complete=True, stage="C",
                inference_correction_enabled=True, identity=dict(data=identity.files, controls=dict(lambda_h=.01)))
        checkpoint = tmp_path/"icr.pt"
        atomic_save(checkpoint, payload)
        report = dict(complete=True, split="validation", input_view="compensated", channel=payload["channel"],
            smoke=True, checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), snr_db=snr,
            data_identity=identity.files, model_config=model.config,
            curves={"ICR-MMSE-Net":dict(errors=[0]*len(snr), total_bits=[bits]*len(snr), ber=[0.]*len(snr))},
            per_snr=[dict(snr_db=value, blocks=2*count, total_bits=bits, bit_errors=[0]) for value in snr])
        report_path = tmp_path/"candidate.json"
        write_json(report_path, report)
        yield data, checkpoint, report_path, payload, report
    torch.set_num_threads(previous)


@pytest.mark.parametrize("precision", ("complex64", "complex128"))
def test_receiver_preparation_matches_original_candidate_inputs(candidate, precision):
    data, *_ = candidate
    with CompactDataset(data/"validation.h5") as dataset:
        for inputs, keys in input_batches(dataset, precision=precision, batch_size=3):
            si, _, block = keys[0]
            batch = prepare_batch(dataset, si, [key[1] for key in keys], block, DenseBackend())
            actual = comparison.model_inputs(dataset, batch, block, precision)
            for name in inputs:
                torch.testing.assert_close(actual[name], inputs[name], rtol=0,
                                           atol=2e-14 if precision == "complex128" else 0)


def test_three_csi_variants_match_independent_full_colored_noise_solves(candidate, tmp_path, monkeypatch):
    data, checkpoint, path, _, original = candidate
    prepare, solve, reference = comparison.prepare_batch, comparison.equalize_batch, CompactDataset.reference_csi
    state = dict(solves=0, checked=0)
    def receiver_batch(*args):
        state["solves"] = 0
        return prepare(*args)
    def checked_solve(batch, backend, **kwargs):
        assert not batch.likelihoods
        assert batch.channel.dtype == np.complex128
        assert batch.cancelled_observations.dtype == np.complex128
        state["solves"] += 1
        state["checked"] += 1
        return solve(batch, backend, **kwargs)
    def isolated_reference(self, frame):
        assert state["solves"] == 2, "Reference entered receiver-only prediction"
        return reference(self, frame)
    def forbidden(*args):
        raise AssertionError("Historical compensated CSI must not enter the comparison")
    opened, file = [], h5py.File
    def checked_file(path, *args, **kwargs):
        opened.append(Path(path).name)
        assert Path(path).name not in ("train.h5", "test.h5")
        return file(path, *args, **kwargs)
    monkeypatch.setattr(comparison, "prepare_batch", receiver_batch)
    monkeypatch.setattr(comparison, "equalize_batch", checked_solve)
    monkeypatch.setattr(CompactDataset, "reference_csi", isolated_reference)
    monkeypatch.setattr(CompactDataset, "raw_model_input", forbidden)
    monkeypatch.setattr(h5py, "File", checked_file)
    before = file_sha256(path), file_sha256(checkpoint)
    actual = comparison.evaluate(data/"validation.h5", path, tmp_path/"comparison", batch_size=3)
    assert before == (file_sha256(path), file_sha256(checkpoint))
    assert set(opened) == {"validation.h5", "model.h5"}
    assert state["checked"] > 0
    monkeypatch.setattr(CompactDataset, "reference_csi", reference)
    model, _ = load_model(checkpoint)
    with CompactDataset(data/"validation.h5") as dataset, torch.inference_mode():
        expected = np.zeros((3, len(dataset.snr_db)), dtype=int)
        nmse = np.zeros((3, len(dataset.snr_db)))
        for inputs, keys in input_batches(dataset, batch_size=1):
            si, frame, block = keys[0]
            output = model(**inputs)
            raw = dataset.model_input(si, frame)
            channels = [raw["csi"][block], output.channels[-1].dense().numpy()[0].astype(complex),
                        dataset.reference_csi(frame)[block]]
            truth = dataset.labels(frame)["symbols"][block]
            covariance = raw["noise"]["rf_variance"]*raw["noise"]["unit_rx_covariance"]
            for vi, channel in enumerate(channels):
                h = channel[:, raw["data_indices"]]
                residual = raw["observations"][block+1]-channel@raw["known"][block+1]
                symbols = np.linalg.solve(h.conj().T@np.linalg.solve(covariance, h)+np.eye(h.shape[-1]),
                                           h.conj().T@np.linalg.solve(covariance, residual))
                expected[vi, si] += np.count_nonzero((symbols.real < 0) != (truth.real < 0))
                expected[vi, si] += np.count_nonzero((symbols.imag < 0) != (truth.imag < 0))
                nmse[vi, si] += np.sum(abs(channel-channels[-1])**2)/np.sum(abs(channels[-1])**2)
        for vi, name in enumerate(comparison.VARIANTS):
            curve = actual["curves"][name]
            assert curve["errors"] == expected[vi].tolist()
            assert curve["total_bits"] == original["curves"]["ICR-MMSE-Net"]["total_bits"]
            np.testing.assert_allclose(curve["mean_nmse_linear"], nmse[vi]/(2*dataset.frame_count), rtol=1e-12)
    assert actual["curves"]["true"]["mean_nmse_linear"] == [0.]*len(original["snr_db"])
    assert actual["curves"]["true"]["mean_nmse_db"] == [None]*len(original["snr_db"])
    assert actual["complete"] and not actual["test_access"] and not actual["fitting"] and not actual["gpu_used"]
    for name in ("metrics.json", "metrics.csv", "ber_snr.png", "ber_snr.svg", "nmse_snr.png", "nmse_snr.svg"):
        assert (tmp_path/"comparison"/name).is_file()


@pytest.mark.parametrize("change", ("lambda", "incomplete", "data"))
def test_rejects_wrong_candidate_before_prediction(candidate, tmp_path, monkeypatch, change):
    data, checkpoint, path, payload, report = candidate
    if change == "lambda":
        payload["identity"]["controls"]["lambda_h"] = .1
    elif change == "incomplete":
        payload["complete"] = False
    else:
        payload["identity"]["data"]["validation.h5"] = dict(sha256="wrong", bytes=1)
    atomic_save(checkpoint, payload)
    report["checkpoint_sha256"] = file_sha256(checkpoint)
    write_json(path, report)
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid candidate reached prediction")
    monkeypatch.setattr(comparison, "prepare_batch", forbidden)
    with pytest.raises(ValueError):
        comparison.evaluate(data/"validation.h5", path, tmp_path/"invalid")
