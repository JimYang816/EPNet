"""Frozen-prior source identity and validation isolation for compact-v2 generation."""
from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT, digest
from data_generation.src.pdd import load_pdd_config, CACHE_SCHEMA, ESTIMATION_MODE
from data_generation.src.dataset.pdd_calibration import save_pdd_prior, load_pdd_prior, collect
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel, build_model
from data_generation.src.receiver.estimation import FrozenPrior
from src.common.numerics import DenseBackend


def test_channel_configurations_share_the_exact_fixed_time_basis():
    cs = [load_pdd_config(ROOT/"configs/data"/(ch+"_pddp.json")) for ch in ("ncs1", "nof1")]
    assert {k: v for k, v in cs[0].items() if k != "channel"} == {k: v for k, v in cs[1].items() if k != "channel"}
    models = [build_model(c) for c in cs]
    assert all(m.count == 34 and m.length == 736 for m in models)
    assert cs[0]["delay_block_size"] == 32 and cs[0]["csi_model_error_ratio"] == .03
    np.testing.assert_array_equal(models[0].basis, models[1].basis)
    with pytest.raises(ValueError, match="unexpected basis count"):
        build_model(dict(cs[0], dpss_expected_count=33))


def toy_contract():
    c = load_pdd_config()
    c.update(nfft=14, cp=6, fft_advance=1, guard_samples=4, null_per_edge=0, data_count=8, pilot_count=2,
             prior_records=[51], validation_records=[56], starts_seconds=[2], snr_db=[0, 30], smoke=True,
             delay_length=4, delay_block_size=2, dpss_half_bandwidth_hz=1,
             dpss_tail_ratio=.001, dpss_expected_count=2)
    rng = np.random.default_rng(31)
    basis, _ = np.linalg.qr(rng.normal(size=(80, 2)))
    model = DPSSModel(basis, 1, .001, 4, c)
    coeff = rng.normal(size=(6, 2, 4))+1j*rng.normal(size=(6, 2, 4))
    prior = FrozenPrior.fit_coefficients(model, coeff, 0, 1, [51], delay_block_size=2)
    chain = SimpleNamespace(noise_covariance=np.eye(14)*.7, noise_gain=.7)

    def frame(sounding, start, tx):
        taps = np.stack([np.pad(model.reconstruct(coeff[0], s), ((0, 0), (0, 10))) for s in range(4)])
        matrices = np.stack([taps_to_matrix(g) for g in taps])
        y = np.einsum("sij,sj->si", matrices, tx.x)
        return np.zeros(88), y, taps, matrices[1:3], dict(prediction_errors=[0, 0, 0, 0])

    chain.frame = frame
    return c, prior, chain


def test_frozen_prior_roundtrip(tmp_path):
    c, prior, _ = toy_contract()
    path = tmp_path/"prior.h5"
    save_pdd_prior(path, prior, c, {}, dict(archive_sha256="fixture"))
    restored = load_pdd_prior(path, c, dict(archive_sha256="fixture"))
    assert restored.delay_block_size == 2 and restored.feedback_iterations == 0
    np.testing.assert_array_equal(restored.model.basis, prior.model.basis)
    for name in ("coefficients", "temporal_covariance", "delay_covariance"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(prior, name))


def test_fixed_validation_truth_cannot_change_fitted_prior(tmp_path, monkeypatch):
    import data_generation.src.dataset.pdd_calibration as calibration
    from data_generation.src.transmitter.ofdm import make_frame
    c, prior, chain = toy_contract()
    cache = tmp_path/"cache.h5"
    coefficients = prior.coefficients[:1]
    full = np.einsum("nq,mql->mnl", prior.model.basis, coefficients)
    taps = np.stack([prior.model.reconstruct(coefficients, s) for s in range(4)], axis=1)
    _, clean, _, _, _ = chain.frame(56, 2, make_frame(c, 56, 2, split=1))
    with h5py.File(cache, "w") as h:
        h.create_dataset("prior/full_frame_taps", data=full)
        h.create_dataset("prior/support_residuals", data=np.zeros((1, 4)))
        h.create_dataset("validation/taps", data=taps)
        h.create_dataset("validation/total_energy", data=np.sum(abs(taps)**2, axis=(2, 3)))
        h.create_dataset("validation/support_residuals", data=np.zeros((1, 4)))
        h.create_dataset("validation/rf_base_variance", data=[.2])
        h.create_dataset("validation/clean_y", data=clean[None])
        h.create_dataset("validation/standard_noise", data=np.zeros((1, 4, 14), complex))
    monkeypatch.setattr(calibration, "collect", lambda *args: cache)
    monkeypatch.setattr(calibration, "build_model", lambda config: prior.model)
    runs = [tmp_path/"first_validation", tmp_path/"second_validation"]
    fitted, scores = [], []
    with threadpool_limits(2):
        for run in runs:
            run.mkdir()
            fitted.append(calibration.calibrate(cache, run, c, None, chain, DenseBackend(),
                                                 dict(archive_sha256="fixture")))
            scores.append(json.loads((run/"validation.json").read_text(encoding="utf-8"))["mean_linear_nmse"])
            with h5py.File(cache, "r+") as h:
                h["validation/taps"][:] *= -1
    np.testing.assert_array_equal(fitted[0].coefficients, fitted[1].coefficients)
    np.testing.assert_array_equal(fitted[0].temporal_covariance, fitted[1].temporal_covariance)
    assert scores[0] != pytest.approx(scores[1])


@pytest.mark.parametrize("attribute,value", [
    ("estimation_mode", "unrecognized"), ("archive_sha256", "different-source"),
    ("config_hash", "different-config"), ("complete", False),
    ("fit_domain", "fft_windows"), ("source_records_json", "[1]"),
])
def test_frozen_prior_rejects_mismatched_source_or_contract(tmp_path, attribute, value):
    c, prior, _ = toy_contract()
    manifest = dict(archive_sha256="fixture")
    path = tmp_path/"prior.h5"
    save_pdd_prior(path, prior, c, {}, manifest)
    with h5py.File(path, "r+") as h:
        h.attrs[attribute] = value
    with pytest.raises(ValueError):
        load_pdd_prior(path, c, manifest)


@pytest.mark.parametrize("attribute,value", [("schema", "ncs1-pdd-calibration-v1"),
                                            ("estimation_mode", "frozen_prior_prefix_no_feedback_v1")])
def test_joint_calibration_rejects_historical_cache(tmp_path, attribute, value):
    c = load_pdd_config()
    path = tmp_path/"cache.h5"
    with h5py.File(path, "w") as h:
        h.attrs.update(schema=CACHE_SCHEMA, estimation_mode=ESTIMATION_MODE,
                       config_hash=digest(c), archive_sha256="fixture")
        h.attrs[attribute] = value
    with pytest.raises(ValueError, match="historical caches"):
        collect(path, c, None, None, dict(archive_sha256="fixture"))
