from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.src.config import ROOT, digest
from data_generation.src.pdd import load_pdd_config, DATA_SCHEMA, CACHE_SCHEMA, ESTIMATION_MODE
from data_generation.src.dataset.pdd_calibration import save_pdd_prior, load_pdd_prior, collect
from data_generation.src.dataset.pdd_export import generate
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.receiver.dpss import DPSSModel, build_model
from data_generation.src.receiver.estimation import FrozenPrior
from src.common.numerics import DenseBackend
from src.data.ncs1 import NCS1Dataset
from src.evaluation.pdd import CURVES, summarize


def test_two_algorithm_bit_counts_and_linear_nmse_aggregation():
    errors = np.ones((2, 7, 25, 2), dtype=int)
    nmse = np.tile(np.array([.01, 1.]), (7, 25, 1))
    result = summarize(errors, nmse, 512)
    assert list(result["curves"]) == ["MMSE", "EP"]
    assert result["frame_count"] == 25 and result["data_symbol_count"] == 50
    assert all(x["total_bits"] == [51200]*7 for x in result["curves"].values())
    np.testing.assert_allclose(result["mean_nmse_linear"], .505)
    np.testing.assert_allclose(result["mean_nmse_db"], 10*np.log10(.505))
    np.testing.assert_allclose(result["nmse_d1_d2_linear"], [[.01, 1]]*7)


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
             prior_records=[51], validation_records=[56], test_records=[1], starts_seconds=[2],
             odd_test_starts_seconds=[2, 14], even_test_starts_seconds=[8], snr_db=[0, 30], smoke=True,
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

    def noise(clean, record, start):
        rng = np.random.default_rng([record, int(start)])
        standard = rng.normal(size=(4, 14))+1j*rng.normal(size=(4, 14))
        return standard, np.array([.2, .0002]), .14

    chain.frame, chain.noise = frame, noise
    return c, prior, chain


def test_generation_and_two_algorithm_checkpoint_resume(tmp_path, monkeypatch):
    import src.evaluation.pdd as evaluation
    c, prior, chain = toy_contract()
    archive = SimpleNamespace(load=lambda record: record)
    provenance = dict(commit="fixture", config_hash=digest(c))
    path = tmp_path/"resumed.h5"
    run = tmp_path/"run"
    run.mkdir()
    original = chain.frame
    counter = [0]

    def interrupted(*args):
        counter[0] += 1
        if counter[0] == 2:
            raise RuntimeError("simulated interruption")
        return original(*args)

    chain.frame = interrupted
    with threadpool_limits(2):
        with pytest.raises(RuntimeError, match="interruption"):
            generate(path, run, c, archive, chain, prior, DenseBackend(), provenance)
        with h5py.File(path, "r") as h:
            np.testing.assert_array_equal(h["sample_complete"][:], [True, False])
        chain.frame = original
        generate(path, run, c, archive, chain, prior, DenseBackend(), provenance)
        fresh = tmp_path/"fresh.h5"
        generate(fresh, run, c, archive, chain, prior, DenseBackend(), provenance)
        with h5py.File(path, "r") as a, h5py.File(fresh, "r") as b:
            for key in ("observations", "reference_csi", "estimated_csi", "data_bits", "standard_noise_sha256", "actual_snr_db", "nmse_linear"):
                np.testing.assert_array_equal(a[key][:], b[key][:])
        monkeypatch.setattr(evaluation, "plot_results", lambda *args: None)
        (run/"validation.json").write_text(json.dumps(dict(complete=True)), encoding="utf-8")
        first = evaluation.evaluate(path, run, DenseBackend())
        assert list(first["curves"]) == ["MMSE", "EP"] and first["csi_source"] == "estimated"
        assert first["ep_summary"]["total"] == 2*2*2
        assert sum(first["ep_summary"]["stop_reasons"].values()) == 8
        report = (run/"report.html").read_text(encoding="utf-8")
        assert "共2816行" in report and "MMSE／EP" in report and "LMMSE" not in report
        with h5py.File(run/"equalization.h5", "r+") as h:
            assert h["bit_errors"].shape == (2, 2, 2, 2)
            assert h["ep_diagnostics"].shape == (2, 2, 2)
            expected_errors = h["bit_errors"][:]
            expected_diag = h["ep_diagnostics"][:]
            h["complete"][1, 0, 1] = False
            h["bit_errors"][:, 1, 0, 1] = 9999
        second = evaluation.evaluate(path, run, DenseBackend())
        assert first["curves"] == second["curves"]
        with h5py.File(run/"equalization.h5", "r") as h:
            np.testing.assert_array_equal(h["bit_errors"][:], expected_errors)
            np.testing.assert_array_equal(h["ep_diagnostics"][:], expected_diag)
        for attribute, invalid in (("schema", "ncs1-pddp-equalization-v1"), ("estimation_mode", None),
                                   ("ep_parameters_json", "{}"), ("config_hash", "other")):
            with h5py.File(run/"equalization.h5", "r+") as h:
                original_attribute = h.attrs[attribute]
                if invalid is None:
                    del h.attrs[attribute]
                else:
                    h.attrs[attribute] = invalid
            with pytest.raises(ValueError, match="checkpoint provenance mismatch"):
                evaluation.evaluate(path, run, DenseBackend())
            with h5py.File(run/"equalization.h5", "r+") as h:
                np.testing.assert_array_equal(h["bit_errors"][:], expected_errors)
                h.attrs[attribute] = original_attribute


def test_reference_csi_does_not_enter_equalizer_ber(tmp_path, monkeypatch):
    import src.evaluation.pdd as evaluation
    c, prior, chain = toy_contract()
    dataset = tmp_path/"data.h5"
    first_run, second_run = tmp_path/"first", tmp_path/"second"
    for run in (first_run, second_run):
        run.mkdir()
    monkeypatch.setattr(evaluation, "plot_results", lambda *args: None)
    monkeypatch.setattr(evaluation, "write_report", lambda *args: None)
    with threadpool_limits(2):
        generate(dataset, first_run, c, SimpleNamespace(load=lambda r: r), chain, prior,
                 DenseBackend(), dict(commit="fixture"))
        first = evaluation.evaluate(dataset, first_run, DenseBackend())
        with h5py.File(dataset, "r+") as h:
            h["reference_csi"][:] = 0
        second = evaluation.evaluate(dataset, second_run, DenseBackend())
    assert first["curves"] == second["curves"]
    assert second["csi_source"] == "estimated"


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


@pytest.mark.parametrize("policy", ["reject", "diagnose"])
@pytest.mark.parametrize("failed_symbol", [2, 3])
def test_support_mismatch_policy_rejects_or_records_diagnostics(tmp_path, policy, failed_symbol):
    from threadpoolctl import threadpool_limits
    c, prior, chain = toy_contract()
    c["model_mismatch_policy"] = policy
    original = chain.frame

    def omitted_support(*args):
        clean, y, taps, reference, info = original(*args)
        taps[failed_symbol, :, -1] = 10  # D2 or tail P alone must trigger the frame support gate.
        return clean, y, taps, reference, info

    chain.frame = omitted_support
    path = tmp_path/"failed.h5"
    with threadpool_limits(2):
        if policy == "reject":
            with pytest.raises(RuntimeError, match="stop without reselection"):
                generate(path, tmp_path, c, SimpleNamespace(load=lambda r: r), chain, prior, DenseBackend(), dict(commit="fixture"))
        else:
            generate(path, tmp_path, c, SimpleNamespace(load=lambda r: r), chain, prior, DenseBackend(), dict(commit="fixture"))
    with h5py.File(path, "r") as h:
        if policy == "reject":
            assert not h.attrs["complete"] and not np.any(h["sample_complete"][:])
        else:
            assert h.attrs["complete"] and np.all(h["sample_complete"][:])
            for raw in h["diagnostics_json"][:]:
                diagnostic = json.loads(raw)
                assert not diagnostic["support_passed"]
                assert diagnostic["model_mismatch_policy"] == "diagnose"
                assert diagnostic["support_residuals_p1_d1_d2_p2"][failed_symbol] > 10**(c["support_error_db"]/10)


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


@pytest.mark.parametrize("channel", ["NCS1", "NOF1"])
def test_result_figures_have_two_ber_lines_and_db_heatmaps(tmp_path, monkeypatch, channel):
    from matplotlib.figure import Figure
    from src.evaluation.pdd import plot_results
    c, prior, chain = toy_contract()
    c["channel"] = channel
    path = tmp_path/"joint.h5"
    with threadpool_limits(2):
        generate(path, tmp_path, c, SimpleNamespace(load=lambda r: r), chain, prior,
                 DenseBackend(), dict(commit="fixture"))
    savefig = Figure.savefig
    captured = []

    def inspect_figure(figure, filename, *args, **kwargs):
        if filename.name == "ber_snr.png":
            assert channel in figure.axes[0].get_title()
            assert len(figure.axes[0].lines) == 2
            assert [line.get_label() for line in figure.axes[0].lines] == ["MMSE", "EP"]
        if filename.name == "nmse_snr.png":
            assert len(figure.axes) == 1 and len(figure.axes[0].lines) == 1
            captured.append(np.array(figure.axes[0].lines[0].get_ydata()))
        if filename.name.startswith("csi_snr_"):
            assert f"{channel}_001" in figure._suptitle.get_text()
            images = [image for ax in figure.axes for image in ax.images]
            assert len(images) == 6 and all(image.get_clim()[0] == -80 for image in images)
        return savefig(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    with h5py.File(path, "r") as h:
        report = dict(summarize(np.ones((2, 2, 2, 2), int), h["nmse_linear"][:], c["data_count"]), snr_db=c["snr_db"])
        plot_results(h, report, tmp_path)
        expected = 10*np.log10(h["nmse_linear"][:].mean(axis=(1, 2)))
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], expected)
    for name in ("ber_snr.png", "nmse_snr.png", "csi_snr_00.png", "csi_snr_30.png"):
        assert (tmp_path/name).stat().st_size > 1000
    scale = json.loads((tmp_path/"heatmap_scale.json").read_text(encoding="utf-8"))
    assert scale["minimum_db"] == -80 and scale["error"].startswith("complex estimated-reference")
