import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from data_generation.src.dataset.compact_math import reconstruct
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset, data_system
from src.evaluation.comparison import (evaluate, dpss_predict, shrink_second_moment,
                                       likelihood_batch, make_whitener, test_batch as prepare_test_batch)
from src.algorithms.MMSE.single_tap import mmse


def test_dpss_action_matches_all_full_matrix_couplings():
    c, prior, _, _ = fixture()
    rng = np.random.default_rng(221)
    coeff = rng.normal(size=(3, 2, 4))+1j*rng.normal(size=(3, 2, 4))
    x = rng.normal(size=(3, 2, 14))+1j*rng.normal(size=(3, 2, 14))
    basis = prior.model.sampled_basis.reshape(4, 14, 2)[[1, 2]]
    actual = dpss_predict(coeff, basis, x, DenseBackend())
    expected = np.stack([np.einsum("dij,dj->di", reconstruct(prior.model, co), xx) for co, xx in zip(coeff, x)])
    np.testing.assert_allclose(actual, expected, atol=3e-15)


def test_shrinkage_preserves_powers_and_colored_psd_covariance():
    rng = np.random.default_rng(63)
    x = rng.normal(size=(700, 5))+1j*rng.normal(size=(700, 5))
    x[:, 1] += 2j*x[:, 0]
    covariance, shrinkage = shrink_second_moment(x)
    np.testing.assert_allclose(np.diag(covariance).real, np.mean(abs(x)**2, axis=0))
    np.testing.assert_allclose(covariance, covariance.conj().T, atol=1e-15)
    assert 0 <= shrinkage < 1
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert abs(covariance[0, 1]) > .5


def test_zero_error_covariance_reproduces_ordinary_likelihood():
    rng = np.random.default_rng(82)
    h = rng.normal(size=(2, 6, 3))+1j*rng.normal(size=(2, 6, 3))
    y = rng.normal(size=(2, 6))+1j*rng.normal(size=(2, 6))
    a = rng.normal(size=(6, 6))+1j*rng.normal(size=(6, 6))
    noise = a@a.conj().T+np.eye(6)
    b = DenseBackend()
    ordinary = likelihood_batch(h, y, [.2, .3], make_whitener(noise, b), b)
    robust = likelihood_batch(h, y, [.2, .3], make_whitener(noise+np.zeros((6, 6)), b), b)
    for left, right in zip(ordinary, robust):
        np.testing.assert_array_equal(left, right)


def test_cpu_end_to_end_train_only_covariance_full_test_counts_and_resume(tmp_path):
    with threadpool_limits(2):
        c, prior, chain, archive = fixture()
        c["learning_dataset"]["repeats"]["train"] = 4
        data, generation, run = tmp_path/"data", tmp_path/"generation", tmp_path/"evaluation"
        generation.mkdir()
        generate(data, generation, c, archive, chain, prior, DenseBackend(),
                 dict(commit="test", archive_sha256="test"))
        cache = tmp_path/"cache"
        report = evaluate(data/"test.h5", run, DenseBackend(), batch_size=2, iterations=2, cache_dir=cache)
        assert report["complete"] and report["frame_count"] == 2
        assert report["bits_per_snr"] == 64
        assert len(report["curves"]) == 4
        assert report["robust_covariance"]["source"] == "train only"
        assert (run/"ber_snr.png").is_file()
        assert (run/"nmse_snr.png").is_file()
        for curve in report["curves"].values():
            with h5py.File(curve["provenance"]["cache_file"], "r") as h:
                assert np.all(h["complete"][:]) and h.attrs["complete"]
                assert h["bit_errors"].shape == (7, 2, 2)
        resumed = evaluate(data/"test.h5", run, DenseBackend(), batch_size=1, iterations=2, cache_dir=cache)
        assert all(c["provenance"]["reused"] for c in resumed["curves"].values())
        for name in report["curves"]:
            assert resumed["curves"][name]["errors"] == report["curves"][name]["errors"]
        updated = evaluate(data/"test.h5", run, DenseBackend(), iterations=3, cache_dir=cache)
        assert [c["provenance"]["reused"] for c in updated["curves"].values()] == [True, True, False, False]
        with pytest.raises(ValueError, match="test split"):
            evaluate(data/"validation.h5", tmp_path/"invalid", DenseBackend())
        with CompactDataset(data/"test.h5") as ds, h5py.File(data/"train.h5", "r") as train:
            for d in range(2):
                channel, y, variances, single = prepare_test_batch(ds, 1, [0, 1], d, DenseBackend())
                for j, fi in enumerate([0, 1]):
                    inputs = ds.model_input(1, fi)
                    residual, full = data_system(inputs)
                    np.testing.assert_allclose(channel[j], full[d], atol=2e-15)
                    np.testing.assert_allclose(y[j], residual[d], atol=2e-15)
                    expected = mmse(inputs["csi"][d], inputs["observations"][d+1], inputs["masks"]["data"],
                                    variances[j]*np.diag(inputs["noise"]["unit_rx_covariance"]).real)
                    np.testing.assert_allclose(single[j], expected, atol=2e-15)
            # Calibrate the accepted training CSI using the training split's alpha.
            si = 2
            residuals = []
            for fi in range(4):
                x = ds._known[[1, 2]].copy()
                from data_generation.src.transmitter.ofdm import qpsk
                x[:, ds._indices] = qpsk(train["samples/bits"][fi])
                reference = train["windows/reference_csi"][0].astype(complex)
                raw = reconstruct(ds._model, train["samples/raw_coefficients"][si, fi].astype(complex))
                clean = train["samples/clean_observations"][fi][[1, 2]]
                alpha = float(train["compensation/alpha"][si])
                channel = reference+alpha*(raw-reference)
                residual = clean-np.einsum("dij,dj->di", channel, x)
                scale = train["samples/rf_noise_variance"][si, fi]*10**(float(ds.snr_db[si])/10)
                residuals.append(residual/np.sqrt(scale))
            residuals = np.asarray(residuals)
            with h5py.File(report["robust_covariance"]["provenance"]["cache_file"], "r") as h:
                for d in range(2):
                    expected, _ = shrink_second_moment(residuals[:, d])
                    np.testing.assert_allclose(h["error_covariance_unit"][si, d], expected, rtol=2e-6, atol=2e-9)
