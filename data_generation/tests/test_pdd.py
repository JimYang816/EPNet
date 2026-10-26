"""Independent dense formula, row selection, joint and truth-isolation tests."""
from __future__ import annotations

import inspect
import os
import numpy as np
import pytest
from scipy import linalg
from threadpoolctl import threadpool_limits

from data_generation.src.pdd import load_pdd_config, identities
from data_generation.src.receiver.dpss import DPSSModel, build_model
from data_generation.src.receiver.estimation import FrozenPrior, input_windows
from data_generation.src.receiver.pdd import JointEstimator, selected_components, selected_time_covariance
from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.src.dataset.pdd_calibration import coefficient_nmse
from data_generation.src.transmitter.ofdm import carrier_masks, make_frame, fft, ifft
from src.common.numerics import DenseBackend


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(2):
        yield


def test_pdd_mapping_dual_d_and_test_schedule():
    c = load_pdd_config()
    m = carrier_masks(c)
    assert [m[k].sum() for k in ("active", "data", "pilot", "protection", "edge", "tracking")] == [896, 512, 128, 256, 128, 384]
    assert m["data"][512] and not m["tracking"][512]
    np.testing.assert_array_equal(m["pilot"][64:960], np.tile([0, 0, 0, 1, 0, 0, 0], 128))
    f = make_frame(c, 1, 2)
    assert f.bits.shape == (2, 512, 2)
    assert f.baseband.shape == (6232,)
    assert f.x.shape == f.known.shape == (4, 1024)
    for s in (0, 3):
        np.testing.assert_allclose(abs(f.x[s, m["active"]]), 1)
        np.testing.assert_array_equal(f.x[s, m["edge"]], 0)
        np.testing.assert_array_equal(f.known[s], f.x[s])
    assert not np.array_equal(f.x[0], f.x[3])
    np.testing.assert_allclose(abs(f.x[1:3, m["data"] | m["pilot"]]), 1)
    np.testing.assert_array_equal(f.x[1:3, m["null"]], 0)
    np.testing.assert_array_equal(f.known[1:3, m["data"]], 0)
    np.testing.assert_array_equal(make_frame(c, 9, 26).known, f.known)
    assert not np.array_equal(f.bits[0], f.bits[1])
    ids = identities(c, "test")
    assert len(ids) == 25 and len(set(ids)) == 25
    assert [s for r, s in ids if r == 1] == [2, 14, 26]
    assert [s for r, s in ids if r == 2] == [8, 20]
    assert 25*2*512*2 == 51200
    model = build_model(c)
    assert model.count == 34 and model.length == 736 and model.basis.shape[0] == 5632
    np.testing.assert_allclose(model.basis.T@model.basis, np.eye(model.count), atol=1e-12)



def small_fixture():
    rng = np.random.default_rng(62)
    n, cp, q, length = 14, 6, 3, 4
    c = dict(nfft=n, cp=cp, fft_advance=1, test_records=[1])
    basis, _ = np.linalg.qr(rng.normal(size=(4*(n+cp), q)))
    model = DPSSModel(basis, 1, .001, length, c)
    coeff = rng.normal(size=(8, q, length))+1j*rng.normal(size=(8, q, length))
    prior = FrozenPrior.fit_coefficients(model, coeff, 0, 1, [51], delay_block_size=2)
    known = rng.normal(size=(4, n))+1j*rng.normal(size=(4, n))
    data = np.zeros(n, bool)
    data[[1, 2, 5, 8, 11]] = True
    tracking = np.zeros(n, bool)
    tracking[[3, 4, 6, 7, 9, 10]] = True
    known[1:3, data] = 0
    masks = dict(data=data, tracking=tracking)
    a = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    noise = a@a.conj().T/n+np.eye(n)*.2
    y = rng.normal(size=(4, n))+1j*rng.normal(size=(4, n))
    return prior, known, masks, noise, y


def dense_contract(prior, known, masks):
    model = prior.model
    n, q, length = model.c["nfft"], model.count, model.length
    rc = np.zeros((q, length, q, length), complex)
    for select, t, d in prior.separable_blocks():
        rc[:, select, :, select] = np.einsum("qr,lk->qlrk", t, d)
    rc = rc.reshape(q*length, q*length)
    f = fft(np.eye(n)).T
    transform = linalg.block_diag(f, f, f, f)
    rows = np.r_[np.arange(n), n+np.flatnonzero(masks["tracking"]), 2*n+np.flatnonzero(masks["tracking"]), 3*n+np.arange(n)]

    def design(symbols):
        x = input_windows(symbols, length).reshape(4*n, length)
        time = np.einsum("nq,nl->nql", model.sampled_basis, x).reshape(4*n, -1)
        return (transform@time)[rows]

    a = design(known)
    ri = np.zeros((len(rows), len(rows)), complex)
    for s in (1, 2):
        for k in np.flatnonzero(masks["data"]):
            unit = np.zeros((4, n), complex)
            unit[s, k] = 1
            tj = design(unit)
            ri += tj@rc@tj.conj().T
    return a@rc@a.conj().T+ri, rc@a.conj().T, rows


def test_selected_frequency_covariance_and_whitening():
    prior, known, masks, noise, y = small_fixture()
    backend = DenseBackend()
    expected_cov, expected_cross, rows = dense_contract(prior, known, masks)
    actual_cov, actual_cross, _, _ = selected_components(prior, known, masks, backend)
    np.testing.assert_allclose(actual_cov, expected_cov, atol=2e-13)
    np.testing.assert_allclose(actual_cross, expected_cross, atol=2e-13)
    selected = np.flatnonzero(masks["tracking"])
    w_full = linalg.solve_triangular(linalg.cholesky(noise, lower=True), np.eye(len(noise)), lower=True)
    assert not np.allclose(w_full[np.ix_(selected, selected)]@noise[np.ix_(selected, selected)]@w_full[np.ix_(selected, selected)].conj().T, np.eye(len(selected)))
    kernel = ifft(np.eye(len(noise))).T @ noise @ ifft(np.eye(len(noise))).conj()
    np.testing.assert_allclose(selected_time_covariance(kernel, selected), noise[np.ix_(selected, selected)], atol=1e-12)


@pytest.mark.parametrize("model_error_ratio", [0, .03])
def test_joint_dense_formula_and_shared_reconstruction(model_error_ratio):
    prior, known, masks, noise, y = small_fixture()
    prior.model.c["csi_model_error_ratio"] = model_error_ratio
    backend = DenseBackend()
    joint = JointEstimator(prior, known, masks, noise, backend)
    observations = np.stack([y, y*.7+1j])
    variances = np.array([.3, 0.])
    coefficients = joint.estimate_coefficients(observations, known, masks, variances)
    assert coefficients.shape == (2, prior.model.count, prior.model.length)
    assert len(joint.diagnostics) == 1
    assert joint.diagnostics[0]["observations"] == int(joint.offsets[-1])
    covariance, cross, rows = dense_contract(prior, known, masks)
    selected_noise = linalg.block_diag(noise, noise, noise, noise)[np.ix_(rows, rows)]
    w = linalg.solve_triangular(linalg.cholesky(selected_noise, lower=True), np.eye(len(rows)), lower=True)
    ridge = prior.regularizer*np.trace(w@covariance@w.conj().T).real/len(rows)
    for k, variance in enumerate(variances):
        noise_gain = np.trace(noise).real/len(noise)
        p_energy = np.mean(np.sum(abs(known[[0, 3]])**2, axis=1))
        d_energy = np.mean(np.sum(abs(known[[1, 2]])**2, axis=1)) + masks["data"].sum()
        rx_power = np.mean(abs(observations[k, [0, 3]])**2)
        effective = variance+model_error_ratio*(d_energy/p_energy)*max(rx_power/noise_gain-variance, 0)
        expected = cross@np.linalg.solve(covariance+(effective+ridge)*selected_noise,
                                         observations[k].ravel()[rows])
        np.testing.assert_allclose(coefficients[k].ravel(), expected, atol=3e-12)
        np.testing.assert_allclose(joint.effective_noise_variances(observations[k], variance), [effective])
    matrices = joint.estimate(observations, known, masks, variances)
    assert matrices.shape == (2, 2, 14, 14)
    for k in range(2):
        for d, s in enumerate((1, 2)):
            expected = taps_to_matrix(prior.model.reconstruct(coefficients[k], s))
            np.testing.assert_allclose(matrices[k, d], expected, atol=3e-12)
    rng = np.random.default_rng(721)
    truth = rng.normal(size=(2, 4, 14, 14))+1j*rng.normal(size=(2, 4, 14, 14))
    energy = np.sum(abs(truth)**2, axis=(2, 3))
    scores = coefficient_nmse(prior.model, coefficients, truth, energy)
    expected_scores = [[np.sum(abs(matrices[k, d]-taps_to_matrix(truth[k, s]))**2)/energy[k, s]
                        for d, s in enumerate((1, 2))] for k in range(2)]
    np.testing.assert_allclose(scores, expected_scores, atol=1e-13)


@pytest.mark.parametrize("observation_symbol", [2, 3])
@pytest.mark.parametrize("model_error_ratio", [0, .03])
def test_joint_d2_and_tail_p_update_both_d_and_reject_unknown_labels(observation_symbol, model_error_ratio):
    prior, known, masks, noise, y = small_fixture()
    prior.model.c["csi_model_error_ratio"] = model_error_ratio
    joint = JointEstimator(prior, known, masks, noise, DenseBackend())
    first = joint.estimate(y, known, masks, [.3])
    modified = y.copy()
    modified[observation_symbol] += 100-20j
    changed = joint.estimate(modified, known, masks, [.3])
    assert all(not np.allclose(first[:, d], changed[:, d]) for d in range(2))
    modified = y.copy()
    modified[1:3, ~masks["tracking"]] += 1e8+3e7j
    np.testing.assert_array_equal(first, joint.estimate(modified, known, masks, [.3]))
    signature = str(inspect.signature(joint.estimate_coefficients))
    assert not any(s in signature for s in ("bits", "truth", "reference", "labels"))
    contaminated = known.copy()
    contaminated[2, masks["data"]] = 1
    with pytest.raises(ValueError, match="training/masks mismatch"):
        joint.estimate(y, contaminated, masks, [.3])
    with pytest.raises(ValueError, match="labels"):
        JointEstimator(prior, contaminated, masks, noise, DenseBackend())


def test_pddp_config_rejects_old_frame_contract(tmp_path):
    import json
    c = load_pdd_config()
    c["frame_structure"] = "PDD"
    path = tmp_path/"old.json"
    path.write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(ValueError, match="PDDP"):
        load_pdd_config(path)


@pytest.mark.skipif(os.environ.get("EPNET_TEST_CUDA") != "1", reason="Enable explicitly on the GPU server")
def test_pdd_cuda_matches_independent_cpu_estimator():
    outputs = []
    for device in ("cpu", "cuda"):
        prior, known, masks, noise, y = small_fixture()
        model = JointEstimator(prior, known, masks, noise, DenseBackend(device))
        outputs.append(model.estimate_coefficients(y, known, masks, [.3]))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-10, atol=2e-11)


def test_pdd_noise_uses_both_useful_windows_and_reuses_standard_noise():
    from types import SimpleNamespace
    from data_generation.src.receiver.chain import ReceiverChain
    from data_generation.src.transmitter.ofdm import window_indices
    c = load_pdd_config()
    clean = np.zeros(6232)
    clean[window_indices(c, 1)] = 2
    clean[window_indices(c, 2)] = 4
    clean[window_indices(c, 3)] = 100  # Tail P must not enter the D-window SNR definition.
    mock = SimpleNamespace(c=c, noise_gain=.5, receive=lambda x: x)
    first, variance, power = ReceiverChain.noise(mock, clean, 1, 2)
    second, _, _ = ReceiverChain.noise(mock, clean, 1, 2)
    assert power == 10  # (2^2 + 4^2)/2, neither P nor CP participates.
    np.testing.assert_allclose(variance, 20/10**(np.asarray(c["snr_db"])/10))
    np.testing.assert_array_equal(first, second)
