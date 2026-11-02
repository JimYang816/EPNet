"""Independent receive-space references for the OAMP-Net2 detector."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.algorithms.oamp_net2.inference import equalize_batch as equalize_oamp
from src.algorithms.oamp_net2.model import FEATURES, OAMPNet2
from src.common.numerics import DenseBackend
from src.data.oamp_net2 import spectral_features

CONSTELLATION = np.array([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2.)


def posterior(z, variance):
    logits = -abs(z[..., None]-CONSTELLATION)**2/variance[..., None]
    weights = np.exp(logits-logits.max(axis=-1, keepdims=True))
    return (weights@CONSTELLATION)/weights.sum(axis=-1)


def system(seed=51):
    rng = np.random.default_rng(seed)
    h = (rng.normal(size=(2, 7, 5))+1j*rng.normal(size=(2, 7, 5)))/np.sqrt(7)
    data = np.array([0, 2, 4])
    a = rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7))
    covariance = a@a.conj().T/7+np.eye(7)*.2
    variances = np.array([.15, .7])
    known = np.zeros(5, complex)
    known[[1, 3]] = [1, 1j]
    payload = CONSTELLATION[rng.integers(0, 4, size=(2, 3))]
    residual = (h[..., data]@payload[..., None])[..., 0]
    noise = rng.normal(size=(2, 7))+1j*rng.normal(size=(2, 7))
    residual += (np.linalg.cholesky(covariance)@noise[..., None])[..., 0]*np.sqrt(variances[:, None]/2)
    raw = residual+h@known
    return EqualizerBatch(h, raw, residual, variances, data, covariance)


def oamp_reference(h, y, model):
    n, m = h.shape[1], h.shape[0]
    estimate = np.zeros(n, complex)
    outputs = []
    for gamma, theta, phi, xi in zip(*(p.detach().numpy() for p in (model.gamma, model.theta, model.phi, model.xi))):
        residual = y-h@estimate
        v = max((np.vdot(residual, residual).real-m)/np.sum(abs(h)**2), model.config["variance_floor"])
        w = v*h.conj().T@np.linalg.inv(v*h@h.conj().T+np.eye(m))
        w *= n/np.trace(w@h).real
        r = estimate+gamma*w@residual
        c = np.eye(n)-theta*w@h
        tau = max((np.sum(abs(c)**2)*v+theta**2*np.sum(abs(w)**2))/n, model.config["variance_floor"])
        estimate = phi*(posterior(r, np.full(n, tau))-xi*r)
        outputs.append(estimate.copy())
    return np.stack(outputs)


def test_oamp_every_layer_matches_dense_complex_paper_equations_and_has_gradients():
    batch, backend = system(), DenseBackend()
    whitener = make_whitener(batch.noise_covariance, backend)
    features = spectral_features(batch, whitener, backend)
    model = OAMPNet2(layers=3)
    with torch.no_grad():
        model.gamma.copy_(torch.tensor([.9, .8, 1.1]))
        model.theta.copy_(torch.tensor([1.05, .7, .95]))
        model.phi.copy_(torch.tensor([.95, 1.1, .9]))
        model.xi.copy_(torch.tensor([.05, -.03, .07]))
    outputs = model(*(torch.from_numpy(features[name]) for name in FEATURES), return_layers=True)
    expected = []
    for h, y, variance in zip(batch.channel, batch.cancelled_observations, batch.variances):
        hw = whitener@h[:, batch.data_indices]/np.sqrt(variance)
        yw = whitener@y/np.sqrt(variance)
        expected.append(oamp_reference(hw, yw, model))
    np.testing.assert_allclose(outputs.detach(), np.stack(expected), rtol=3e-11, atol=3e-12)
    outputs[:, -1].abs().square().sum().backward()
    assert sum(p.numel() for p in model.parameters()) == 12
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.count_nonzero(p.grad) == 3 for p in model.parameters())


def test_receive_gain_invariance_zero_and_rank_deficient_channels():
    batch, backend = system(), DenseBackend()
    whitener = make_whitener(batch.noise_covariance, backend)
    model = OAMPNet2(layers=3)
    reference = equalize_oamp(batch, backend, model=model, whitener=whitener).symbols
    for scale in (1e-5, 1e5):
        scaled = EqualizerBatch(batch.channel*scale, batch.raw_observations*scale,
            batch.cancelled_observations*scale, batch.variances*scale**2, batch.data_indices, batch.noise_covariance)
        actual = equalize_oamp(scaled, backend, model=model, whitener=whitener).symbols
        np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=2e-10)
    for zero in (False, True):
        h = np.zeros_like(batch.channel) if zero else batch.channel.copy()
        if not zero:
            h[..., 2] = h[..., 0]
        modified = EqualizerBatch(h, batch.raw_observations, batch.cancelled_observations,
                                 batch.variances, batch.data_indices, batch.noise_covariance)
        result = equalize_oamp(modified, backend, model=model, whitener=whitener)
        assert np.isfinite(result.symbols).all()
        if zero:
            np.testing.assert_array_equal(result.symbols, 0.)


def test_torch_spectral_preprocessing_matches_numpy_without_cuda_hardware():
    batch = system()
    backend = SimpleNamespace(torch=torch, device="cpu", array=torch.as_tensor,
                              numpy=lambda value: value.detach().cpu().numpy())
    white_cpu = make_whitener(batch.noise_covariance, DenseBackend())
    white = torch.as_tensor(white_cpu)
    features = spectral_features(batch, white, backend)
    numpy_features = spectral_features(batch, white_cpu, DenseBackend())
    # Eigenvector phases are arbitrary; compare the represented Gram/matched.
    for f in (features, numpy_features):
        f["gram"] = (f["eigenvectors"]*f["eigenvalues"][:, None, :])@f["eigenvectors"].conj().swapaxes(-1, -2)
        f["matched"] = (f["eigenvectors"]@f["projected_matched"][..., None])[..., 0]
    np.testing.assert_allclose(features["gram"], numpy_features["gram"], atol=2e-13)
    np.testing.assert_allclose(features["matched"], numpy_features["matched"], atol=2e-13)
