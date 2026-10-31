"""Compare block MMSE-DFE against independent receive-space MMSE cancellation."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.algorithms.mmse_dfe.detector import equalize_batch, mmse_dfe
from src.common.numerics import DenseBackend


def system():
    rng = np.random.default_rng(816)
    h = (rng.normal(size=(3, 7, 5))+1j*rng.normal(size=(3, 7, 5)))/np.sqrt(7)
    data = np.array([0, 2, 4])
    a = rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7))
    covariance = a@a.conj().T/7+.2*np.eye(7)
    variances = np.array([.03, .3, 2.])
    known = np.array([0, 1, 0, 1j, 0])
    payload = (1-2*rng.integers(2, size=(3, 3, 2)))@np.array([1, 1j])/np.sqrt(2)
    noise = rng.normal(size=(3, 7))+1j*rng.normal(size=(3, 7))
    y = (h[..., data]@payload[..., None])[..., 0]
    y += (np.linalg.cholesky(covariance)@noise[..., None])[..., 0]*np.sqrt(variances[:, None]/2)
    return EqualizerBatch(h, y+h@known, y, variances, data, covariance)


def receive_reference(h, y, covariance):
    """Rebuild the receive covariance after each hard decision, with no Cholesky DFE."""
    estimates = np.zeros(h.shape[1], complex)
    residual = y.copy()
    for i in range(h.shape[1]-1, -1, -1):
        remaining = h[:, :i+1]
        weight = np.linalg.solve(covariance+remaining@remaining.conj().T, h[:, i])
        estimates[i] = np.vdot(weight, residual)
        decision = ((-1 if estimates[i].real < 0 else 1)
                    +1j*(-1 if estimates[i].imag < 0 else 1))/np.sqrt(2)
        residual -= h[:, i]*decision
    return estimates


def test_colored_noise_pilots_and_hard_feedback_match_receive_space_reference():
    batch, backend = system(), DenseBackend()
    result = equalize_batch(batch, backend, whitener=make_whitener(batch.noise_covariance, backend))
    expected = np.stack([receive_reference(h[:, batch.data_indices], y, v*batch.noise_covariance)
                         for h, y, v in zip(batch.channel, batch.cancelled_observations, batch.variances)])
    np.testing.assert_allclose(result.symbols, expected, atol=2e-13, rtol=2e-12)
    np.testing.assert_array_equal(result.rejected_sites, 0)
    # The first decision equals LMMSE. Subsequent decisions use hard feedback.
    for k, (h, y, v) in enumerate(zip(batch.channel, batch.cancelled_observations, batch.variances)):
        h = h[:, batch.data_indices]
        linear = h.conj().T@np.linalg.solve(v*batch.noise_covariance+h@h.conj().T, y)
        np.testing.assert_allclose(result.symbols[k, -1], linear[-1], atol=2e-13)
        assert not np.allclose(result.symbols[k, :-1], linear[:-1])


def test_receive_gain_invariance_and_rank_deficient_or_zero_channels():
    batch, backend = system(), DenseBackend()
    whitener = make_whitener(batch.noise_covariance, backend)
    expected = equalize_batch(batch, backend, whitener=whitener).symbols
    for scale in (1e-5, 1e5):
        scaled = EqualizerBatch(batch.channel*scale, batch.raw_observations*scale,
            batch.cancelled_observations*scale, batch.variances*scale**2, batch.data_indices, batch.noise_covariance)
        np.testing.assert_allclose(equalize_batch(scaled, backend, whitener=whitener).symbols, expected, atol=2e-12)
    for zero in (False, True):
        h = np.zeros_like(batch.channel) if zero else batch.channel.copy()
        h[..., 2] = h[..., 0]
        modified = EqualizerBatch(h, batch.raw_observations, batch.cancelled_observations,
                                 batch.variances, batch.data_indices, batch.noise_covariance)
        result = equalize_batch(modified, backend, whitener=whitener)
        assert np.isfinite(result.symbols).all()
        if zero:
            np.testing.assert_array_equal(result.symbols, 0.)
        else:
            reference = np.stack([receive_reference(x[:, batch.data_indices], y, v*batch.noise_covariance)
                                  for x, y, v in zip(h, batch.cancelled_observations, batch.variances)])
            np.testing.assert_allclose(result.symbols, reference, atol=2e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_path_matches_numpy_with_multiple_leading_batch_axes(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    batch, backend = system(), DenseBackend()
    from src.algorithms.likelihood import batch_likelihood
    gram, matched = batch_likelihood(batch, make_whitener(batch.noise_covariance, backend), backend)
    gram = np.stack([gram, 2*gram])
    matched = np.stack([matched, 2*matched])
    expected = mmse_dfe(gram, matched, backend)
    tensor_backend = SimpleNamespace(torch=torch, array=lambda x: torch.as_tensor(x, device=device))
    actual = mmse_dfe(torch.as_tensor(gram, device=device), torch.as_tensor(matched, device=device), tensor_backend)
    assert actual.dtype == torch.complex128
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(mmse_dfe(gram[0, 0], matched[0, 0], backend), expected[0, 0], atol=2e-13)


@pytest.mark.parametrize("gram,matched", [
    (np.eye(3), np.zeros(2, complex)),
    (np.zeros((0, 0)), np.zeros(0, complex)),
    (np.full((2, 2), np.nan), np.zeros(2, complex)),
    (np.eye(2), np.full(2, np.inf, complex)),
])
def test_invalid_likelihood_is_rejected(gram, matched):
    with pytest.raises(ValueError, match="MMSE-DFE likelihood"):
        mmse_dfe(gram, matched, DenseBackend())
