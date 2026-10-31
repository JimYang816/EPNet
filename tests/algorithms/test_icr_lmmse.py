"""Independent exact-solve and soft-feedback checks for the A/C detectors."""
import numpy as np
import pytest
import torch

from src.algorithms.icr_mmse.model import ICRMMSEnet, layerwise_loss
from tests.algorithms.test_icr_mmse import inputs


@pytest.mark.parametrize("noise_mode", ["diagonal", "full"])
def test_exact_solver_soft_probabilities_and_hard_decisions(noise_mode):
    torch.set_num_threads(2)
    item = inputs(size=11, batch=2, dtype=torch.complex128)
    model = ICRMMSEnet(layers=3, bandwidth=2, detector_mode="lmmse",
                       correction_mode="fixed", precision="complex128", noise_mode=noise_mode)
    result = model(**item)
    assert result.symbols.shape[1] == 1  # No redundant repeated exact solves.
    assert list(model.equalizers.parameters()) == []
    data = item["data_mask"].numpy()
    points = model.constellation.detach().numpy()
    for i in range(2):
        h = item["initial_channel"][i].numpy()
        hd = h[:, data]
        y = item["observations"][i].numpy() - h @ item["known"][i].numpy()
        c = item["unit_covariance"].numpy()
        covariance = c if noise_mode == "full" else np.diag(np.diag(c))
        w = np.linalg.inv(covariance)
        sigma = item["rf_variance"][i].item()
        system = hd.conj().T @ w @ hd + sigma*np.eye(data.sum())
        expected = np.linalg.solve(system, hd.conj().T @ w @ y)
        v = sigma*np.linalg.inv(system).diagonal().real
        gain = 1-v
        z, tau = expected/gain, v/gain
        logits = -np.abs(z[:, None]-points)**2/tau[:, None]
        p = np.exp(logits-logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        np.testing.assert_allclose(result.diagnostics[0]["linear_estimate"][i].detach(), expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result.probabilities[i, 0, data].detach(), p, rtol=1e-9, atol=1e-10)
        assert np.array_equal(result.symbols[i, 0, data].detach().numpy().real < 0, expected.real < 0)
        assert np.array_equal(result.symbols[i, 0, data].detach().numpy().imag < 0, expected.imag < 0)
    torch.testing.assert_close(result.symbols[:, 0, item["pilot_mask"]], item["known"][:, item["pilot_mask"]])
    assert result.variances[:, :, ~item["data_mask"]].count_nonzero() == 0


def test_exact_detector_propagates_detection_loss_to_csi_corrector():
    item = inputs(size=11, batch=2, dtype=torch.complex128)
    model = ICRMMSEnet(layers=2, bandwidth=2, detector_mode="lmmse",
                       precision="complex128", noise_mode="diagonal")
    result = model(**item)
    classes = torch.zeros(2, int(item["data_mask"].sum()), dtype=torch.long)
    loss = layerwise_loss(result, classes, None, item["data_mask"], lambda_h=0)
    loss.backward()
    gradient = model.corrector.head.bias.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient[:10].abs().sum() > 0
    # A finite-difference check catches a detached or incorrect complex solve.
    delta = 1e-6
    with torch.no_grad():
        model.corrector.head.bias[0] += delta
        high = layerwise_loss(model(**item), classes, None, item["data_mask"], lambda_h=0)
        model.corrector.head.bias[0] -= 2*delta
        low = layerwise_loss(model(**item), classes, None, item["data_mask"], lambda_h=0)
    torch.testing.assert_close(gradient[0], (high-low)/(2*delta), rtol=2e-4, atol=1e-7)


def test_zero_channel_produces_uniform_soft_feedback():
    item = inputs(size=8, batch=1)
    item["initial_channel"].zero_()
    model = ICRMMSEnet(layers=1, bandwidth=1, detector_mode="lmmse", correction_mode="fixed")
    output = model(**item)
    torch.testing.assert_close(output.probabilities[:, 0, item["data_mask"]], torch.full((1, 3, 4), .25))
    assert torch.isfinite(output.symbols).all()
