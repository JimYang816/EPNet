"""Convergence bounds, stable bit supervision and consistent soft damping."""
from dataclasses import replace

import pytest
import torch

from src.algorithms.icr_mmse.model import ICRMMSEnet, ICRMMSEOutput, layerwise_loss
from tests.algorithms.test_icr_mmse import inputs


@pytest.mark.parametrize("noise_mode", ["diagonal", "full"])
def test_rayleigh_step_decreases_full_quadratic_even_when_unit_step_is_unstable(noise_mode):
    torch.set_num_threads(2)
    item = inputs(size=14, batch=2, dtype=torch.complex128)
    # Strongly correlated columns make the preconditioned eigenvalue exceed 2.
    item["initial_channel"] = torch.ones_like(item["initial_channel"])
    item["unit_covariance"] = torch.eye(14, dtype=torch.complex128)
    item["observations"] = torch.ones_like(item["observations"])*3
    fixed = ICRMMSEnet(layers=1, bandwidth=1, correction_mode="fixed", precision="complex128", noise_mode=noise_mode)
    safe = ICRMMSEnet(**dict(fixed.config, step_mode="rayleigh"))
    safe.load_state_dict(fixed.state_dict())
    a, b = fixed(**item), safe(**item)
    assert torch.all(a.diagnostics[0]["linear_objective_after"] > a.diagnostics[0]["linear_objective_before"])
    assert torch.all(b.diagnostics[0]["linear_objective_after"] <= b.diagnostics[0]["linear_objective_before"]+1e-10)
    assert torch.all(b.diagnostics[0]["linear_step"] < 1)


def test_soft_damping_preserves_distribution_mean_variance_and_pilots():
    item = inputs(size=11, batch=2, dtype=torch.complex128)
    raw = ICRMMSEnet(layers=1, bandwidth=1, correction_mode="fixed", precision="complex128")
    damped = ICRMMSEnet(**dict(raw.config, initial_soft_damping=.5))
    damped.load_state_dict(raw.state_dict(), strict=False)
    a, b = raw(**item), damped(**item)
    mask = item["data_mask"]
    torch.testing.assert_close(b.probabilities[:, :, mask], .5*a.probabilities[:, :, mask]+.125)
    torch.testing.assert_close(b.log_probabilities[:, :, mask].exp(), b.probabilities[:, :, mask])
    torch.testing.assert_close(b.symbols[:, 0, mask], .5*a.symbols[:, 0, mask])
    torch.testing.assert_close(b.variances[:, 0, mask], 1-b.symbols[:, 0, mask].abs().square())
    torch.testing.assert_close(b.symbols[:, 0, item["pilot_mask"]], item["known"][:, item["pilot_mask"]])


def test_bit_nll_is_stable_and_has_gradient_for_underflowed_wrong_probability():
    logits = torch.tensor([[[[1000., 0., 0., -1000.]]]], requires_grad=True)
    logp = logits.log_softmax(-1)
    output = ICRMMSEOutput(logits.softmax(-1), None, None, (), (), logp)
    classes = torch.tensor([[3]])
    mask = torch.tensor([True])
    legacy = layerwise_loss(output, classes, None, mask, lambda_h=0)
    old_gradient, = torch.autograd.grad(legacy, logits, retain_graph=True)
    loss = layerwise_loss(output, classes, None, mask, lambda_h=0, objective="bit_ce", final_weight=1.)
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert old_gradient.count_nonzero() == 0
    assert gradient.abs().sum() > 0


def test_final_bit_loss_matches_qpsk_mapping_and_final_layer_weighting():
    logits = torch.tensor([[[[.2, -.1, .4, -.3]], [[1., 2., -1., 0.]]]], requires_grad=True)
    p = logits.softmax(-1)
    output = ICRMMSEOutput(p, None, None, (), (), logits.log_softmax(-1))
    classes, mask = torch.tensor([[1]]), torch.tensor([True])  # bits 0,1
    expected = -.5*((p[0, 1, 0, :2].sum()).log() + p[0, 1, 0, (1, 3)].sum().log())
    loss = layerwise_loss(output, classes, None, mask, 0., objective="bit_ce", final_weight=1.)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[:, 0].count_nonzero() == 0
    assert logits.grad[:, 1].abs().sum() > 0


def test_safe_model_preserves_gradients_through_multiple_soft_iterations():
    item = inputs(size=11, batch=2, dtype=torch.complex128)
    model = ICRMMSEnet(layers=3, bandwidth=1, correction_mode="fixed", precision="complex128",
                       step_mode="rayleigh", initial_soft_damping=.5)
    result = model(**item)
    classes = torch.zeros(2, int(item["data_mask"].sum()), dtype=torch.long)
    loss = layerwise_loss(result, classes, None, item["data_mask"], 0., objective="bit_ce", final_weight=.9)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.equalizers.parameters())
    assert model.equalizers[-1].classifier.bias.grad.abs().sum() > 0
