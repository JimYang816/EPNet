"""Convergence, physical scaling, gradients and the full-channel work budget."""
from unittest.mock import patch

import pytest
import torch

from src.algorithms.icr_mmse.model import ICRMMSEnet, layerwise_loss
from src.algorithms.icr_mmse.operators import BandChannel
from tests.algorithms.test_icr_mmse import inputs


@pytest.mark.parametrize("noise_mode", ["diagonal", "full"])
def test_loaded_line_search_descends_for_correlated_channels(noise_mode):
    torch.set_num_threads(2)
    item = inputs(14, 2, torch.complex128)
    item["initial_channel"] = torch.ones_like(item["initial_channel"])
    item["unit_covariance"] = torch.eye(14, dtype=torch.complex128)
    model = ICRMMSEnet(layers=4, bandwidth=1, detector_mode="budgeted", correction_mode="fixed",
                       initial_soft_damping=.5, precision="complex128", noise_mode=noise_mode)
    result = model(**item)
    for row in result.diagnostics:
        assert torch.all(row["linear_objective_after"] <= row["linear_objective_before"] + 1e-10)
    assert torch.isfinite(result.log_probabilities).all()
    mask = item["data_mask"]
    torch.testing.assert_close(result.probabilities[:, :, mask].sum(-1),
                               torch.ones_like(result.variances[:, :, mask]))
    torch.testing.assert_close(result.variances, (1-result.symbols.abs().square()).clamp_min(0)*mask)


def test_budget_has_three_full_actions_per_layer_and_no_factorization():
    item = inputs(20, 2)
    calls = {"matvec": 0, "adjoint": 0}
    mv, adj = BandChannel.matvec, BandChannel.adjoint

    def matvec(self, value):
        calls["matvec"] += 1
        return mv(self, value)

    def adjoint(self, value):
        calls["adjoint"] += 1
        return adj(self, value)

    counts = {}
    for mode in ("unfolded", "budgeted"):
        model = ICRMMSEnet(layers=3, bandwidth=1, detector_mode=mode, noise_mode="diagonal")
        calls.update(matvec=0, adjoint=0)
        with patch.object(BandChannel, "matvec", matvec), patch.object(BandChannel, "adjoint", adjoint), \
                patch.object(BandChannel, "dense", side_effect=AssertionError("No dense channel copy")), \
                patch("torch.linalg.cholesky", side_effect=AssertionError("No factorization")):
            model(**item, return_diagnostics=False)
        counts[mode] = dict(calls)
    # One additional forward action/layer is used by the unchanged CSI corrector.
    assert sum(counts["unfolded"].values()) == 15
    assert sum(counts["budgeted"].values()) == 12


def test_budgeted_detector_is_gain_invariant_and_all_parameters_receive_gradients():
    item = inputs(17, 2, torch.complex128)
    model = ICRMMSEnet(layers=3, bandwidth=1, detector_mode="budgeted", correction_mode="fixed",
                       initial_soft_damping=.5, precision="complex128", noise_mode="diagonal")
    output = model(**item, return_diagnostics=False)
    scaled = dict(item, initial_channel=item["initial_channel"]*7,
                  observations=item["observations"]*7, rf_variance=item["rf_variance"]*49)
    other = model(**scaled, return_diagnostics=False)
    torch.testing.assert_close(output.probabilities, other.probabilities, rtol=1e-10, atol=1e-10)
    classes = torch.zeros(2, int(item["data_mask"].sum()), dtype=torch.long)
    loss = layerwise_loss(output, classes, None, item["data_mask"], lambda_h=0,
                         objective="bit_ce", final_weight=.9)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.equalizers.parameters())
    assert model.equalizers[-1].csi_loading_log.grad.abs() > 0

