"""Paired full-reference/compact-reference loss and gradient checks."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.algorithms.icr_mmse.dpss import DPSSBasis, DPSSChannel, DPSSReference, reference_energy
from src.algorithms.icr_mmse.model import ICRMMSEnet, csi_nmse, layerwise_loss
from src.training.icr_mmse import configure_stage


@pytest.mark.parametrize("precision", ("complex64", "complex128"))
@pytest.mark.parametrize("stage", tuple("ABC"))
def test_exact_full_nmse_outputs_losses_and_parameter_gradients(precision, stage):
    torch.set_num_threads(2)
    torch.manual_seed(761)
    dtype, n, width = getattr(torch, precision), 17, 2
    basis = DPSSBasis(torch.randn(n, 3, dtype=dtype) / n**.5)
    coefficients = torch.randn(2, 3, 9, dtype=dtype) / 3
    raw = DPSSChannel(coefficients, basis)
    dense = raw.dense()
    # Arbitrary full truth deliberately has energy outside the saved subspace.
    truth = dense.detach() + .1 * torch.randn_like(dense)
    reference = DPSSReference.from_dense(truth, basis, 9, width)
    torch.testing.assert_close(reference_energy(reference), reference_energy(truth), rtol=0, atol=0)
    assert reference.nbytes < truth.numel() * truth.element_size()
    data, pilot = torch.arange(n) % 3 == 1, torch.arange(n) % 3 == 0
    model = ICRMMSEnet(layers=3, bandwidth=width, window_size=8, center_size=4,
                       noise_mode="diagonal", precision=precision)
    with torch.no_grad():
        model.corrector.head.weight[:10].normal_(std=.01)
        for layer in model.equalizers:
            layer.classifier.weight.normal_(std=.01)
    configure_stage(model, stage)
    known = torch.zeros(2, n, dtype=dtype)
    known[:, pilot] = (1+1j)/2**.5
    inputs = dict(observations=torch.randn(2, n, dtype=dtype), known=known,
        data_mask=data, pilot_mask=pilot, null_mask=~(data|pilot), carrier_positions=torch.arange(n),
        rf_variance=torch.full((2,), .15), unit_covariance=torch.diag(torch.linspace(.8, 1.2, n)).to(dtype))
    classes = torch.arange(2*int(data.sum())).reshape(2, -1) % 4
    captured = []
    for channel, target, reuse, diagnostics in ((dense, truth, False, True),
                                               (raw, reference, True, False)):
        model.zero_grad(set_to_none=True)
        out = model(**inputs, initial_channel=channel, correction_enabled=stage != "A",
                    detach_feedback=stage == "B", reuse_base_energy=reuse, return_diagnostics=diagnostics)
        loss = layerwise_loss(out, classes, target, data, lambda_h=0 if stage == "A" else .1,
                              csi_only=stage == "B")
        loss.backward()
        captured.append((out.probabilities.detach(), csi_nmse(out, target).detach(), loss.detach(),
                         torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])))
    rtol, atol = ((8e-4, 5e-6) if precision == "complex64" else (5e-10, 3e-12))
    for expected, actual in zip(*captured):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_reference_does_not_project_away_unmodelled_energy():
    dtype = torch.complex128
    basis = DPSSBasis(torch.ones(8, 1, dtype=dtype))
    channel = DPSSChannel(torch.ones(1, 1, 1, dtype=dtype), basis)
    truth = torch.randn(1, 8, 8, dtype=dtype)
    stats = DPSSReference.from_dense(truth, basis, 1, 1)
    torch.testing.assert_close(channel.error_energy(stats),
                               (channel.dense()-truth).abs().square().sum((-2, -1)), rtol=2e-14, atol=2e-14)
