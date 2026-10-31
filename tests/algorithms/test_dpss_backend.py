"""Compare the operator to the independent legacy time-kernel construction."""
import numpy as np
import pytest
import json
from pathlib import Path

torch = pytest.importorskip("torch")

from data_generation.src.receiver.chain import taps_to_matrix
from src.algorithms.icr_mmse.dpss import DPSSBasis, DPSSChannel
from src.algorithms.icr_mmse.operators import BandChannel
from src.algorithms.icr_mmse.model import ICRMMSEnet, layerwise_loss
from src.training.icr_mmse import configure_stage


def dense_from_coefficients(basis, coefficients):
    n, q = basis.shape
    length = coefficients.shape[-1]
    taps = torch.einsum("nq,bql->bnl", basis, coefficients)
    rows = torch.arange(n)[:, None]
    columns = (rows - torch.arange(length)) % n
    matrix = torch.zeros(len(coefficients), n, n, dtype=coefficients.dtype)
    matrix = matrix.scatter(-1, columns[None].expand(len(coefficients), -1, -1), taps)
    matrix = torch.fft.ifft(torch.fft.fft(matrix, dim=-2, norm="ortho"), dim=-1, norm="ortho")
    return torch.fft.fftshift(matrix, dim=(-2, -1))


@pytest.mark.parametrize("size", (9, 16))
def test_complete_operator_entries_energies_adjoint_and_gradients(size):
    torch.set_num_threads(2)
    torch.manual_seed(15)
    basis = torch.randn(size, 3, dtype=torch.complex128, requires_grad=True)
    coefficients = torch.randn(2, 3, size-2, dtype=torch.complex128, requires_grad=True)
    dense = dense_from_coefficients(basis, coefficients)
    # NumPy legacy construction independently checks FFT order and normalization.
    for i in range(2):
        expected = taps_to_matrix((basis @ coefficients[i]).detach().numpy())
        np.testing.assert_allclose(dense[i].detach().numpy(), expected, atol=2e-14)
    raw = DPSSChannel(coefficients, DPSSBasis(basis))
    bands = torch.randn(2, size, 7, dtype=torch.complex128, requires_grad=True)
    operator, reference = BandChannel(raw, bands), BandChannel(dense, bands)
    x = torch.randn(2, size, dtype=torch.complex128, requires_grad=True)
    r = torch.randn_like(x)
    weights = torch.linspace(.7, 1.5, size, requires_grad=True, dtype=torch.float64)
    truth = torch.randn_like(dense)  # Includes energy outside the DPSS delay support.
    fast_values = (operator.matvec(x), operator.adjoint(r), operator.local_entries(),
                   operator.column_energy(weights), operator.error_energy(truth), raw.frobenius_energy())
    dense_values = (reference.matvec(x), reference.adjoint(r), reference.local_entries(),
                    reference.column_energy(weights), reference.error_energy(truth),
                    dense.abs().square().sum((-2, -1)))
    for actual, expected in zip(fast_values, dense_values):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-12)
    torch.testing.assert_close((operator.matvec(x).conj()*r).sum(-1),
                               (x.conj()*operator.adjoint(r)).sum(-1))
    parameters = (basis, coefficients, bands, x, weights)
    losses = [sum(v.abs().square().mean() for v in values) for values in (fast_values, dense_values)]
    fast_grad = torch.autograd.grad(losses[0], parameters, retain_graph=True)
    dense_grad = torch.autograd.grad(losses[1], parameters)
    for actual, expected in zip(fast_grad, dense_grad):
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-11)


@pytest.mark.parametrize("precision", ("complex64", "complex128"))
@pytest.mark.parametrize("noise_mode", ("diagonal", "full"))
@pytest.mark.parametrize("stage", tuple("ABC"))
def test_network_all_layers_repairs_loss_and_parameter_gradients(precision, noise_mode, stage):
    torch.set_num_threads(2)
    torch.manual_seed(30)
    dtype = getattr(torch, precision)
    n = 20
    basis = torch.randn(n, 3, dtype=dtype) / n**.5
    coefficients = torch.randn(2, 3, 8, dtype=dtype) / 4
    dense = dense_from_coefficients(basis, coefficients)
    raw = DPSSChannel(coefficients, DPSSBasis(basis))
    data = torch.arange(n)%3 == 1
    pilot = torch.arange(n)%3 == 0
    known = torch.zeros(2, n, dtype=dtype)
    known[:, pilot] = (1+1j)/2**.5
    root = torch.randn(n, n, dtype=dtype) / n**.5
    covariance = root @ root.mH + torch.eye(n)
    model = ICRMMSEnet(layers=3, bandwidth=3, window_size=12, center_size=4,
                       noise_mode=noise_mode, precision=precision)
    with torch.no_grad():
        model.corrector.head.weight[:14].normal_(std=.02)
        for layer in model.equalizers:
            layer.classifier.weight.normal_(std=.01)
    configure_stage(model, stage)
    common = dict(observations=torch.randn(2, n, dtype=dtype), known=known,
                  data_mask=data, pilot_mask=pilot, null_mask=~(data|pilot),
                  carrier_positions=torch.arange(n), rf_variance=torch.full((2,), .1, dtype=dense.real.dtype),
                  unit_covariance=covariance)
    classes = torch.arange(2*int(data.sum())).reshape(2, -1)%4
    truth = .9*dense.detach()
    values = []
    for initial in (dense, raw):
        model.zero_grad(set_to_none=True)
        out = model(initial_channel=initial, **common, correction_enabled=stage != "A", detach_feedback=stage == "B")
        loss = layerwise_loss(out, classes, truth, data, lambda_h=.1 if stage != "A" else 0., csi_only=stage == "B")
        loss.backward()
        values.append((out, loss.detach(), [p.grad.clone() for p in model.parameters() if p.requires_grad]))
    (a, la, ga), (b, lb, gb) = values
    rtol, atol = ((2e-4, 3e-6) if precision == "complex64" else (2e-10, 2e-12))
    for name in ("probabilities", "symbols", "variances", "bands"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=rtol, atol=atol)
    torch.testing.assert_close(la, lb, rtol=rtol, atol=atol)
    for actual, expected in zip(gb, ga):
        torch.testing.assert_close(actual, expected, rtol=rtol*10, atol=atol)


def test_basis_kernel_is_recomputed_when_weights_change():
    torch.manual_seed(19)
    n = 12
    v = torch.randn(n, 3, dtype=torch.complex128)
    coefficients = torch.randn(1, 3, 5, dtype=torch.complex128)
    basis = DPSSBasis(v)
    weights = torch.ones(n, dtype=torch.float64)
    for factor in (1., 2., 3.):
        weights[0] = factor
        raw = DPSSChannel(coefficients, basis)
        expected = (dense_from_coefficients(v, coefficients).abs().square()*weights[:, None]).sum(-2)
        torch.testing.assert_close(raw.column_energy(weights), expected)


def test_dpss_path_never_materializes_dense_during_inference(monkeypatch):
    torch.manual_seed(6)
    n = 16
    raw = DPSSChannel(torch.randn(1, 2, 6, dtype=torch.complex64),
                      DPSSBasis(torch.randn(n, 2, dtype=torch.complex64)))
    monkeypatch.setattr(DPSSChannel, "dense", lambda *a: pytest.fail("Dense channel materialized"))
    data = torch.arange(n)%2 == 0
    model = ICRMMSEnet(layers=2, bandwidth=2, noise_mode="diagonal")
    out = model(torch.randn(1, n, dtype=torch.complex64), raw, torch.zeros(1, n, dtype=torch.complex64),
                data, ~data, torch.zeros(n, dtype=torch.bool), torch.arange(n), torch.ones(1),
                torch.eye(n, dtype=torch.complex64))
    out.probabilities[..., 0].sum().backward()
    assert torch.isfinite(out.bands).all()


@pytest.mark.parametrize("block", (1, 2))
def test_formal_size_and_fixed_dpss_configuration(block):
    from data_generation.src.receiver.dpss import build_model
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    config = json.loads((Path(__file__).resolve().parents[2]/"configs/data/nof1_pddp.json").read_text())
    rng = np.random.default_rng(71)
    with threadpool_limits(2):
        saved = build_model(config)
        n, q, length = config["nfft"], saved.count, saved.length
        assert (n, q, length) == (1024, 34, 736)
        basis = saved.sampled_basis[block*n:(block+1)*n]
        coefficients = (rng.normal(size=(q, length))+1j*rng.normal(size=(q, length)))/np.sqrt(q*length)
        expected = torch.tensor(taps_to_matrix(basis @ coefficients))[None]
    raw = DPSSChannel(torch.tensor(coefficients)[None], DPSSBasis(torch.tensor(basis, dtype=torch.complex128)))
    x = torch.tensor(rng.normal(size=(1, n))+1j*rng.normal(size=(1, n)))
    torch.testing.assert_close(raw.matvec(x), (expected @ x[..., None]).squeeze(-1), rtol=2e-11, atol=2e-14)
    torch.testing.assert_close(raw.adjoint(x), (expected.mH @ x[..., None]).squeeze(-1), rtol=2e-11, atol=2e-14)
    weights = torch.linspace(.6, 1.4, n, dtype=torch.float64)
    torch.testing.assert_close(raw.column_energy(weights),
                               (expected.abs().square()*weights[:, None]).sum(-2), rtol=2e-11, atol=2e-14)
