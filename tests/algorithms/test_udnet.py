"""Physical masks, eq. (13)--(15), whitening/ZF and label-free detection."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits

from data_generation.src.transmitter.ofdm import qpsk
from src.algorithms.interface import EqualizerBatch
from src.algorithms.udnet.model import UDNet
from src.algorithms.udnet.inference import equalize_batch
from src.common.numerics import DenseBackend
from src.data.udnet import LocalPreprocessor, WindowPlan
from src.training.udnet_supervision import multilayer_cross_entropy


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)
    with threadpool_limits(2):
        yield


def test_physical_coverage_boundaries_and_masks():
    indices = np.array([0, 1, 15, 16, 19, 31, 32, 47, 63, 64, 69])
    plan = WindowPlan.build(70, indices)
    assert plan.positions.shape[1] == 32
    assert plan.positions[0, 0] == -8
    assert plan.positions[-1, -1] == 87
    np.testing.assert_array_equal(plan.positions[:, 8:24][plan.data_mask[:, 8:24]], indices)
    values = np.broadcast_to(plan.positions[None], (3, *plan.positions.shape))
    np.testing.assert_array_equal(plan.stitch(values), np.broadcast_to(indices, (3, len(indices))))
    assert not plan.data_mask[~plan.valid].any()
    assert len(WindowPlan.build(1024, np.arange(64, 960)).positions) == 56


def test_double_whitening_and_reduced_zf_have_no_snr_scaling():
    rng = np.random.default_rng(18)
    n = 16
    indices = np.array([0, 2, 5, 9, 15])
    h = rng.normal(size=(2, n, n))+1j*rng.normal(size=(2, n, n))
    root = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
    covariance = root@root.conj().T+np.eye(n)
    x = qpsk(rng.integers(0, 2, size=(2, len(indices), 2)))
    y = (h[..., indices]@x[..., None])[..., 0]
    batch = EqualizerBatch(h, y, y, np.array([.01, 100.]), indices, covariance)
    plan = WindowPlan.build(n, indices)
    prep = LocalPreprocessor(plan, covariance)
    features = prep(batch)
    positions = np.flatnonzero(plan.data_mask[0])
    complex_zf = features["zf"][:, :32]+1j*features["zf"][:, 32:]
    np.testing.assert_allclose(complex_zf[:, positions], x, atol=1e-6)
    expected = h[..., indices].conj().swapaxes(-1, -2)@np.linalg.solve(covariance, h[..., indices])
    power = np.trace(expected, axis1=-2, axis2=-1).real/len(indices)
    np.testing.assert_allclose(features["gram"][:, positions[:, None], positions],
                               expected.real/power[:, None, None], rtol=1e-5, atol=1e-6)
    expected_match = (h[..., indices].conj().swapaxes(-1, -2)@np.linalg.solve(covariance, y[..., None]))[..., 0]
    np.testing.assert_allclose(features["matched"][:, positions], expected_match.real/power[:, None], rtol=1e-5, atol=1e-6)
    modified = EqualizerBatch(h, y, y, np.array([100., .01]), indices, covariance)
    for name, value in prep(modified).items():
        np.testing.assert_array_equal(value, features[name])
    assert features["gram"].dtype == np.float32
    assert not np.any(features["zf"][:, np.tile(~plan.data_mask[0], 2)])
    result = equalize_batch(batch, DenseBackend(), model=UDNet().eval(), preprocessor=prep)
    assert result.symbols.shape == x.shape


def test_receiver_gain_invariance_and_gradients_at_real_channel_scales():
    rng = np.random.default_rng(91)
    torch.manual_seed(23)
    n, count = 32, 4
    indices = np.arange(0, n, 2)
    h = (rng.normal(size=(count, n, n))+1j*rng.normal(size=(count, n, n)))/np.sqrt(n)
    x = qpsk(rng.integers(0, 2, size=(count, len(indices), 2)))
    y = (h[..., indices]@x[..., None])[..., 0]
    plan = WindowPlan.build(n, indices)
    covariance = np.eye(n)
    prep, model = LocalPreprocessor(plan, covariance), UDNet()
    reference = None
    for gain in (1., 1e-4, 1e4):
        features = prep(EqualizerBatch(gain*h, gain*y, gain*y, np.full(count, gain**2), indices, covariance))
        if reference is None:
            reference = features
        else:
            for key in features:
                np.testing.assert_allclose(features[key], reference[key], rtol=2e-6, atol=2e-6)
        inputs = [torch.as_tensor(features[key]) for key in ("gram", "matched", "zf", "data_mask")]
        mask = inputs[-1]
        saturation = []
        hook = model.hidden[0].register_forward_hook(
            lambda module, args, output: saturation.append(float((output.abs() > 3).float().mean())))
        probabilities = model(*inputs)
        hook.remove()
        model.zero_grad(set_to_none=True)
        multilayer_cross_entropy(probabilities, torch.zeros_like(mask, dtype=torch.long).masked_fill(~mask, -1), mask).backward()
        assert saturation[0] < .05
        assert all(torch.isfinite(layer.weight.grad).all() and layer.weight.grad.norm() > 0 for layer in model.hidden)


def test_zero_power_window_is_finite_and_uninformative():
    n = 32
    indices = np.arange(n)
    h, y = np.zeros((1, n, n), complex), np.ones((1, n), complex)
    features = LocalPreprocessor(WindowPlan.build(n, indices), np.eye(n))(
        EqualizerBatch(h, y, y, np.ones(1), indices, np.eye(n)))
    for key in ("gram", "matched", "zf"):
        assert np.isfinite(features[key]).all() and not features[key].any()


def statistics(batch=3):
    torch.manual_seed(25)
    root = torch.randn(batch, 64, 64)
    return root.transpose(-1, -2)@root/64, torch.randn(batch, 64), torch.randn(batch, 64), torch.rand(batch, 32) > .4


def test_classes_residual_probabilities_expectation_loss_and_all_gradients():
    model = UDNet()
    assert len({id(layer.weight) for layer in model.hidden}) == 22
    assert sum(p.numel() for p in model.parameters()) == 341792
    inputs = statistics()
    probabilities = model(*inputs)
    mask = inputs[-1]
    assert probabilities.shape == (3, 22, 32, 4)
    torch.testing.assert_close(probabilities.sum(-1), mask[:, None, :].expand(-1, 22, -1).float())
    assert (probabilities >= 0).all()
    torch.testing.assert_close(probabilities[:, :, ~mask[0], :][0], torch.zeros_like(probabilities[0, :, ~mask[0], :]))
    bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    constellation = model.constellation.detach().numpy()
    np.testing.assert_allclose(constellation[:, 0]+1j*constellation[:, 1], qpsk(bits), atol=1e-7)
    labels = torch.randint(0, 4, mask.shape).masked_fill(~mask, -1)
    loss = multilayer_cross_entropy(probabilities, labels, mask)
    expected = sum(-probabilities[b, :, k, labels[b, k]].log().sum()
                   for b, k in mask.nonzero())/mask.sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
    assert list(inspect.signature(model.forward).parameters) == ["gram", "matched", "zf", "data_mask"]


def test_residual_equations_and_probability_expected_symbol_feedback():
    torch.manual_seed(40)
    model = UDNet()
    gram, matched, zf, mask = statistics(2)
    captured = []
    hook = model.hidden[1].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    output = model(gram, matched, zf, mask)
    hook.remove()
    coord_mask = torch.cat((mask, mask), -1)
    state = zf*coord_mask
    v0 = model.hidden[0](torch.cat((state, matched*coord_mask,
                                 (gram@state[..., None]).squeeze(-1)*coord_mask), -1)).tanh()
    p0 = model.classifiers[0](v0).reshape(2, 32, 4).softmax(-1)*mask[..., None]
    torch.testing.assert_close(output[:, 0], p0)
    moment = p0@model.constellation
    feedback = torch.cat((moment[..., 0], moment[..., 1]), -1)*coord_mask
    torch.testing.assert_close(captured[0][:, :64], feedback)
    v1 = model.hidden[1](captured[0]).tanh()+v0
    fresh = model.classifiers[1](v1).reshape(2, 32, 4).softmax(-1)
    expected = (fresh+p0)/(fresh+p0).sum(-1, keepdim=True)*mask[..., None]
    torch.testing.assert_close(output[:, 1], expected)


def test_nondata_coordinates_cannot_enter_any_layer():
    model = UDNet()
    gram, matched, zf, mask = statistics(2)
    original = model(gram, matched, zf, mask)
    coordinate_mask = torch.cat((mask, mask), -1)
    matched = matched.masked_fill(~coordinate_mask, 1e6)
    zf = zf.masked_fill(~coordinate_mask, -1e6)
    # Rows and columns corresponding to nondata slots cannot affect data outputs.
    gram = gram.masked_fill(~(coordinate_mask[:, :, None] & coordinate_mask[:, None, :]), 1e6)
    torch.testing.assert_close(model(gram, matched, zf, mask), original)
