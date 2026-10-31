"""Physical alignment, full-matrix operators, stage ordering and gradients."""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.icr_mmse.model import (
    FEEDBACK_SLICES, ICRMMSEnet, build_feedback_features, csi_nmse, layerwise_loss,
)
from src.algorithms.icr_mmse.operators import BandChannel, band_indices


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def inputs(size=20, batch=2, dtype=torch.complex64):
    torch.manual_seed(17)
    channel = torch.randn(batch, size, size, dtype=dtype) / size**.5
    channel = channel + 2 * torch.eye(size)[None]
    data = torch.arange(size) % 3 == 1
    pilot = torch.arange(size) % 3 == 0
    null = ~(data | pilot)
    known = torch.zeros(batch, size, dtype=channel.dtype)
    known[:, pilot] = (1 + 1j) / 2**.5
    root = torch.randn(size, size, dtype=channel.dtype) / size**.5
    covariance = root @ root.mH + torch.eye(size)
    return dict(observations=torch.randn(batch, size, dtype=channel.dtype), initial_channel=channel,
                known=known, data_mask=data, pilot_mask=pilot, null_mask=null,
                carrier_positions=torch.arange(size), rf_variance=torch.full((batch,), .2, dtype=channel.real.dtype),
                unit_covariance=covariance)


def test_full_base_plus_band_matvec_adjoint_energy_and_nmse():
    item = inputs(11, dtype=torch.complex128)
    base = item["initial_channel"].requires_grad_()
    bands = torch.randn(2, 11, 7, dtype=base.dtype, requires_grad=True)
    operator = BandChannel(base, bands)
    dense = operator.dense()
    x = torch.randn(2, 11, dtype=base.dtype)
    y = torch.randn_like(x)
    torch.testing.assert_close(operator.matvec(x), (dense @ x[..., None]).squeeze(-1))
    torch.testing.assert_close(operator.adjoint(y), (dense.mH @ y[..., None]).squeeze(-1))
    torch.testing.assert_close((operator.matvec(x).conj() * y).sum(-1),
                               (x.conj() * operator.adjoint(y)).sum(-1))
    weights = torch.rand(2, 11, dtype=torch.float64)
    torch.testing.assert_close(operator.column_energy(weights),
                               (dense.abs().square() * weights[..., None]).sum(-2))
    reference = torch.randn_like(base)
    torch.testing.assert_close(operator.error_energy(reference),
                               (dense - reference).abs().square().sum((-2, -1)))
    # Base entries outside the local band must survive exactly.
    rows = torch.arange(11)[:, None]
    far = (torch.arange(11)[None, :] - rows).abs() > 3
    torch.testing.assert_close(dense[:, far], base[:, far])
    (operator.matvec(x).abs().square().mean() + operator.error_energy(reference).mean()).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in (base, bands))


def test_explicit_k_plus_16_alignment_conjugation_known_slots_and_boundaries():
    item = inputs(40, 1, dtype=torch.complex128)
    h = item["initial_channel"]
    channel = BandChannel(h, torch.zeros(1, 40, 33, dtype=h.dtype))
    mean = item["known"].clone()
    variance = item["data_mask"].double()[None]
    k, j = 3, 19  # j is a data carrier; no compressed-carrier interpretation.
    before = build_feedback_features(h, channel, item["observations"], mean, variance,
                                     item["data_mask"], item["pilot_mask"], item["null_mask"])
    mean[:, j] = .3 + .4j
    after = build_feedback_features(h, channel, item["observations"], mean, variance,
                                    item["data_mask"], item["pilot_mask"], item["null_mask"])
    value = after.offsets[0, k, 32]
    torch.testing.assert_close(value[FEEDBACK_SLICES["mean"]], torch.tensor([.3, .4], dtype=torch.float64))
    assert not torch.equal(value, before.offsets[0, k, 32])
    residual = item["observations"] - channel.matvec(mean)
    product = residual[0, k] * mean[0, j].conj()
    torch.testing.assert_close(value[FEEDBACK_SLICES["residual_mean"]], torch.view_as_real(product))
    torch.testing.assert_close(value[FEEDBACK_SLICES["channel_variance"]], torch.view_as_real(h[0, k, j]))
    assert value[FEEDBACK_SLICES["data"]] == 1
    torch.testing.assert_close(after.residual, torch.view_as_real(residual))
    assert after.offsets[0, 0, :16].count_nonzero() == 0
    assert after.offsets[0, -1, 17:].count_nonzero() == 0
    # Non-data feedback comes from actual known symbols, never from labels.
    torch.testing.assert_close(after.offsets[0, 3, 16, FEEDBACK_SLICES["mean"]],
                               torch.view_as_real(item["known"][0, 3]))
    # Negative offsets address the same uncompressed physical grid.
    mean[:, 1] = -.2 + .1j
    negative = build_feedback_features(h, channel, item["observations"], mean, variance,
                                      item["data_mask"], item["pilot_mask"], item["null_mask"])
    torch.testing.assert_close(negative.offsets[0, 17, 0, FEEDBACK_SLICES["mean"]],
                               torch.view_as_real(mean[0, 1]))
    without_v = build_feedback_features(h, channel, item["observations"], mean, variance,
        item["data_mask"], item["pilot_mask"], item["null_mask"], use_uncertainty=False)
    assert without_v.offsets[..., FEEDBACK_SLICES["variance"]].count_nonzero() == 0
    assert without_v.offsets[..., FEEDBACK_SLICES["channel_variance"]].count_nonzero() == 0


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_compact_features_recover_legacy_information_with_physical_products(dtype):
    item = inputs(9, 2, dtype=dtype)
    initial = item["initial_channel"]
    increment = .1 * torch.randn(2, 9, 9, dtype=dtype)
    channel = BandChannel(initial, increment)
    mean = item["known"].clone()
    mean[:, item["data_mask"]] = .3 + .4j
    variance = (1 - mean.abs().square()) * item["data_mask"]
    feedback = build_feedback_features(initial, channel, item["observations"], mean, variance,
        item["data_mask"], item["pilot_mask"], item["null_mask"])
    offsets = feedback.offsets
    assert offsets.shape == (2, 9, 9, 14)
    assert feedback.residual.shape == (2, 9, 2)
    valid = offsets[..., FEEDBACK_SLICES["valid"]]
    data = offsets[..., FEEDBACK_SLICES["data"]]
    pilot = offsets[..., FEEDBACK_SLICES["pilot"]]
    null = valid - data - pilot
    assert torch.all((null == 0) | (null == 1))
    repeated_residual = feedback.residual[:, :, None, :].expand(-1, -1, 9, -1) * valid
    recovered = torch.cat((offsets[..., :4], repeated_residual,
        offsets[..., FEEDBACK_SLICES["mean"]], offsets[..., FEEDBACK_SLICES["variance"]],
        offsets[..., FEEDBACK_SLICES["residual_mean"]], offsets[..., FEEDBACK_SLICES["channel_variance"]],
        data, pilot, null, valid), -1)
    # Build the historical 17-field representation independently on the explicit
    # full matrix, including nonzero band increments and all boundary positions.
    dense = channel.dense()
    residual = item["observations"] - (dense @ mean[..., None]).squeeze(-1)
    expected = torch.zeros(2, 9, 9, 17, dtype=initial.real.dtype)
    for k in range(9):
        for offset, d in enumerate(range(-4, 5)):
            j = k + d
            if not 0 <= j < 9:
                continue
            product = residual[:, k] * mean[:, j].conj()
            uncertain_channel = dense[:, k, j] * variance[:, j]
            expected[:, k, offset] = torch.cat((torch.view_as_real(initial[:, k, j]),
                torch.view_as_real(dense[:, k, j]), torch.view_as_real(residual[:, k]),
                torch.view_as_real(mean[:, j]), variance[:, j, None], torch.view_as_real(product),
                torch.view_as_real(uncertain_channel),
                torch.stack((item["data_mask"][j], item["pilot_mask"][j], item["null_mask"][j],
                             torch.tensor(True))).to(initial.real.dtype)[None].expand(2, -1)), -1)
    torch.testing.assert_close(recovered, expected)
    torch.testing.assert_close(feedback.residual, torch.view_as_real(residual))


def test_shared_correction_uses_467_channels_for_all_eight_iterations():
    model = ICRMMSEnet(layers=8, bandwidth=16)
    assert model.corrector.head.weight[:66].count_nonzero() == 0
    assert model.corrector.head.bias[:66].count_nonzero() == 0
    captured = []
    hook = model.corrector.conv1.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    item = inputs(40, 1)
    output = model(**item)
    hook.remove()
    assert len(captured) == 8
    assert all(value.shape == (1, 467, 40) for value in captured)
    assert len([module for module in model.modules() if module is model.corrector]) == 1
    assert output.probabilities.shape == (1, 8, 40, 4)
    torch.testing.assert_close(output.probabilities, model(**item, correction_enabled=False).probabilities)
    # Exercise nonzero B=16 updates: every layer must retain the raw CSI outside the band.
    original = item["initial_channel"].clone()
    with torch.no_grad():
        model.corrector.head.bias[:66].fill_(.25)
    repaired = model(**item)
    positions = torch.arange(40)
    outside = (positions[:, None] - positions[None]).abs() > 16
    for channel in repaired.channels:
        assert channel.bandwidth == 16
        assert channel.bands.count_nonzero() > 0
        assert torch.equal(channel.dense()[:, outside], original[:, outside])
    assert torch.equal(item["initial_channel"], original)


def test_zero_repair_fixed_csi_uniform_prior_probabilities_and_all_finite_gradients():
    item = inputs()
    model = ICRMMSEnet(layers=3, bandwidth=2, window_size=8, center_size=4)
    initial_states = []
    hook = model.equalizers[0].register_forward_pre_hook(
        lambda module, args: initial_states.append((args[3].detach(), args[4].detach())))
    output = model(**item)
    hook.remove()
    fixed = model(**item, correction_enabled=False)
    torch.testing.assert_close(output.probabilities, fixed.probabilities)
    torch.testing.assert_close(initial_states[0][0], item["known"])
    torch.testing.assert_close(initial_states[0][1], item["data_mask"].to(item["initial_channel"].real.dtype)[None].expand(2, -1))
    torch.testing.assert_close(output.probabilities.sum(-1),
                               item["data_mask"].to(item["initial_channel"].real.dtype)[None, None].expand(2, 3, -1))
    torch.testing.assert_close(output.variances[..., ~item["data_mask"]],
                               torch.zeros_like(output.variances[..., ~item["data_mask"]]))
    for operator in output.channels:
        torch.testing.assert_close(operator.dense(), item["initial_channel"])
    reference = .8 * item["initial_channel"]
    expected = ((item["initial_channel"] - reference).abs().square().sum((-2, -1))
                / reference.abs().square().sum((-2, -1)))
    torch.testing.assert_close(csi_nmse(output, reference), expected[:, None].expand(-1, 3))
    classes = torch.zeros(2, int(item["data_mask"].sum()), dtype=torch.long)
    assert torch.isfinite(layerwise_loss(output, classes, None, item["data_mask"], lambda_h=0))
    loss = layerwise_loss(output, classes, reference, item["data_mask"])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.corrector.head.weight.grad[:10].abs().sum() > 0
    assert len({id(layer.step_logit) for layer in model.equalizers}) == 3
    assert list(inspect.signature(model.forward).parameters) == ["observations", "initial_channel", "known",
        "data_mask", "pilot_mask", "null_mask", "carrier_positions", "rf_variance", "unit_covariance",
        "correction_enabled", "detach_feedback", "return_diagnostics", "reuse_base_energy"]


def test_changed_channel_recomputes_pilot_residual_features_and_later_decisions():
    item = inputs()
    model = ICRMMSEnet(layers=3, bandwidth=2, window_size=8, center_size=4)
    fixed = model(**item, correction_enabled=False)
    with torch.no_grad():
        model.corrector.head.bias[:5].fill_(.6)
        model.corrector.head.bias[5:10].fill_(.2)
    repaired = model(**item)
    for index, operator in enumerate(repaired.channels):
        detail = repaired.diagnostics[index]
        torch.testing.assert_close(detail["pilot_cancelled"], item["observations"] - operator.matvec(item["known"]))
        previous = item["known"] if index == 0 else repaired.symbols[:, index - 1]
        torch.testing.assert_close(detail["residual"], item["observations"] - operator.matvec(previous))
        assert not torch.equal(detail["pilot_cancelled"], fixed.diagnostics[index]["pilot_cancelled"])
        assert not torch.equal(detail["matched"], fixed.diagnostics[index]["matched"])
        assert not torch.equal(repaired.probabilities[:, index], fixed.probabilities[:, index])
    assert not torch.equal(repaired.channels[0].bands, repaired.channels[1].bands)
    # One correction is held fixed while all equalizer layers still recompute.
    once = ICRMMSEnet(**{**model.config, "correction_mode": "once"})
    once.load_state_dict(model.state_dict())
    result = once(**item)
    torch.testing.assert_close(result.channels[0].bands, result.channels[-1].bands)


def test_pretraining_uses_live_model_feedback_and_stops_its_gradients_only():
    item = inputs()
    model = ICRMMSEnet(layers=3, bandwidth=2, window_size=8, center_size=4)
    with torch.no_grad():
        model.corrector.head.weight[:10].normal_(std=.02)
    model.corrector.requires_grad_(True)
    model.equalizers.requires_grad_(False)
    captured = []
    hook = model.corrector.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].offsets.detach()))
    output = model(**item, detach_feedback=True)
    hook.remove()
    assert len(captured) == 3
    assert not torch.equal(captured[0][..., FEEDBACK_SLICES["mean"]], captured[1][..., FEEDBACK_SLICES["mean"]])
    assert not torch.equal(captured[1][..., FEEDBACK_SLICES["mean"]], captured[2][..., FEEDBACK_SLICES["mean"]])
    reference = .9 * item["initial_channel"]
    csi_nmse(output, reference).mean().backward()
    assert all(p.grad is None for p in model.equalizers.parameters())
    assert model.corrector.head.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.eta_logits.grad).all()
    # Input gain changes preserve the detection and repair behavior.
    with torch.no_grad():
        changed = {**item, "initial_channel": 1e-3 * item["initial_channel"],
                   "observations": 1e-3 * item["observations"],
                   "rf_variance": 1e-6 * item["rf_variance"]}
        other = model(**changed, detach_feedback=True)
    torch.testing.assert_close(other.probabilities, output.probabilities, rtol=1e-4, atol=1e-5)


def test_correction_architecture_exact_and_noise_changes_model_state():
    model = ICRMMSEnet(layers=2, bandwidth=16)
    assert model.corrector.conv1.kernel_size == model.corrector.conv2.kernel_size == (3,)
    assert model.corrector.conv1.in_channels == 467
    assert model.corrector.conv1.out_channels == model.corrector.conv2.out_channels == 32
    assert model.corrector.head.kernel_size == (1,)
    assert model.corrector.head.out_channels == 99
    assert sum(parameter.numel() for parameter in model.corrector.parameters()) == 51235
    assert (model.damping > 0).all() and (model.damping < 1).all()
    item = inputs()
    first = model(**item)
    second = model(**{**item, "rf_variance": item["rf_variance"] * 100})
    assert not torch.equal(first.probabilities, second.probabilities)


def test_once_budget_repeats_are_effective_without_additional_parameters():
    item = inputs()
    model = ICRMMSEnet(layers=2, bandwidth=2, correction_mode="once", window_size=8, center_size=4)
    longer = ICRMMSEnet(**{**model.config, "equalizer_repeats": 3})
    longer.load_state_dict(model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in longer.parameters())
    original, repeated = model(**item), longer(**item)
    assert not torch.equal(original.probabilities, repeated.probabilities)
    assert len(repeated.channels) == model.config["layers"]


def test_repeat_feedback_stop_and_uncertainty_ablation_cover_detector_features():
    item = inputs()
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4,
                      equalizer_repeats=3, use_uncertainty=False)
    with torch.no_grad():
        model.corrector.head.weight[:10].normal_(std=.02)
    states = []
    hook = model.equalizers[0].register_forward_pre_hook(
        lambda module, args: states.append(args[3].requires_grad))
    model(**item, detach_feedback=True)
    hook.remove()
    assert states == [False, False, False]
    # Replace only the variance feedback at every equalizer input. Without
    # uncertainty neither the local features nor the denoiser may depend on it.
    baseline = model(**item)
    hooks = []
    for layer in model.equalizers:
        def replace_variance(module, args):
            return (*args[:4], torch.full_like(args[4], 1000.), *args[5:])
        hooks.append(layer.register_forward_pre_hook(replace_variance))
    changed = model(**item)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(changed.probabilities, baseline.probabilities, atol=0, rtol=0)


def test_noise_cholesky_reused_across_batches_variances_and_equal_copies(monkeypatch):
    item = inputs(12)
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4)
    original, calls = torch.linalg.cholesky, []
    def counted(value):
        calls.append(value.shape)
        return original(value)
    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    model(**item)
    changed = {**item, "unit_covariance": item["unit_covariance"].clone(),
               "rf_variance": item["rf_variance"] * 2}
    cached = model(**changed)
    assert len(calls) == 1
    assert not any("cache" in name for name in model.state_dict())
    model._noise_factor_cache = None
    fresh = model(**changed)
    assert len(calls) == 2
    torch.testing.assert_close(cached.probabilities, fresh.probabilities, atol=0, rtol=0)
    # Cache invalidates even if .to() keeps the same device.
    model.to("cpu")
    assert model._noise_factor_cache is None


def test_noise_cholesky_detects_inplace_and_shared_numpy_changes(monkeypatch):
    item = inputs(12)
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4)
    original, calls = torch.linalg.cholesky, []
    def counted(value):
        calls.append(value.shape)
        return original(value)
    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    array = item["unit_covariance"].numpy().copy()
    item["unit_covariance"] = torch.from_numpy(array)
    model(**item)
    array[0, 0] += .5
    model(**item)
    item["unit_covariance"][1, 1] += .5
    model(**item)
    assert len(calls) == 3


def test_noise_cholesky_rebuilds_for_dtype_shape_and_failed_replacement():
    model = ICRMMSEnet(layers=1, bandwidth=1)
    covariance = inputs(12)["unit_covariance"]
    first = model._noise_factor(covariance, torch.complex128, "cpu")
    assert model._noise_factor(covariance.clone(), torch.complex128, "cpu") is first
    lower_precision = model._noise_factor(covariance, torch.complex64, "cpu")
    assert lower_precision.dtype == torch.complex64
    small = model._noise_factor(covariance[:8, :8], torch.complex128, "cpu")
    assert small.shape == (8, 8)
    with pytest.raises(torch.linalg.LinAlgError):
        model._noise_factor(-covariance[:8, :8], torch.complex128, "cpu")
    assert model._noise_factor(covariance[:8, :8], torch.complex128, "cpu") is small


def test_inference_created_noise_cache_can_be_used_for_training():
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4)
    with torch.inference_mode():
        item = inputs(12)
        model(**item)
        # Inference tensors have no mutation version: content changes still invalidate.
        first = model._noise_factor_cache[1]
        item["unit_covariance"][0, 0] += .5
        model(**item)
        assert model._noise_factor_cache[1] is not first
    assert not model._noise_factor_cache[1].is_inference()
    factor = model._noise_factor_cache[1]
    training_inputs = {**inputs(12), "unit_covariance": item["unit_covariance"]}
    result = model(**training_inputs)
    assert model._noise_factor_cache[1] is factor
    result.probabilities[..., 0].sum().backward()
    assert torch.isfinite(model.equalizers[0].step_logit.grad)


def test_differentiable_covariance_bypasses_cache_for_repeated_backward(monkeypatch):
    item = inputs(12)
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4)
    model(**item)
    item["unit_covariance"] = item["unit_covariance"].clone().requires_grad_()
    original, calls = torch.linalg.cholesky, []
    def counted(value):
        calls.append(value.shape)
        return original(value)
    monkeypatch.setattr(torch.linalg, "cholesky", counted)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        item["unit_covariance"].grad = None
        output = model(**item)
        output.probabilities[..., 0].sum().backward()
        assert torch.isfinite(item["unit_covariance"].grad).all()
        assert item["unit_covariance"].grad.abs().sum() > 0
    assert len(calls) == 2


def test_direct_covariance_diagonal_matches_factor_reconstruction():
    model = ICRMMSEnet(layers=3, bandwidth=2, window_size=8, center_size=4, precision="complex128")
    item = inputs(12, dtype=torch.complex128)
    direct = model(**item)
    def reconstruct_diagonal(module, args, kwargs):
        kwargs["covariance_diagonal"] = None
        return args, kwargs
    hooks = [layer.register_forward_pre_hook(reconstruct_diagonal, with_kwargs=True)
             for layer in model.equalizers]
    reconstructed = model(**item)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(direct.probabilities, reconstructed.probabilities, rtol=1e-10, atol=1e-12)


def test_diagonal_and_full_noise_agree_for_diagonal_covariance_and_gradients():
    item = inputs(12, dtype=torch.complex128)
    full = ICRMMSEnet(layers=3, bandwidth=2, window_size=8, center_size=4, precision="complex128")
    with torch.no_grad():
        full.corrector.head.weight[:10].normal_(std=.02)
        for layer in full.equalizers:
            layer.classifier.weight.normal_(std=.01)
    diagonal = ICRMMSEnet(**{**full.config, "noise_mode": "diagonal"})
    diagonal.load_state_dict(full.state_dict())
    outputs, gradients = [], []
    for model in (full, diagonal):
        local = {**item}
        differentiable = []
        for name in ("observations", "initial_channel", "rf_variance"):
            local[name] = item[name].detach().clone().requires_grad_()
            differentiable.append(local[name])
        noise_diagonal = torch.linspace(.7, 1.3, 12, dtype=torch.float64, requires_grad=True)
        local["unit_covariance"] = torch.diag(noise_diagonal).to(torch.complex128)
        output = model(**local)
        loss = (output.probabilities[..., 0].sum() + .03 * output.symbols.abs().square().sum()
                + .01 * output.bands.abs().square().sum())
        loss.backward()
        outputs.append(output)
        gradients.append([value.grad for value in (*differentiable, noise_diagonal, *model.parameters())])
    for name in ("probabilities", "symbols", "variances", "bands"):
        torch.testing.assert_close(getattr(outputs[0], name), getattr(outputs[1], name), rtol=1e-10, atol=1e-12)
    for full_gradient, diagonal_gradient in zip(*gradients):
        assert full_gradient is not None and diagonal_gradient is not None
        torch.testing.assert_close(full_gradient, diagonal_gradient, rtol=1e-9, atol=1e-11)


def test_diagonal_noise_bypasses_cholesky_and_preserves_frequency_power(monkeypatch):
    item = inputs(12)
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4, noise_mode="diagonal")
    def forbidden(*args, **kwargs):
        raise AssertionError("Diagonal noise weighting must not use Cholesky")
    monkeypatch.setattr(torch.linalg, "cholesky", forbidden)
    monkeypatch.setattr(torch, "cholesky_solve", forbidden)
    output = model(**item)
    cancelled = item["observations"] - (item["initial_channel"] @ item["known"][..., None]).squeeze(-1)
    expected = item["initial_channel"].mH @ (cancelled / item["unit_covariance"].diagonal().real)[..., None]
    torch.testing.assert_close(output.diagnostics[0]["matched"], expected.squeeze(-1) * item["data_mask"])
    assert model._noise_factor_cache is None
    output.probabilities[..., 0].sum().backward()
    assert torch.isfinite(model.equalizers[0].step_logit.grad)


def test_only_full_noise_weighting_uses_off_diagonal_covariance():
    item = inputs(12)
    covariance = item["unit_covariance"]
    changed = {**item, "unit_covariance": .5 * covariance + .5 * torch.diag(covariance.diagonal())}
    full = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4, noise_mode="full")
    diagonal = ICRMMSEnet(**{**full.config, "noise_mode": "diagonal"})
    diagonal.load_state_dict(full.state_dict())
    first, second = full(**item), full(**changed)
    assert not torch.allclose(first.probabilities, second.probabilities)
    first, second = diagonal(**item), diagonal(**changed)
    torch.testing.assert_close(first.probabilities, second.probabilities, atol=0, rtol=0)
    torch.testing.assert_close(first.bands, second.bands, atol=0, rtol=0)


@pytest.mark.parametrize("noise_mode", ["white", "", None, True])
def test_noise_mode_validation_and_legacy_default(noise_mode):
    with pytest.raises(ValueError, match="controls"):
        ICRMMSEnet(noise_mode=noise_mode)
    assert ICRMMSEnet().config["noise_mode"] == "full"


@pytest.mark.parametrize("precision,complex_dtype,real_dtype", [
    ("complex64", torch.complex64, torch.float32), ("complex128", torch.complex128, torch.float64)])
def test_precision_controls_inputs_model_state_and_gradients(precision, complex_dtype, real_dtype):
    model = ICRMMSEnet(layers=2, bandwidth=2, window_size=8, center_size=4, precision=precision)
    assert model.config["precision"] == precision
    assert ICRMMSEnet().config["precision"] == "complex64"
    assert model.constellation.dtype == complex_dtype
    assert all(parameter.dtype == real_dtype for parameter in model.parameters())
    result = model(**inputs(12, dtype=torch.complex128))
    assert result.initial_channel.dtype == result.symbols.dtype == complex_dtype
    assert result.probabilities.dtype == result.variances.dtype == real_dtype
    assert model._noise_factor_cache[1].dtype == complex_dtype
    torch.testing.assert_close(result.probabilities.sum(-1),
                               inputs(12)["data_mask"].to(real_dtype)[None, None].expand(2, 2, -1))
    result.probabilities[..., 0].sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
