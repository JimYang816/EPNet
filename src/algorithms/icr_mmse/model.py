"""ICR-MMSE-Net: shared local CSI repair and independent MMSE unfolding.

The linear substep is a learned diagonally preconditioned Richardson step for
the colored-noise LMMSE normal equations, followed by a QPSK denoiser and an
independent local logit calibrator. It approximates the LMMSE solve; it is not
a closed-form LMMSE inverse. Every normal action retains the full channel.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as functional

from .operators import BandChannel, band_indices
from .lmmse import FullMatrixLMMSELayer
from .dpss import DPSSChannel, base_energy, base_local, reference_energy


FEEDBACK_LAYOUT = "aligned-band14-row2-noise3-v1"
# The final axis is per-offset. These offsets never refer to compressed data indices.
FEEDBACK_SLICES = dict(initial=slice(0, 2), current=slice(2, 4),
                       mean=slice(4, 6), variance=slice(6, 7), residual_mean=slice(7, 9),
                       channel_variance=slice(9, 11), data=slice(11, 12),
                       pilot=slice(12, 13), valid=slice(13, 14))


@dataclass
class CSIFeedback:
    offsets: torch.Tensor  # [batch, receive row, physical offset, 14]
    residual: torch.Tensor  # [batch, receive row, Re/Im], shared across offsets


def _real_pair(value):
    return torch.stack((value.real, value.imag), dim=-1)


def build_feedback_features(initial_channel, channel, observations, mean, variance,
                            data_mask, pilot_mask, null_mask, *, use_uncertainty=True):
    """Literal F[k,d] including conjugate products, zero padding, validity.

    Returns 14 real features per offset and one complex residual per row.
    The removed null mask is exactly valid-data-pilot, including at boundaries.
    No labels or references are accepted; physical products remain explicit.
    """
    batch, size, _ = initial_channel.shape
    columns, valid = band_indices(size, channel.bandwidth, initial_channel.device)
    residual = observations - channel.matvec(mean)
    h0, ht = base_local(initial_channel, channel.bandwidth), channel.local_entries()
    xj = mean[:, columns]
    vj = variance[:, columns] if use_uncertainty else torch.zeros_like(variance[:, columns])
    rk = residual[..., None]
    masks = [mask[columns].to(variance.dtype)[None].expand(batch, -1, -1)[..., None]
             for mask in (data_mask, pilot_mask)]
    features = torch.cat((_real_pair(h0), _real_pair(ht), _real_pair(xj),
                          vj[..., None], _real_pair(rk * xj.conj()), _real_pair(ht * vj),
                          *masks, valid.to(variance.dtype)[None, ..., None].expand(batch, -1, -1, -1)), -1)
    return CSIFeedback(features * valid[None, ..., None], _real_pair(residual))


class CSICorrector(nn.Module):
    """Exactly two k=3 convolutions and one k=1 output (receptive field five)."""

    def __init__(self, bandwidth):
        super().__init__()
        self.width = 2 * bandwidth + 1
        self.conv1 = nn.Conv1d(14 * self.width + 5, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.head = nn.Conv1d(32, 3 * self.width, 1)
        # Zero ONLY the complex correction channels. Gates start finite and learnable.
        with torch.no_grad():
            self.head.weight[:2 * self.width].zero_()
            self.head.bias[:2 * self.width].zero_()

    def forward(self, feedback, noise_features):
        features = torch.cat((feedback.offsets.flatten(-2), feedback.residual, noise_features), -1).transpose(1, 2)
        features = features.to(self.conv1.weight.dtype)
        hidden = functional.silu(self.conv1(features))
        output = self.head(functional.silu(self.conv2(hidden))).transpose(1, 2)
        real, imaginary, gate = output.split(self.width, dim=-1)
        return torch.complex(real, imaginary), gate.sigmoid()


class MMSEUnfoldedLayer(nn.Module):
    """Global linear step, then local windows whose centers cover every row once."""

    def __init__(self, window_size=32, center_size=16, hidden_size=48):
        super().__init__()
        self.window_size, self.center_size = window_size, center_size
        self.step_logit = nn.Parameter(torch.zeros(()))
        self.temperature_log = nn.Parameter(torch.zeros(()))
        self.uncertainty_log = nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.hidden = nn.Linear(14 * window_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 4 * center_size)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _window_logits(self, features):
        batch, size, _ = features.shape
        count = (size + self.center_size - 1) // self.center_size
        left = (self.window_size - self.center_size) // 2
        starts = torch.arange(count, device=features.device)[:, None] * self.center_size - left
        positions = starts + torch.arange(self.window_size, device=features.device)
        valid = (positions >= 0) & (positions < size)
        windows = features[:, positions.clamp(0, size - 1)] * valid[None, ..., None]
        hidden = torch.tanh(self.hidden(windows.flatten(-2).to(self.hidden.weight.dtype)))
        centers = self.classifier(hidden).reshape(batch, count * self.center_size, 4)
        return centers[:, :size]

    def forward(self, channel, observations, known, mean, variance, data_mask, pilot_mask,
                null_mask, positions, rf_variance, covariance_factor, constellation,
                use_uncertainty=True, covariance_diagonal=None, column_weights=None,
                base_column_energy=None, return_diagnostics=True):
        # H changed: reconstruct pilot cancellation AND both complete normal actions.
        cancelled = observations - channel.matvec(known)
        if covariance_diagonal is None:
            if covariance_factor is None:
                raise ValueError("Diagonal noise requires a covariance diagonal")
            covariance_diagonal = (covariance_factor.abs().square()).sum(-1)
        if covariance_factor is None:
            solve_cov = lambda value: value / covariance_diagonal
        else:
            solve_cov = lambda value: torch.cholesky_solve(value[..., None], covariance_factor).squeeze(-1)
        data_mean = mean * data_mask
        matched = channel.adjoint(solve_cov(cancelled)) * data_mask
        normal = channel.adjoint(solve_cov(channel.matvec(data_mean))) * data_mask
        weights = covariance_diagonal.reciprocal() if column_weights is None else column_weights
        diagonal = channel.column_energy(weights, base_energy=base_column_energy).clamp_min(1e-20)
        denominator = (diagonal + rf_variance[:, None]).clamp_min(1e-20)
        step = 2 * self.step_logit.sigmoid()
        estimate = data_mean + step * (matched - normal - rf_variance[:, None] * data_mean) / denominator
        feedback_variance = variance if use_uncertainty else torch.zeros_like(variance)
        uncertainty = (feedback_variance * data_mask).sum(-1) / data_mask.sum().clamp_min(1)
        tau = (rf_variance[:, None] / diagonal + functional.softplus(self.uncertainty_log)
               * uncertainty[:, None]).clamp_min(1e-5) * self.temperature_log.exp().clamp(.01, 100)
        masks = torch.stack((data_mask, pilot_mask, null_mask), -1).to(variance.dtype)
        coordinates = (positions.to(variance.dtype) - positions.float().mean()) / max(len(positions), 1)
        features = torch.cat((_real_pair(data_mean), _real_pair(estimate),
                              _real_pair(matched / denominator), _real_pair(normal / denominator),
                              feedback_variance[..., None], torch.log1p(rf_variance[:, None] / diagonal)[..., None],
                              masks[None].expand(len(mean), -1, -1),
                              coordinates[None, :, None].expand(len(mean), -1, -1)), -1)
        local_logits = self._window_logits(features)
        logits = -(estimate[..., None] - constellation).abs().square() / tau[..., None]
        probabilities = (logits + local_logits.to(logits.dtype)).softmax(-1) * data_mask[None, :, None]
        next_mean = (probabilities * constellation).sum(-1) + known
        next_variance = (1 - next_mean.abs().square()).clamp_min(0) * data_mask
        diagnostics = (dict(pilot_cancelled=cancelled, matched=matched, normal_action=normal,
                            residual=observations - channel.matvec(mean)) if return_diagnostics else {})
        return probabilities, next_mean, next_variance, diagnostics


@dataclass
class ICRMMSEOutput:
    probabilities: torch.Tensor
    symbols: torch.Tensor
    variances: torch.Tensor
    channels: tuple[BandChannel, ...]
    diagnostics: tuple[dict, ...]

    @property
    def bands(self):
        return torch.stack([channel.bands for channel in self.channels], dim=1)

    @property
    def initial_channel(self):
        return self.channels[0].base


class ICRMMSEnet(nn.Module):
    """Shared CSI repair and T MMSE layers with diagonal or full noise weighting.

    Missing noise_mode in historical checkpoints retains full covariance weighting.
    """

    def __init__(self, layers=8, bandwidth=16, initial_damping=.1, window_size=32,
                 center_size=16, hidden_size=48, correction_mode="iterative",
                 use_uncertainty=True, use_gate=True, correction_passes=1, equalizer_repeats=1,
                 precision="complex64", noise_mode="full", detector_mode="unfolded"):
        super().__init__()
        if (not isinstance(layers, int) or isinstance(layers, bool) or layers < 1
                or not isinstance(bandwidth, int) or isinstance(bandwidth, bool) or bandwidth < 0
                or not math.isfinite(initial_damping) or not 0 < initial_damping < 1
                or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0
                           for x in (window_size, center_size, hidden_size, correction_passes, equalizer_repeats))
                or center_size > window_size or correction_mode not in ("iterative", "once", "fixed")
                or not isinstance(use_uncertainty, bool) or not isinstance(use_gate, bool)
                or precision not in ("complex64", "complex128")
                or noise_mode not in ("diagonal", "full")
                or detector_mode not in ("unfolded", "lmmse")
                or (correction_mode != "once" and correction_passes != 1)):
            raise ValueError("Invalid ICR-MMSE-Net controls")
        self.config = dict(layers=layers, bandwidth=bandwidth, initial_damping=initial_damping,
                           window_size=window_size, center_size=center_size, hidden_size=hidden_size,
                           correction_mode=correction_mode, use_uncertainty=use_uncertainty,
                           use_gate=use_gate, correction_passes=correction_passes,
                           equalizer_repeats=equalizer_repeats, precision=precision,
                           noise_mode=noise_mode, detector_mode=detector_mode)
        real_dtype = torch.float32 if precision == "complex64" else torch.float64
        complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
        self.corrector = CSICorrector(bandwidth).to(dtype=real_dtype)
        self.eta_logits = nn.Parameter(torch.full((layers,), math.log(initial_damping / (1 - initial_damping)),
                                                 dtype=real_dtype))
        self.equalizers = nn.ModuleList(
            MMSEUnfoldedLayer(window_size, center_size, hidden_size) if detector_mode == "unfolded"
            else FullMatrixLMMSELayer() for _ in range(layers)).to(dtype=real_dtype)
        self.register_buffer("constellation", torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                                                            dtype=complex_dtype) / math.sqrt(2))
        # Runtime-only single-entry cache: never serialized into checkpoints.
        self._noise_factor_cache = None

    def _apply(self, fn, recurse=True):
        self._noise_factor_cache = None
        return super()._apply(fn, recurse=recurse)

    def _noise_factor(self, covariance, dtype, device):
        covariance = covariance.to(device=device, dtype=dtype)
        if torch.is_grad_enabled() and covariance.requires_grad:
            # A graph-bearing factor cannot be reused across optimizer steps.
            return torch.linalg.cholesky(covariance)
        cached = self._noise_factor_cache
        if (cached is not None and cached[0].shape == covariance.shape
                and cached[0].dtype == covariance.dtype and cached[0].device == covariance.device
                and torch.equal(cached[0], covariance)):
            return cached[1]
        # Compare contents, including for inference tensors and shared NumPy storage.
        # Ordinary tensors also allow an inference-created cache to serve training.
        with torch.inference_mode(False), torch.no_grad():
            snapshot = covariance.detach().clone()
            factor = torch.linalg.cholesky(snapshot)
        self._noise_factor_cache = (snapshot, factor)
        return factor

    @property
    def damping(self):
        return self.eta_logits.sigmoid()

    def forward(self, observations, initial_channel, known, data_mask, pilot_mask, null_mask,
                carrier_positions, rf_variance, unit_covariance, *, correction_enabled=True,
                detach_feedback=False, return_diagnostics=True, reuse_base_energy=True):
        if (initial_channel.ndim != 3 or initial_channel.shape[-1] != initial_channel.shape[-2]
                or observations.shape != initial_channel.shape[:2] or not initial_channel.is_complex()
                or observations.dtype != initial_channel.dtype or observations.device != initial_channel.device):
            raise ValueError("Expected matching complex batched observations and full square initial CSI")
        initial_channel = initial_channel.to(dtype=self.constellation.dtype)
        if isinstance(initial_channel, DPSSChannel):
            initial_channel.clear_sample_cache()
        observations = observations.to(dtype=self.constellation.dtype)
        rf_variance = rf_variance.to(device=initial_channel.device, dtype=observations.real.dtype)
        batch, size = observations.shape
        for mask in (data_mask, pilot_mask, null_mask):
            if mask.shape != (size,) or mask.dtype != torch.bool or mask.device != observations.device:
                raise ValueError("Masks must be boolean vectors on the input device")
        if not torch.all(data_mask.to(torch.int8) + pilot_mask + null_mask == 1):
            raise ValueError("Data, pilot and null masks must form a disjoint full physical grid")
        if (carrier_positions.shape != (size,) or unit_covariance.shape != (size, size)
                or rf_variance.shape != (batch,) or not torch.isfinite(rf_variance).all()
                or torch.any(rf_variance < 0)
                or not (initial_channel.finite() if isinstance(initial_channel, DPSSChannel)
                        else torch.isfinite(initial_channel).all())
                or not torch.isfinite(observations).all()):
            raise ValueError("Invalid physical carrier/noise/channel inputs")
        if known.shape == (size,):
            known = known[None].expand(batch, -1)
        if known.shape != observations.shape or not torch.isfinite(known).all():
            raise ValueError("Known symbols must have shape [N] or [batch,N]")
        # Symbols supplied outside pilot slots cannot change a prediction.
        known = known.to(device=initial_channel.device, dtype=initial_channel.dtype) * pilot_mask
        real_dtype = observations.real.dtype
        diagonal = unit_covariance.diagonal().real.to(device=initial_channel.device, dtype=real_dtype).clamp_min(1e-20)
        column_weights = diagonal.reciprocal()
        shared_energy = None
        if reuse_base_energy:
            shared_energy = (initial_channel.column_energy(column_weights) if isinstance(initial_channel, DPSSChannel)
                             else (initial_channel.abs().square() * column_weights[..., None]).sum(-2))
        covariance_factor = None
        if self.config["noise_mode"] == "full":
            covariance = unit_covariance.to(device=initial_channel.device, dtype=initial_channel.dtype)
            covariance_factor = self._noise_factor(covariance, initial_channel.dtype, initial_channel.device)
            covariance_factor = covariance_factor[None].expand(batch, -1, -1)
        scale = ((base_energy(initial_channel) / size).sqrt() if isinstance(initial_channel, DPSSChannel)
                 else initial_channel.abs().square().sum(-1).mean(-1).sqrt()).clamp_min(1e-12)
        entry_scale = scale / math.sqrt(size)
        normalized_initial = (initial_channel.scaled(scale) if isinstance(initial_channel, DPSSChannel)
                              else initial_channel / scale[:, None, None])
        normalized_y = observations / scale[:, None]
        normalized_noise = rf_variance / scale.square()
        noise_features = torch.stack((torch.log1p(normalized_noise)[:, None].expand(-1, size),
                                      torch.log(diagonal)[None].expand(batch, -1),
                                      (carrier_positions.to(real_dtype) / max(size, 1))[None].expand(batch, -1)), -1)
        bands = torch.zeros(batch, size, 2 * self.config["bandwidth"] + 1,
                            dtype=initial_channel.dtype, device=initial_channel.device)
        # Uniform QPSK data prior. Known pilots and nulls have deterministic variance zero.
        mean = known
        variance = data_mask.to(real_dtype)[None].expand(batch, -1)
        probabilities, states, variances, channels, diagnostics = [], [], [], [], []
        enabled = correction_enabled and self.config["correction_mode"] != "fixed"
        for index, equalizer in enumerate(self.equalizers):
            feedback_mean = mean.detach() if detach_feedback else mean
            feedback_variance = variance.detach() if detach_feedback else variance
            should_correct = enabled and (self.config["correction_mode"] == "iterative" or index == 0)
            passes = self.config["correction_passes"] if index == 0 else 1
            if should_correct:
                for correction_pass in range(passes):
                    normalized_channel = BandChannel(normalized_initial, bands / scale[:, None, None])
                    feedback = build_feedback_features(normalized_initial, normalized_channel, normalized_y,
                        feedback_mean, feedback_variance, data_mask, pilot_mask, null_mask,
                        use_uncertainty=self.config["use_uncertainty"])
                    increment, gate = self.corrector(feedback, noise_features)
                    if not self.config["use_gate"]:
                        gate = torch.ones_like(gate)
                    _, valid = band_indices(size, self.config["bandwidth"], initial_channel.device)
                    eta_index = min(index + correction_pass, len(self.equalizers) - 1)
                    bands = bands + self.damping[eta_index] * (increment * gate).to(initial_channel.dtype) * entry_scale[:, None, None] * valid
            channel = BandChannel(initial_channel, bands)
            # Extra effective detector steps support an empirically timed one-repair
            # control. They share this layer's parameters and retain the same H.
            for repeat in range(self.config["equalizer_repeats"]):
                if repeat:
                    feedback_mean = mean.detach() if detach_feedback else mean
                    feedback_variance = variance.detach() if detach_feedback else variance
                # During B no gradient can bypass the stop through an earlier decision.
                p, mean, variance, detail = equalizer(channel, observations, known, feedback_mean,
                    feedback_variance, data_mask, pilot_mask, null_mask, carrier_positions, rf_variance,
                    covariance_factor, self.constellation.to(initial_channel.dtype),
                    use_uncertainty=self.config["use_uncertainty"], covariance_diagonal=diagonal,
                    column_weights=column_weights, base_column_energy=shared_energy,
                    return_diagnostics=return_diagnostics)
            probabilities.append(p)
            states.append(mean)
            variances.append(variance)
            channels.append(channel)
            diagnostics.append(detail)
            if self.config["detector_mode"] == "lmmse" and not enabled:
                # The same fixed channel and unit prior give the same solution.
                # Charge only the single useful solve, including during warmup.
                break
        return ICRMMSEOutput(torch.stack(probabilities, 1), torch.stack(states, 1),
                             torch.stack(variances, 1), tuple(channels), tuple(diagnostics))


ICRMMSE = ICRMMSEnet


def csi_nmse(output, reference_channel):
    denominator = reference_energy(reference_channel).clamp_min(1e-30)
    return torch.stack([channel.error_energy(reference_channel) / denominator
                        for channel in output.channels], dim=1)


def layerwise_loss(output, classes, reference_channel, data_mask, lambda_h=.1, *, csi_only=False):
    """Independent supervision entry point; averaged over layers and examples."""
    if not math.isfinite(lambda_h) or lambda_h < 0:
        raise ValueError("lambda_h must be finite and nonnegative")
    if csi_only:
        return csi_nmse(output, reference_channel).mean()
    probabilities = output.probabilities[:, :, data_mask]
    if classes.shape != (len(probabilities), probabilities.shape[-2]):
        raise ValueError("Supervised classes must cover data carriers in physical order")
    selected = probabilities.gather(-1, classes[:, None, :, None].expand(-1, probabilities.shape[1], -1, 1))
    ce = -selected.clamp_min(torch.finfo(selected.dtype).tiny).log().mean()
    return ce if lambda_h == 0 else ce + lambda_h * csi_nmse(output, reference_channel).mean()
