"""Exact inference preparation for frozen residual-unfolding parameters.

Training/checkpoint modules stay unchanged. The prepared model is for inference
only: activated scalar parameters are cached and exactly zero classifier heads
are omitted. No input-dependent computation or learned weight is approximated.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from src.algorithms.icr_mmse.model import BudgetedMMSELayer, _real_pair


class FrozenBudgetedMMSELayer(nn.Module):
    def __init__(self, layer, forward_cache=None):
        super().__init__()
        self.original = layer
        self._forward_cache = forward_cache
        self.register_buffer("step_multiplier", 2*layer.step_logit.sigmoid())
        self.register_buffer("loading_coefficient", functional.softplus(layer.csi_loading_log))
        self.register_buffer("uncertainty_coefficient", functional.softplus(layer.uncertainty_log))
        self.register_buffer("temperature_multiplier", layer.temperature_log.exp().clamp(.01, 100))
        self.register_buffer("log_damping", None if layer.soft_damping_logit is None else
                             functional.logsigmoid(layer.soft_damping_logit))
        self.register_buffer("log_previous_damping", None if layer.soft_damping_logit is None else
                             functional.logsigmoid(-layer.soft_damping_logit))
        self.zero_classifier = all(torch.count_nonzero(p).item() == 0 for p in layer.classifier.parameters())
        self._window_geometry = None

    def _window_logits(self, features):
        batch, size, _ = features.shape
        layer = self.original
        count = (size+layer.center_size-1)//layer.center_size
        key = (size, features.device)
        if self._window_geometry is None or self._window_geometry[0] != key:
            left = (layer.window_size-layer.center_size)//2
            starts = torch.arange(count, device=features.device)[:, None]*layer.center_size-left
            positions = starts+torch.arange(layer.window_size, device=features.device)
            valid = ((positions >= 0) & (positions < size))[None, ..., None]
            self._window_geometry = (key, positions.clamp(0, size-1), valid)
        _, indices, valid = self._window_geometry
        windows = features[:, indices]*valid
        hidden = torch.tanh(layer.hidden(windows.flatten(-2).to(layer.hidden.weight.dtype)))
        centers = layer.classifier(hidden).reshape(batch, count*layer.center_size, 4)
        return centers[:, :size]

    @property
    def csi_loading_log(self):
        return self.original.csi_loading_log

    def forward(self, channel, observations, known, mean, variance, data_mask, pilot_mask,
                null_mask, positions, rf_variance, covariance_factor, constellation,
                use_uncertainty=True, covariance_diagonal=None, column_weights=None,
                base_column_energy=None, return_diagnostics=True,
                previous_log_probabilities=None, return_log_probabilities=False):
        if self.training:
            raise RuntimeError("Prepared unfolding layers cannot be trained")
        if covariance_diagonal is None:
            if covariance_factor is None:
                raise ValueError("Diagonal noise requires a covariance diagonal")
            covariance_diagonal = covariance_factor.abs().square().sum(-1)
        solve_cov = ((lambda value: value/covariance_diagonal) if covariance_factor is None else
                     (lambda value: torch.cholesky_solve(value[..., None], covariance_factor).squeeze(-1)))
        weights = covariance_diagonal.reciprocal() if column_weights is None else column_weights
        diagonal = channel.column_energy(weights, base_energy=base_column_energy).clamp_min(1e-20)
        shared = self._forward_cache if self._forward_cache is not None else {}
        if "count" not in shared:
            shared["count"] = data_mask.sum().clamp_min(1)
        count = shared["count"]
        energy = (diagonal*data_mask).sum(-1)/count
        loading = self.loading_coefficient*energy
        regularizer = rf_variance+loading
        denominator = diagonal+regularizer[:, None]
        data_mean = mean*data_mask
        residual = observations-channel.matvec(data_mean+known)
        weighted_residual = solve_cov(residual)
        matched_residual = channel.adjoint(weighted_residual)*data_mask
        gradient = matched_residual-regularizer[:, None]*data_mean
        direction = gradient/denominator
        direction_prediction = channel.matvec(direction)
        numerator = (gradient.conj()*direction).real.sum(-1).clamp_min(0)
        curvature = (direction_prediction.conj()*solve_cov(direction_prediction)).real.sum(-1)
        curvature = curvature+regularizer*direction.abs().square().sum(-1)
        step = self.step_multiplier*numerator/curvature.clamp_min(1e-20)
        estimate = data_mean+step[:, None]*direction
        feedback_variance = variance if use_uncertainty else torch.zeros_like(variance)
        uncertainty = (feedback_variance*data_mask).sum(-1)/count
        tau = (regularizer[:, None]/diagonal+self.uncertainty_coefficient*uncertainty[:, None]
               ).clamp_min(1e-5)*self.temperature_multiplier
        logits = -(estimate[..., None]-constellation).abs().square()/tau[..., None]
        if not self.zero_classifier:
            if "masks" not in shared:
                shared["masks"] = torch.stack((data_mask, pilot_mask, null_mask), -1).to(variance.dtype)
                shared["coordinates"] = (positions.to(variance.dtype)-positions.float().mean())/max(len(positions), 1)
            masks, coordinates = shared["masks"], shared["coordinates"]
            features = torch.cat((_real_pair(data_mean), _real_pair(estimate),
                _real_pair(matched_residual/denominator), _real_pair(direction),
                feedback_variance[..., None], torch.log1p(rf_variance[:, None]/diagonal)[..., None],
                masks[None].expand(len(mean), -1, -1), coordinates[None, :, None].expand(len(mean), -1, -1)), -1)
            logits = logits+self._window_logits(features).to(logits.dtype)
        logp = logits.log_softmax(-1)
        if self.log_damping is not None:
            if previous_log_probabilities is None:
                raise ValueError("Soft-state damping requires the previous QPSK distribution")
            logp = torch.logaddexp(logp+self.log_damping, previous_log_probabilities+self.log_previous_damping)
        probabilities = logp.exp()*data_mask[None, :, None]
        next_mean = (probabilities*constellation).sum(-1)+known
        next_variance = (1-next_mean.abs().square()).clamp_min(0)*data_mask
        detail = {}
        if return_diagnostics:
            before = ((residual.conj()*weighted_residual).real.sum(-1)
                      +regularizer*data_mean.abs().square().sum(-1))
            after_residual = residual-step[:, None]*direction_prediction
            after = ((after_residual.conj()*solve_cov(after_residual)).real.sum(-1)
                     +regularizer*estimate.abs().square().sum(-1))
            detail = dict(residual=residual, linear_step=step, csi_loading=loading,
                          linear_objective_before=before, linear_objective_after=after,
                          state_change_energy=((next_mean-mean)*data_mask).abs().square().mean(-1))
        result = probabilities, next_mean, next_variance, detail
        return (*result, logp) if return_log_probabilities else result


def prepare_frozen_inference(model):
    """Prepare an already frozen model in place; do not save it as a checkpoint.

    Call after load_model and device/dtype selection. Loading the original
    checkpoint afresh always restores the trainable representation.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Inference preparation requires eval mode and frozen weights")
    if model.config["detector_mode"] == "budgeted":
        if not hasattr(model, "_budgeted_forward_cache"):
            model._budgeted_forward_cache = {}
            model.register_forward_pre_hook(lambda owner, args: owner._budgeted_forward_cache.clear())
        model.equalizers = nn.ModuleList(FrozenBudgetedMMSELayer(layer, model._budgeted_forward_cache).eval()
            if type(layer) is BudgetedMMSELayer else layer for layer in model.equalizers)
    return model
