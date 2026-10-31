"""Validation metrics and deterministic training-only numerical health probes."""
from __future__ import annotations

import numpy as np
import torch

from src.data.udnet import FEATURES
from src.training.udnet_supervision import multilayer_cross_entropy


class ValidationMetrics:
    """Feature cache order is SNR, D, frame, window; center slots are 8:24."""

    def __init__(self, snr_db, count, layers, device):
        self.snr_db = list(map(float, snr_db))
        if not self.snr_db or count % len(self.snr_db):
            raise ValueError("Validation cache must have equal coverage per SNR")
        self.windows_per_snr = count//len(self.snr_db)
        self.ce = torch.zeros((len(self.snr_db), layers), dtype=torch.float64, device=device)
        self.counts = torch.zeros((len(self.snr_db), 3), dtype=torch.float64, device=device)

    @torch.no_grad()
    def observe(self, indices, probabilities, batch):
        mask, labels = batch["data_mask"], batch["labels"].long()
        selected = probabilities.gather(-1, labels.clamp_min(0)[:, None, :, None].expand(-1, probabilities.shape[1], -1, 1)).squeeze(-1)
        ce = -(selected.clamp_min(torch.finfo(selected.dtype).tiny).log()*mask[:, None, :]).sum(-1)
        classes = probabilities[:, -1, 8:24].argmax(-1)
        target, central = labels[:, 8:24], mask[:, 8:24]
        errors = (classes//2 != target//2).long()+(classes%2 != target%2).long()
        counts = torch.stack((mask.sum(-1), (errors*central).sum(-1), 2*central.sum(-1)), -1)
        groups = torch.as_tensor(indices//self.windows_per_snr, device=self.ce.device)
        self.ce.index_add_(0, groups, ce.double())
        self.counts.index_add_(0, groups, counts.double())

    def result(self):
        ce, counts = self.ce.cpu().numpy(), self.counts.cpu().numpy()
        if np.any(counts[:, 0] == 0) or np.any(counts[:, 2] == 0):
            raise ValueError("Incomplete validation diagnostics")
        per_layer = ce/counts[:, :1]
        rows = [dict(snr_db=snr, loss=float(layer.sum()), mean_layer_ce=float(layer.mean()),
                     last_layer_ce=float(layer[-1]), layer_ce=layer.tolist(),
                     central_ber=float(count[1]/count[2]), central_bits=int(count[2]),
                     valid_window_labels=int(count[0]))
                for snr, layer, count in zip(self.snr_db, per_layer, counts)]
        overall = ce.sum(0)/counts[:, 0].sum()
        return dict(per_snr=rows, mean_layer_ce=float(overall.mean()),
                    last_layer_ce=float(overall[-1]), layer_ce=overall.tolist(),
                    central_ber=float(counts[:, 1].sum()/counts[:, 2].sum()))


def training_health(model, cache, device, sample_size=256):
    """Probe fixed training windows without optimizer/RNG/parameter updates."""
    indices = np.linspace(0, cache.count-1, min(sample_size, cache.count), dtype=np.int64)
    batch = cache.batch(indices, device)
    activations, handles = [], []
    was_training = model.training
    try:
        model.eval()
        for layer in model.hidden:
            handles.append(layer.register_forward_hook(
                lambda module, args, output: activations.append(output.detach())))
        with torch.enable_grad():
            probabilities = model(*(batch[name] for name in FEATURES))
            loss = multilayer_cross_entropy(probabilities, batch["labels"], batch["data_mask"])
            gradients = torch.autograd.grad(loss, [layer.weight for layer in model.hidden])
        rows = [dict(layer=i+1, saturated_fraction=float((a.abs() > 3).float().mean()),
                     mean_tanh_derivative=float((1-a.tanh().square()).mean()),
                     weight_gradient_norm=float(g.norm()))
                for i, (a, g) in enumerate(zip(activations, gradients))]
        if any(not np.isfinite(row["weight_gradient_norm"]) for row in rows):
            raise FloatingPointError("Nonfinite UDNet diagnostic gradient")
        return dict(windows=len(indices), split="train", layers=rows,
                    zero_gradient_layers=[row["layer"] for row in rows if row["weight_gradient_norm"] == 0])
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
