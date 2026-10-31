"""The only UDNet training routines that read target bits."""
from __future__ import annotations

import numpy as np
import torch


def window_labels(dataset, frames, d, plan):
    bits = dataset._h["samples/bits"][frames, d]
    classes = (2*bits[..., 0]+bits[..., 1]).astype(np.int64)
    full = np.full((len(frames), plan.nfft), -1, dtype=np.int64)
    full[:, plan.data_indices] = classes
    labels = full[:, np.clip(plan.positions, 0, plan.nfft-1)]
    labels[:, ~plan.data_mask] = -1
    return labels.reshape(-1, 32).astype(np.int8)


def multilayer_cross_entropy(probabilities, labels, data_mask):
    """Sum ALL layer CEs; divide once by the number of valid window labels."""
    if labels.shape != data_mask.shape or probabilities.shape[0] != len(labels):
        raise ValueError("UDNet supervision shape mismatch")
    valid = data_mask & (labels >= 0)
    if not torch.equal(valid, data_mask) or torch.any(labels[valid] > 3) or not valid.any():
        raise ValueError("Missing/invalid labels on data slots")
    indices = labels.clamp_min(0).long()[:, None, :, None].expand(-1, probabilities.shape[1], -1, 1)
    selected = probabilities.gather(-1, indices).squeeze(-1)
    return -(selected.clamp_min(torch.finfo(selected.dtype).tiny).log()*valid[:, None, :]).sum()/valid.sum()
