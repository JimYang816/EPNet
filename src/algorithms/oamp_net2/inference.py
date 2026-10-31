"""Label-free OAMP-Net2 batch adapter with no checkpoint fallback."""
from __future__ import annotations

import numpy as np
import torch

from src.algorithms.interface import EqualizerResult
from src.algorithms.oamp_net2.model import FEATURES
from src.data.oamp_net2 import spectral_features


@torch.no_grad()
def equalize_batch(batch, backend, *, model, whitener):
    features = spectral_features(batch, whitener, backend)
    device = next(model.parameters()).device
    inputs = [torch.as_tensor(features[name], device=device) for name in FEATURES]
    symbols = model(*inputs).cpu().numpy()
    if not np.isfinite(symbols).all():
        raise FloatingPointError("Nonfinite OAMP-Net2 prediction")
    return EqualizerResult(symbols, np.zeros(len(batch.variances), dtype=np.int64))
