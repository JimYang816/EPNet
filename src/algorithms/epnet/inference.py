"""Label-free adapter to the existing full-ICI comparison likelihood."""
from __future__ import annotations

import numpy as np
import torch

from src.algorithms.interface import EqualizerResult
from src.algorithms.likelihood import batch_real_likelihood


def equalize_batch(batch, backend, *, whitener, model):
    gram, matched = batch_real_likelihood(batch, whitener, backend)
    with torch.no_grad():
        result = model(torch.as_tensor(gram, dtype=model.beta.dtype, device=model.beta.device),
                       torch.as_tensor(matched, dtype=model.beta.dtype, device=model.beta.device))
    return EqualizerResult(result.symbols.cpu().numpy(), result.rejected_sites.cpu().numpy())
