"""Stream compact-v2 D1/D2 likelihood batches; supervision is read separately."""
from __future__ import annotations

import numpy as np
import torch

from data_generation.src.transmitter.ofdm import qpsk
from src.algorithms.likelihood import batch_real_likelihood, make_whitener
from src.evaluation.inputs import prepare_batch


def likelihood_batches(dataset, snr_index, backend, *, batch_size=4, frame_limit=None):
    """Use exactly the matrix, pilot cancellation and thermal noise of baselines.

    No audit() call, reference target, uncertainty adjustment or test labels
    enter the detection tensors. Existing constructed CSI is reused as-is.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Batch size must be positive")
    if frame_limit is not None and (not isinstance(frame_limit, int) or isinstance(frame_limit, bool) or frame_limit < 1):
        raise ValueError("Frame limit must be positive")
    count = dataset.frame_count if frame_limit is None else min(dataset.frame_count, frame_limit)
    whitener = make_whitener(dataset._shared["unit_rf_noise_covariance"][:], backend)
    for d in range(2):
        for start in range(0, count, batch_size):
            frames = list(range(start, min(start+batch_size, count)))
            batch = prepare_batch(dataset, snr_index, frames, d, backend)
            gram, matched = batch_real_likelihood(batch, whitener, backend)
            # Labels are accessed only after preparation of label-free inputs.
            symbols = qpsk(dataset._h["samples/bits"][frames, d])
            yield (torch.as_tensor(gram, dtype=torch.float64, device=backend.device),
                   torch.as_tensor(matched, dtype=torch.float64, device=backend.device),
                   torch.as_tensor(symbols, dtype=torch.complex128, device=backend.device))
