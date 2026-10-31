"""Shared BER definition, kept separate from scheduling and plotting."""
from __future__ import annotations

import numpy as np

from data_generation.src.transmitter.ofdm import decisions


def bit_errors(symbols, truth):
    if symbols.shape != truth.shape[:-1] or not np.all(np.isfinite(symbols)):
        raise ValueError("Detector returned invalid symbol dimensions or values")
    return np.count_nonzero(decisions(symbols) != truth, axis=(-2, -1)).astype(np.int64)


def score_batch(dataset, frames, d, result):
    """Keep label selection inside the versioned metric computation boundary."""
    truth = dataset._h["samples/bits"][frames, d]
    errors = bit_errors(result.symbols, truth)
    rejected = np.asarray(result.rejected_sites)
    if rejected.shape != (len(frames),) or rejected.dtype.kind not in "iu" or np.any(rejected < 0):
        raise ValueError("Invalid rejected-site diagnostics")
    return errors, rejected
