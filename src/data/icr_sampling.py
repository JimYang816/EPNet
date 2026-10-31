"""Complete, reproducible SNR-interleaved batches without changing sample weights."""
from __future__ import annotations

import numpy as np


def epoch_sampling_seed(seed, stage, epoch):
    """An epoch's order is independent of model RNG, prefetch and earlier runs."""
    if stage not in "ABC" or len(stage) != 1 or epoch < 1:
        raise ValueError("Expected stage A/B/C and a positive epoch")
    return int(np.random.SeedSequence([seed, "ABC".index(stage), epoch]).generate_state(1)[0])


def batch_schedule(frames, snr_indices, batch_size, *, sampling_seed=None):
    """Yield (SNR, D-block, frames); each sample occurs once, including tails.

    Each interleaved round visits every SNR once with the same shuffled frames.
    Batches remain homogeneous in SNR and D-block, retaining the existing
    batch size, compensation coefficient and physical time basis.
    """
    if sampling_seed is None:
        for si in snr_indices:
            for block in range(2):
                for start in range(0, len(frames), batch_size):
                    yield si, block, frames[start:start+batch_size]
        return
    rng = np.random.default_rng(sampling_seed)
    previous = None
    for block in rng.permutation(2).tolist():
        shuffled = rng.permutation(frames).tolist()
        for start in range(0, len(shuffled), batch_size):
            order = rng.permutation(snr_indices).tolist()
            if len(order) > 1 and order[0] == previous:
                order = order[1:] + order[:1]
            for si in order:
                yield si, block, shuffled[start:start+batch_size]
                previous = si
