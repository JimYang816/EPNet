"""Symbol supervision is read separately, after label-free preprocessing."""
from __future__ import annotations

import numpy as np


def symbols(dataset, frames, d):
    return np.stack([dataset.labels(frame)["symbols"][d] for frame in frames]).astype(np.complex128)


def mse(estimate, target):
    # Paper equation (32): mean over examples of the squared vector norm.
    return (estimate-target).abs().square().sum(dim=-1).mean()
