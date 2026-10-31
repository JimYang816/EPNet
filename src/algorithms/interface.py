"""Label-free batch contract shared by the comparison equalizers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EqualizerBatch:
    channel: Any
    raw_observations: Any
    cancelled_observations: Any
    variances: np.ndarray
    data_indices: np.ndarray
    noise_covariance: np.ndarray
    likelihoods: dict = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class EqualizerResult:
    symbols: np.ndarray
    rejected_sites: np.ndarray
