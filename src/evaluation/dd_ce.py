"""Public DD-CE receiver setup and separate, post-detection full-CSI scoring."""
from __future__ import annotations

import numpy as np

from src.algorithms.dd_ce_lmmse.detector import build_modes


def receiver_context(dataset, backend, parameters):
    n = dataset._config["nfft"]
    windows = dataset._model.sampled_basis.reshape(4, n, -1)
    return [dict(known=dataset._known[d], prior_scale=parameters["prior_scale"],
                 modes=build_modes(windows[d], dataset._blocks, parameters["rank"], backend))
            for d in (1, 2)]


def corrected_nmse(dataset, frames, d, corrected, backend):
    windows, which = np.unique(dataset._h["samples/window_index"][frames], return_inverse=True)
    reference = np.stack([dataset._h["windows/reference_csi"][int(w), d] for w in windows])
    reference = backend.array(reference.astype(np.complex128))[which]
    axes = (-2, -1)
    energy = (abs(reference)**2).sum(axes)
    if not bool((energy > 0).all()):
        raise ValueError("Zero-energy reference CSI")
    return backend.numpy((abs(corrected-reference)**2).sum(axes)/energy)
