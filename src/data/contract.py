"""User-fixed default CSI view and accepted NOF1/NCS1 NMSE curves."""
from __future__ import annotations

import numpy as np


DEFAULT_INPUT_VIEW = "compensated"
NMSE_ENDPOINTS_DB = {"NOF1": (-12.70, -18.30), "NCS1": (-6.20, -10.67)}


def target_nmse_db(channel, snr_db):
    start, stop = NMSE_ENDPOINTS_DB[channel]
    snr = np.asarray(snr_db, dtype=float)
    return start + (stop-start)*(1-np.exp(-snr/10))/(1-np.exp(-3))


def validate_compensated_contract(dataset):
    """Reject changed formal targets or alpha; never rewrite accepted data."""
    if dataset.input_view != DEFAULT_INPUT_VIEW:
        raise ValueError("Default training/evaluation requires compensated CSI")
    if dataset._config.get("smoke"):
        return
    if dataset.channel not in NMSE_ENDPOINTS_DB:
        raise ValueError("Unknown channel in the fixed default data contract")
    h = dataset._h
    expected = target_nmse_db(dataset.channel, dataset.snr_db)
    saved = h["compensation/target_nmse_db"][:]
    actual = 10*np.log10(h["compensation/raw_mean_nmse_linear"][:]
                        * h["compensation/alpha"][:]**2)
    if (not np.allclose(saved, expected, rtol=0, atol=1e-9)
            or not np.allclose(actual, expected, rtol=0, atol=1e-8)):
        raise ValueError("CSI NMSE differs from the user-fixed NOF1/NCS1 data contract")
