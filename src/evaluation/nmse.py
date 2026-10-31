"""Test CSI NMSE: mean of per-D linear Frobenius errors, then decibels."""
from __future__ import annotations

import json

import numpy as np

from data_generation.src.dataset.compact_math import coefficient_nmse
from data_generation.src.dataset.compact_protocol import SCHEMA


def manifest_values(dataset, files):
    path = dataset.path.parent/"manifest.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        values = np.array(report["splits"]["test"]["mean_nmse_db"], dtype=float)
        if (report.get("complete") and report["protocol"] == SCHEMA
                and report["channel"] == dataset.channel
                and report["config_hash"] == dataset._h.attrs["config_hash"]
                and report["splits"]["test"]["checked_frames"] == dataset.frame_count
                and np.array_equal(report["snr_db"], dataset.snr_db)
                and values.shape == dataset.snr_db.shape and np.all(np.isfinite(values))
                and all(report["files"][name]["sha256"] == files[name]["sha256"] for name in ("test.h5", "model.h5"))):
            return values
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def mean_nmse_db(dataset, si):
    h = dataset._h
    windows = h["samples/window_index"][:]
    total = 0.0
    for wi in np.unique(windows):
        frames = np.flatnonzero(windows == wi)
        coefficients = h["samples/raw_coefficients"][si, frames]
        reference = h["windows/reference_csi"][int(wi)]
        values = coefficient_nmse(dataset._model, coefficients, reference)
        total += float(values.sum())*float(h["compensation/alpha"][si])**2
    mean = total/(dataset.frame_count*2)
    return float(10*np.log10(max(mean, np.finfo(float).tiny)))


def evaluate_nmse(dataset, entry, files):
    accepted = manifest_values(dataset, files)
    for si in range(len(dataset.snr_db)):
        if not entry.complete[si]:
            value = float(accepted[si]) if accepted is not None else mean_nmse_db(dataset, si)
            entry.commit((si,), mean_nmse_db=value)
