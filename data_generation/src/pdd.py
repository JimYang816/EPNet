"""Versioned P-D-D-P experiment contract and record/window identities."""
from __future__ import annotations

import math
from data_generation.src.config import ROOT, load_config

FRAME_STRUCTURE = "PDDP"
SYMBOL_LABELS = ("P1", "D1", "D2", "P2")
SYMBOL_COUNT = len(SYMBOL_LABELS)
DATA_SYMBOL_INDICES = (1, 2)
TRAINING_SYMBOL_INDICES = (0, 3)
DATA_SCHEMA = "ncs1-pddp-full-csi-v1"
CACHE_SCHEMA = "ncs1-pddp-calibration-v1"
PRIOR_SCHEMA = "ncs1-pddp-frozen-prior-v1"
VALIDATION_SCHEMA = "pddp-fixed-validation-v1"
ESTIMATION_MODE = "full_frame_joint_no_feedback_v1"


def load_pdd_config(path=None):
    c = load_config(path or ROOT/"configs/data/ncs1_pddp.json")
    if c.get("channel", "NCS1") not in ("NCS1", "NOF1"):
        raise ValueError("PDDP supports WATERMARK NCS1 and NOF1")
    if c.get("model_mismatch_policy", "reject") not in ("reject", "diagnose"):
        raise ValueError("Unknown physical model mismatch policy")
    if not 0 <= float(c.get("csi_model_error_ratio", 0)) <= 1:
        raise ValueError("CSI model error ratio must be finite and between zero and one")
    if (c.get("frame_structure") != FRAME_STRUCTURE or c.get("estimation_mode") != ESTIMATION_MODE
            or c.get("feedback_iterations") != 0 or c.get("empirical_weight") != 0
            or c.get("prior_scale") != 1 or c.get("interference_scale") != 1
            or c["relative_regularizer"] != 1e-6):
        raise ValueError("PDDP requires the fixed full-frame joint no-feedback contract")
    if (c["null_per_edge"], c["data_count"], c["pilot_count"]) != (64, 512, 128):
        raise ValueError("PDDP requires 64 edge / 512 data / 128 pilot carriers")
    width = c.get("dpss_half_bandwidth_hz", float("nan"))
    if (not math.isfinite(width) or not 0 < width < c["fs_baseband"]/2
            or not 0 < c.get("dpss_tail_ratio", 0) < 1
            or not isinstance(c.get("dpss_expected_count"), int) or c["dpss_expected_count"] < 1
            or not isinstance(c.get("delay_length"), int) or not 1 <= c["delay_length"] <= c["nfft"]
            or not isinstance(c.get("delay_block_size"), int) or not 1 <= c["delay_block_size"] <= c["delay_length"]
            or c.get("validation_snr_db") != 30):
        raise ValueError("PDDP requires one fixed absolute DPSS bandwidth, tail, Q, L and delay block")
    return c


def identities(c, split):
    records = c[split+"_records"]
    if split == "test":
        return [(r, s) for r in records for s in c["odd_test_starts_seconds" if r % 2 else "even_test_starts_seconds"]]
    return [(r, s) for r in records for s in c["starts_seconds"]]
