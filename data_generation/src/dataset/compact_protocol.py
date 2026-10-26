"""Independent compact learning-data contract; historical full-CSI files stay v1."""
from __future__ import annotations

import numpy as np

SCHEMA = "watermark-pddp-compact-v2"
MODEL_SCHEMA = SCHEMA + "-model"
SPLITS = ("train", "validation", "test")
SPLIT_CODES = dict(train=2, validation=3, test=4)
TEST_RECORDS = list(range(11, 48, 4))
VALIDATION_RECORDS = list(range(13, 46, 8))
TRAIN_RECORDS = sorted(set(range(1, 51))-set(TEST_RECORDS)-set(VALIDATION_RECORDS))
TARGET_PARAMETERS = dict(NOF1=(-12.7, 5.6), NCS1=(-6.2, 4.47))


def target_nmse_db(channel, snr):
    a, b = TARGET_PARAMETERS[channel]
    return a-b*(-np.expm1(-np.asarray(snr, dtype=float)/10))/(-np.expm1(-3.0))


def plan(smoke=False):
    return dict(protocol=SCHEMA,
                records=dict(train=[1], validation=[2], test=[3]) if smoke else
                dict(train=TRAIN_RECORDS, validation=VALIDATION_RECORDS, test=TEST_RECORDS),
                starts_seconds=[2] if smoke else list(range(2, 31, 2)),
                repeats=dict(train=1, validation=1, test=2 if smoke else 4))


def physical_identities(c, split):
    p = c["learning_dataset"]
    return [(r, float(t)) for r in p["records"][split] for t in p["starts_seconds"]]


def sample_identities(c, split):
    return [(w, repeat, repeat) for w in range(len(physical_identities(c, split)))
            for repeat in range(c["learning_dataset"]["repeats"][split])]


def validate_plan(c):
    p = c["learning_dataset"]
    if p.get("protocol") != SCHEMA:
        raise ValueError("Compact protocol mismatch")
    groups = [c["prior_records"], c["validation_records"]]+[p["records"][s] for s in SPLITS]
    if any(len(g) != len(set(g)) or not g for g in groups):
        raise ValueError("Records must be unique and nonempty")
    if any(set(a) & set(b) for i, a in enumerate(groups) for b in groups[:i]):
        raise ValueError("Prior, CSI validation and learning splits must be disjoint")
    duration = (4*(c["nfft"]+c["cp"])+2*c["guard_samples"])/c["fs_baseband"]
    starts = p["starts_seconds"]
    if (not starts or any(not np.isfinite(t) or t < c["guard_samples"]/c["fs_baseband"] for t in starts)
            or any(b-a < duration for a, b in zip(starts, starts[1:]))):
        raise ValueError("Learning windows must be valid, increasing and nonoverlapping")
    if any(not isinstance(p["repeats"][s], int) or p["repeats"][s] < 1 for s in SPLITS):
        raise ValueError("Repeat counts must be positive integers")
    if not c.get("smoke"):
        if p != plan() or c["prior_records"] != list(range(51, 56)) or c["validation_records"] != list(range(56, 61)):
            raise ValueError("Formal generation requires the frozen 60-record split and repeat plan")
        if c["snr_db"] != list(range(0, 31, 5)):
            raise ValueError("Formal SNR must be 0:5:30")
    return count_manifest(c)


def count_manifest(c):
    result = {}
    for split in SPLITS:
        windows = len(physical_identities(c, split))
        frames = len(sample_identities(c, split))
        result[split] = dict(records=c["learning_dataset"]["records"][split], physical_windows=windows,
                             repeats=c["learning_dataset"]["repeats"][split], frames_per_snr=frames,
                             d_blocks_all_snr=frames*2*len(c["snr_db"]),
                             bits_per_snr=frames*2*c["data_count"]*2)
    return result
