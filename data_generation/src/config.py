"""Fixed experiment parameters and stable provenance identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    c = json.loads(Path(path or ROOT / "configs/data/ncs1_pddp.json").read_text(encoding="utf-8"))
    if (c["nfft"], c["cp"], c["fs_baseband"], c["fs_passband"], c["fft_advance"]) != (1024, 384, 6000, 96000, 32):
        raise ValueError("This experiment requires the fixed 1024/384, 6/96-kHz, advance-32 configuration")
    sets = [set(c[k]) for k in ("prior_records", "validation_records", "test_records")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Prior, validation and test records must be disjoint")
    return c


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
