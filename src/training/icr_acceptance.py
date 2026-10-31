"""Validation-only, fixed per-SNR BER and full-matrix NMSE acceptance bounds."""
from __future__ import annotations

import math


def acceptance_controls(options):
    defaults = dict(ber_relative_tolerance=0., ber_absolute_tolerance=0., nmse_relative_tolerance=0.)
    if not isinstance(options, dict) or set(options) - set(defaults):
        raise ValueError("Unknown per-SNR acceptance controls")
    defaults.update(options)
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
           for v in defaults.values()):
        raise ValueError("Per-SNR tolerances must be finite and nonnegative")
    return defaults


def _rows(metrics):
    rows = metrics["per_snr"]
    if not rows or len({r["snr_db"] for r in rows}) != len(rows):
        raise ValueError("Per-SNR acceptance requires distinct, nonempty SNR coverage")
    for row in rows:
        if (row["bits"] <= 0 or row["blocks"] <= 0
                or any(row[k] is None or not math.isfinite(row[k]) or row[k] < 0
                       for k in ("ber", "corrected_nmse", "initial_nmse"))):
            raise ValueError("Per-SNR acceptance requires finite BER and full-matrix NMSE")
    return {row["snr_db"]: row for row in rows}


def make_limits(reference, options):
    options = acceptance_controls(options)
    return [dict(snr_db=snr, bits=row["bits"], blocks=row["blocks"],
                 ber_limit=row["ber"]*(1+options["ber_relative_tolerance"])+options["ber_absolute_tolerance"],
                 nmse_limit=row["corrected_nmse"]*(1+options["nmse_relative_tolerance"]))
            for snr, row in sorted(_rows(reference).items())]


def assess(metrics, limits):
    """A favorable average cannot override a violation in any SNR group."""
    rows = _rows(metrics)
    if len(limits) != len(rows) or {r["snr_db"] for r in limits} != set(rows):
        raise ValueError("Per-SNR validation coverage changed")
    violations = []
    for limit in limits:
        row = rows[limit["snr_db"]]
        if any(row[k] != limit[k] for k in ("bits", "blocks")):
            raise ValueError("Per-SNR validation sample counts changed")
        for metric, name in (("ber", "ber_limit"), ("corrected_nmse", "nmse_limit")):
            if row[metric] > limit[name]:
                violations.append(dict(snr_db=row["snr_db"], metric=metric,
                                       value=row[metric], limit=limit[name]))
    return dict(eligible=not violations, violations=violations)
