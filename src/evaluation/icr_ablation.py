"""Aggregate the formal, equal-depth A–D ablation with paired record statistics."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.experiment import write_json
from src.evaluation.icr_budgeted import paired


GROUPS = {
    "A": "Fixed CSI + Full-matrix LMMSE",
    "B": "Fixed CSI + MMSE Unfolding",
    "C": "Iterative CSI + Full-matrix LMMSE",
    "D": "Iterative CSI + MMSE Unfolding",
}
SNR = [0, 5, 10, 15, 20, 25, 30]


def validate_report(report, channel, group, split):
    if (not report.get("complete") or report.get("smoke")
            or report.get("input_view") != "compensated" or report.get("channel") != channel
            or report.get("split") != split or report.get("snr_db") != SNR):
        raise ValueError(f"{channel}/{group}/{split}: requires a complete formal report")
    model = report["effective_model_config"]
    detector = "lmmse" if group in ("A", "C") else "budgeted"
    correction = "fixed" if group in ("A", "B") else "iterative"
    if (model["detector_mode"] != detector or model["correction_mode"] != correction
            or model["layers"] != 8 or model["correction_passes"] != 1
            or model["equalizer_repeats"] != 1):
        raise ValueError(f"{channel}/{group}: wrong ablation architecture")
    blocks = {"validation": 150, "test": 1200}[split]
    layers = 1 if group == "A" else 8
    rows = report["per_snr"]
    if len(rows) != len(SNR) or [row["snr_db"] for row in rows] != SNR:
        raise ValueError("Incomplete SNR coverage")
    for row in rows:
        if (row["blocks"] != blocks or row["total_bits"] != blocks*1024
                or len(row["bit_errors"]) != layers or len(row["mean_nmse_linear"]) != layers+1):
            raise ValueError(f"{channel}/{group}: incomplete bit or layer coverage")
        records = row["by_record"]
        if (len({r["record"] for r in records}) != row["records"]
                or sum(r["bits"] for r in records) != row["total_bits"]
                or sum(r["blocks"] for r in records) != blocks
                or sum(r["bit_errors"][-1] for r in records) != row["bit_errors"][-1]):
            raise ValueError("Physical-record totals disagree with report")


def summarize(reports):
    if set(reports) != set(GROUPS):
        raise ValueError("Formal ablation requires all four groups A–D")
    reference = reports["A"]
    channel, split = reference["channel"], reference["split"]
    curves = {}
    for group, report in reports.items():
        validate_report(report, channel, group, split)
        if report["data_identity"] != reference["data_identity"]:
            raise ValueError("A–D reports must use identical data files")
        rows = report["per_snr"]
        if not np.allclose([r["mean_nmse_linear"][0] for r in rows],
                           [r["mean_nmse_linear"][0] for r in reference["per_snr"]], rtol=1e-6):
            raise ValueError("A–D initial CSI differs")
        if group in ("A", "B") and not all(np.allclose(r["mean_nmse_linear"],
                r["mean_nmse_linear"][0], rtol=1e-6) for r in rows):
            raise ValueError("Fixed-CSI groups changed CSI")
        errors = [row["bit_errors"][-1] for row in rows]
        bits = [row["total_bits"] for row in rows]
        nmse = [row["mean_nmse_linear"][-1] for row in rows]
        curves[group] = dict(label=GROUPS[group], errors=errors, total_bits=bits,
            ber=[e/n for e, n in zip(errors, bits)], mean_ber=sum(errors)/sum(bits),
            mean_nmse_linear=nmse, mean_nmse_db=(10*np.log10(np.maximum(nmse, 1e-30))).tolist(),
            checkpoint=report["checkpoint"], checkpoint_sha256=report["checkpoint_sha256"],
            performance=report["performance"], model_config=report["effective_model_config"])
    for key in ("bandwidth", "precision", "noise_mode", "use_uncertainty", "use_gate"):
        if len({report["effective_model_config"][key] for report in reports.values()}) != 1:
            raise ValueError(f"A–D model setting differs: {key}")
    return dict(complete=True, channel=channel, split=split, input_view="compensated", snr_db=SNR,
        data_identity=reference["data_identity"], curves=curves,
        paired_ber_difference={f"{a}_minus_{b}": paired(reports[a]["per_snr"], reports[b]["per_snr"])
                               for a, b in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C"))},
        statistics="Whole physical-record bootstrap; negative differences favor the first group. "
                   "One training seed; no across-seed stability claim.")


def save_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, logarithmic in (("ber", "BER", True),
                                       ("mean_nmse_db", "Mean full-matrix NMSE (dB)", False)):
        with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False}):
            fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
            for group, row in report["curves"].items():
                values = np.asarray(row[metric])
                if logarithmic:
                    values = np.maximum(values, .5/np.asarray(row["total_bits"]))
                ax.plot(SNR, values, "o-", label=f"{group}: {row['label']}")
            if logarithmic:
                ax.set_yscale("log")
            ax.set(xlabel="SNR (dB)", ylabel=ylabel,
                   title=f"{report['channel']} | A–D ablation | {report['split']}")
            ax.grid(True, which="both", alpha=.25)
            ax.legend(fontsize=9)
            name = "ber_snr" if logarithmic else "nmse_snr"
            for suffix in ("png", "svg"):
                fig.savefig(directory/(name+"."+suffix), dpi=180)
            plt.close(fig)
    with (directory/"comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "split", "group", "snr_db", "errors", "total_bits", "ber", "mean_nmse_db"])
        for group, row in report["curves"].items():
            for i, snr in enumerate(SNR):
                writer.writerow([report["channel"], report["split"], group, snr, row["errors"][i],
                                 row["total_bits"][i], row["ber"][i], row["mean_nmse_db"][i]])
    write_json(directory/"comparison.json", report)
