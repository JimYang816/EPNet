"""Export the completed six-equalizer test comparison as publication figures.

Run: python -m scripts.visualize.equalizer_comparison_paper
The PDF/SVG/600-dpi PNG files are written beside outputs/paper/main.tex.
NMSE has two curves: shared fixed input CSI and ICR's refined CSI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
import numpy as np

from scripts.visualize.icr_ablation_paper import WIDTH, export, select_font


DEFAULT_RUN = Path("outputs/runs/icr_full_20261031_095549")
CHANNELS = ("NOF1", "NCS1")
METHODS = ("Single-tap MMSE", "Full-matrix LMMSE", "MMSE-PIC", "UDNet", "OAMP-Net2", "ICR-MMSE-Net")
STYLES = {
    "Single-tap MMSE": dict(color="#8C6D31", linestyle=":", marker="v", markersize=4.2),
    "Full-matrix LMMSE": dict(color="#454545", linestyle="--", marker="o", markersize=4.2),
    "MMSE-PIC": dict(color="#0072B2", linestyle="-.", marker="s", markersize=3.8),
    "UDNet": dict(color="#AA4499", linestyle=(0, (5, 1, 1, 1)), marker="^", markersize=4.4),
    "OAMP-Net2": dict(color="#009E73", linestyle=(0, (3, 1, 1, 1, 1, 1)), marker="+", markersize=5.),
    "ICR-MMSE-Net": dict(color="#D55E00", linestyle="-", marker="D", markersize=3.7),
}
NMSE_METHODS = ("Shared input CSI (five baselines)", "ICR-MMSE-Net: refined CSI")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reports(run, config):
    if config.get("input_view") != "compensated":
        raise ValueError("The paper comparison requires accepted compensated CSI")
    state = json.loads((run/"progress.json").read_text(encoding="utf-8"))
    if state.get("status") != "completed" or not state.get("complete"):
        raise ValueError("The source experiment has not completed")
    reports = {}
    for channel in CHANNELS:
        report = json.loads((run/"comparison"/channel.lower()/"metrics.json").read_text(encoding="utf-8"))
        if (not report.get("complete") or report.get("smoke") or report.get("channel") != channel
                or report.get("input_view") != "compensated" or set(report.get("curves", {})) != set(METHODS)
                or report.get("snr_db") != [0, 5, 10, 15, 20, 25, 30]
                or report.get("frame_count") != 600 or report.get("bits_per_snr") != 1228800
                or report["nmse"].get("split") != "test"):
            raise ValueError(f"{channel}: incomplete formal six-algorithm test report")
        for name, supplied in config["checkpoints"][channel].items():
            if not report["checkpoint_paths"][name].replace("\\", "/").endswith(Path(supplied).as_posix()):
                raise ValueError(f"{channel}/{name}: report uses a checkpoint outside the current config")
        for name in METHODS:
            curve = report["curves"][name]
            errors, bits = np.asarray(curve["errors"]), np.asarray(curve["total_bits"])
            values = np.asarray(curve["ber"])
            if (bits.tolist() != [1228800]*7 or values.shape != (7,)
                    or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                    or not np.allclose(values, errors/bits, rtol=1e-12, atol=0)):
                raise ValueError(f"{channel}/{name}: inconsistent test BER")
        for key in ("mean_nmse_db", "corrected_mean_nmse_db"):
            values = np.asarray(report["nmse"][key])
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError(f"{channel}: incomplete NMSE curve")
        component = json.loads((run/"comparison"/channel.lower()/"icr_mmse/metrics.json").read_text(encoding="utf-8"))
        if (report["icr_mmse"]["sha256"] != component["checkpoint_sha256"]
                or report["curves"]["ICR-MMSE-Net"] != component["curves"]["ICR-MMSE-Net"]
                or not np.allclose(report["nmse"]["corrected_mean_nmse_db"],
                    [row["mean_nmse_db"][-1] for row in component["per_snr"]], rtol=0, atol=1e-12)):
            raise ValueError(f"{channel}: ICR curve differs from its component report")
        reports[channel] = report
    return reports


def style_axes(ax, channel, metric, index):
    ax.set_xlim(-.7, 30.7)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlabel("SNR (dB)", labelpad=3.)
    ax.set_title(f"({'ab'[index]}) {channel}", fontsize=10., pad=6.)
    ax.grid(which="major", color="#D3D3D3", linewidth=.45, linestyle="--", zorder=0)
    ax.tick_params(axis="both", which="major", direction="in", length=3., width=.65, pad=3.)
    ax.tick_params(axis="both", which="minor", direction="in", length=1.6, width=.5)
    if metric == "ber":
        ax.set_yscale("log")
        ax.set_ylim(1e-3, .35)
        ax.yaxis.set_major_locator(LogLocator(base=10., numticks=4))
        ax.yaxis.set_minor_locator(LogLocator(base=10., subs=np.arange(2, 10)*.1, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("BER", labelpad=4.)
    else:
        ax.set_ylabel("Mean CSI NMSE (dB)", labelpad=4.)
        ax.set_ylim((-21.6, -12.) if channel == "NOF1" else (-16., -5.5))
        ax.yaxis.set_major_locator(MultipleLocator(2.))
        ax.yaxis.set_minor_locator(MultipleLocator(1.))


def add_curve(ax, snr, values, style, *, bits=None):
    values = np.asarray(values)
    shown = values.copy()
    zero = values == 0 if bits is not None else np.zeros(len(values), dtype=bool)
    if zero.any():
        shown[zero] = .5/np.asarray(bits)[zero]
    line, = ax.plot(snr, shown, **style, linewidth=1.3 if style["color"] == "#D55E00" else 1.1,
                    markerfacecolor="white", markeredgewidth=.85, zorder=3)
    assert np.array_equal(line.get_ydata(), shown)
    if zero.any():
        ax.plot(np.asarray(snr)[zero], shown[zero], linestyle="none", marker="v",
                markerfacecolor="white", markeredgecolor=style["color"], markersize=5., zorder=10)
    low, high = ax.get_ylim()
    if np.any(shown <= low) or np.any(shown >= high):
        raise ValueError("A measurement would be outside the visible axis range")


def create_figures(run, output, config):
    reports = load_reports(run, config)
    output.mkdir(parents=True, exist_ok=True)
    font = select_font()
    with plt.rc_context({"font.family": font, "font.size": 9., "mathtext.fontset": "stix",
                         "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "axes.unicode_minus": True, "figure.dpi": 160}):
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
        fig.subplots_adjust(left=.085, right=.987, bottom=.18, top=.755, wspace=.285)
        for index, channel in enumerate(CHANNELS):
            report = reports[channel]
            style_axes(axes[index], channel, "ber", index)
            for name in METHODS:
                row = report["curves"][name]
                add_curve(axes[index], report["snr_db"], row["ber"], STYLES[name], bits=row["total_bits"])
        handles = [Line2D([], [], **STYLES[name], linewidth=1.2, markerfacecolor="white",
                          markeredgewidth=.85, label=name) for name in METHODS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.535, .995), ncol=3,
                   frameon=False, handlelength=3.0, handletextpad=.6, columnspacing=1.7,
                   labelspacing=.45, borderaxespad=0., fontsize=8.7)
        export(fig, output, "equalizer_comparison_ber", "Six-equalizer test comparison: BER",
               subject="Formal six-equalizer comparison; accepted compensated CSI test data")
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.85))
        fig.subplots_adjust(left=.085, right=.987, bottom=.18, top=.79, wspace=.285)
        nmse_styles = (STYLES["Full-matrix LMMSE"], STYLES["ICR-MMSE-Net"])
        for index, channel in enumerate(CHANNELS):
            report = reports[channel]
            style_axes(axes[index], channel, "nmse", index)
            for key, style in zip(("mean_nmse_db", "corrected_mean_nmse_db"), nmse_styles):
                add_curve(axes[index], report["snr_db"], report["nmse"][key], style)
        handles = [Line2D([], [], **style, linewidth=1.2, markerfacecolor="white",
                          markeredgewidth=.85, label=label) for label, style in zip(NMSE_METHODS, nmse_styles)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.535, .995), ncol=2,
                   frameon=False, handlelength=3.0, handletextpad=.65, columnspacing=1.6,
                   borderaxespad=0., fontsize=8.7)
        export(fig, output, "equalizer_comparison_nmse", "Six-equalizer test comparison: CSI NMSE",
               subject="Formal six-equalizer comparison; accepted compensated CSI test data")
    with (output/"equalizer_comparison_plot_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "metric", "method", "snr_db", "errors", "total_bits", "value", "unit"])
        for channel, report in reports.items():
            for name in METHODS:
                row = report["curves"][name]
                for i, snr in enumerate(report["snr_db"]):
                    writer.writerow([channel, "BER", name, snr, row["errors"][i], row["total_bits"][i], row["ber"][i], "ratio"])
            for label, key in zip(NMSE_METHODS, ("mean_nmse_db", "corrected_mean_nmse_db")):
                for i, snr in enumerate(report["snr_db"]):
                    writer.writerow([channel, "full_matrix_NMSE", label, snr, "", "", report["nmse"][key][i], "dB"])
    captions = {
        "ber": "BER comparison of six equalizers on the accepted compensated-CSI test sets: "
               "(a) NOF1 and (b) NCS1. All receivers use the same test observations and initial CSI. "
               "Each SNR point contains 1,228,800 evaluated bits per receiver.",
        "nmse": "Mean full-matrix CSI NMSE on the accepted test sets: (a) NOF1 and (b) NCS1. "
                "Single-tap MMSE, full-matrix LMMSE, MMSE-PIC, UDNet, and OAMP-Net2 retain the same "
                "input CSI and therefore share one NMSE curve. ICR-MMSE-Net additionally reports "
                "its final refined CSI. Linear per-block NMSE values are averaged before conversion to decibels.",
    }
    snippets = []
    for metric, caption in captions.items():
        snippets.append("\\begin{figure*}[!t]\n    \\centering\n"
                        f"    \\includegraphics[width=\\textwidth]{{equalizer_comparison_{metric}.pdf}}\n"
                        f"    \\caption{{{caption}}}\n    \\label{{fig:equalizer_comparison_{metric}}}\n\\end{{figure*}}\n")
    (output/"equalizer_comparison_figures.tex").write_text("\n".join(snippets), encoding="utf-8")
    files = [output/f"equalizer_comparison_{metric}.{suffix}" for metric in ("ber", "nmse") for suffix in ("pdf", "svg", "png")]
    manifest = dict(source_run=str(run.resolve()), split="test", input_view="compensated", font=font,
        width_inches=WIDTH, png_dpi=600, source_reports={channel: dict(
            path=str((run/"comparison"/channel.lower()/"metrics.json").resolve()),
            sha256=sha256(run/"comparison"/channel.lower()/"metrics.json"),
            checkpoint_sha256=reports[channel]["icr_mmse"]["sha256"]) for channel in CHANNELS},
        data_policy="Exact completed six-algorithm report values, no smoothing or offsets. "
                    "Shared input CSI NMSE is shown once for the five fixed-CSI baselines.",
        output_files={path.name: dict(sha256=sha256(path), bytes=path.stat().st_size) for path in files})
    (output/"equalizer_comparison_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), files=list(manifest["output_files"])), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/ber_comparison.json"))
    args = parser.parse_args()
    create_figures(args.run, args.output, json.loads(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
