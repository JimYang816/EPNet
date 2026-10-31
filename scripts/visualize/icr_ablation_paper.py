"""Draw publication figures directly from the completed formal A-D test reports.

Run: python -m scripts.visualize.icr_ablation_paper
All final images are exported beside the paper's main.tex. Source measurements
are neither smoothed nor shifted; marker/line styles remain distinct in grayscale.
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
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
import numpy as np


DEFAULT_RUN = Path("outputs/runs/icr_ad_current_20261031_133445")
DEFAULT_OUTPUT = Path("outputs/paper")
CHANNELS = ("NOF1", "NCS1")
GROUPS = ("A", "B", "C", "D")
STYLES = {
    "A": dict(color="#454545", linestyle="--", marker="o", markersize=4.3),
    "B": dict(color="#0072B2", linestyle="-.", marker="s", markersize=3.7),
    "C": dict(color="#009E73", linestyle=":", marker="^", markersize=4.5),
    "D": dict(color="#D55E00", linestyle="-", marker="D", markersize=3.6),
}
LABELS = {
    "A": "A: Fixed CSI + LMMSE",
    "B": "B: Fixed CSI + MMSE unfolding",
    "C": "C: Iterative CSI + LMMSE",
    "D": "D: Iterative CSI + MMSE unfolding",
}
WIDTH = 7.16  # inches; IEEE double-column width


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reports(run):
    progress = json.loads((run/"progress.json").read_text(encoding="utf-8"))
    digest = json.loads((run/"results_digest.json").read_text(encoding="utf-8"))
    if not progress.get("complete") or not digest.get("complete"):
        raise ValueError("Publication requires a completed formal ablation")
    reports = {}
    for channel in CHANNELS:
        path = run/"test"/channel.lower()/"comparison.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (not report.get("complete") or report.get("split") != "test"
                or report.get("input_view") != "compensated" or report.get("channel") != channel
                or report.get("snr_db") != [0, 5, 10, 15, 20, 25, 30]
                or set(report.get("curves", {})) != set(GROUPS)):
            raise ValueError(f"{channel}: requires all four formal test groups")
        for group in GROUPS:
            row = report["curves"][group]
            errors, bits = np.asarray(row["errors"]), np.asarray(row["total_bits"])
            ber = np.asarray(row["ber"])
            nmse = np.asarray(row["mean_nmse_linear"])
            nmse_db = np.asarray(row["mean_nmse_db"])
            if (bits.tolist() != [1228800]*7 or ber.shape != (7,)
                    or nmse.shape != (7,) or not np.isfinite(ber).all()
                    or not np.isfinite(nmse).all() or np.any(ber <= 0) or np.any(nmse <= 0)
                    or not np.allclose(ber, errors/bits, rtol=1e-12, atol=0)
                    or not np.allclose(nmse_db, 10*np.log10(nmse), rtol=1e-12, atol=1e-12)):
                raise ValueError(f"{channel}/{group}: inconsistent or incomplete test measurements")
        if not np.array_equal(report["curves"]["A"]["mean_nmse_linear"],
                              report["curves"]["B"]["mean_nmse_linear"]):
            raise ValueError("Fixed-CSI A/B NMSE must coincide")
        reports[channel] = report
    return reports


def select_font():
    for name in ("Times New Roman", "STIXGeneral", "DejaVu Serif"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    raise RuntimeError("No suitable publication font")


def legend(fig):
    handles = [Line2D([], [], **STYLES[group], linewidth=1.25, markerfacecolor="white",
                      markeredgewidth=.85, label=LABELS[group]) for group in GROUPS]
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.535, .995),
                      ncol=2, frameon=False, handlelength=3.1, columnspacing=2.1,
                      handletextpad=.7, labelspacing=.45, borderaxespad=0., fontsize=9.)


def panel(ax, report, metric, letter):
    channel = report["channel"]
    snr = np.asarray(report["snr_db"])
    artists = []
    for group in GROUPS:
        values = np.asarray(report["curves"][group][metric])
        style = dict(STYLES[group])
        # Only marker visibility changes on the identical A/B NMSE curves.
        # All seven measurements still form each line at their true positions.
        if metric == "mean_nmse_db" and group in ("A", "B"):
            style["markevery"] = [0, 2, 4, 6] if group == "A" else [1, 3, 5]
        line, = ax.plot(snr, values, **style, linewidth=1.3 if group == "D" else 1.15,
                        markerfacecolor="white", markeredgewidth=.85,
                        zorder=3+GROUPS.index(group))
        # Numerical QA: every published vertex must equal its source value.
        if not np.array_equal(line.get_xdata(), snr) or not np.array_equal(line.get_ydata(), values):
            raise RuntimeError("Plotted measurements changed")
        artists.append(line)
    ax.set_xlim(-.7, 30.7)
    ax.set_xticks(snr)
    ax.set_xlabel("SNR (dB)", labelpad=3.)
    ax.set_title(f"({letter}) {channel}", fontsize=10., pad=6.)
    ax.grid(which="major", color="#D3D3D3", linewidth=.45, linestyle="--", zorder=0)
    ax.tick_params(axis="both", which="major", direction="in", length=3., width=.65, pad=3.)
    ax.tick_params(axis="both", which="minor", direction="in", length=1.6, width=.5)
    if metric == "ber":
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3e-1)
        ax.yaxis.set_major_locator(LogLocator(base=10., numticks=4))
        ax.yaxis.set_minor_locator(LogLocator(base=10., subs=np.arange(2, 10)*.1, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("BER", labelpad=4.)
    else:
        ax.set_ylabel("Mean CSI NMSE (dB)", labelpad=4.)
        ax.set_ylim((-21.6, -12.) if channel == "NOF1" else (-16., -5.5))
        ax.yaxis.set_major_locator(MultipleLocator(2.))
        ax.yaxis.set_minor_locator(MultipleLocator(1.))
    for line in artists:
        low, high = ax.get_ylim()
        if np.any(line.get_ydata() <= low) or np.any(line.get_ydata() >= high):
            raise ValueError("A data point would be outside the visible axis range")


def verify_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    labels = []
    for ax in fig.axes:
        labels.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        for positions, ticks, limits in ((ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
                                        (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())):
            labels.extend(tick for position, tick in zip(positions, ticks)
                          if limits[0] <= position <= limits[1])
    for item in fig.legends:
        labels.extend(item.get_texts())
        box = item.get_window_extent(renderer)
        for ax in fig.axes:
            if box.overlaps(ax.get_window_extent(renderer)):
                raise ValueError("Legend overlaps a plot panel")
    for label in labels:
        if not label.get_visible() or not label.get_text():
            continue
        box = label.get_window_extent(renderer)
        if box.x0 < bounds.x0 or box.y0 < bounds.y0 or box.x1 > bounds.x1 or box.y1 > bounds.y1:
            raise ValueError(f"Figure text is clipped: {label.get_text()}")


def export(fig, output, stem, title, *, subject="Formal A-D ablation; accepted compensated CSI test data"):
    verify_layout(fig)
    for suffix in ("pdf", "svg", "png"):
        options = dict(dpi=600, facecolor="white")
        if suffix == "pdf":
            options["metadata"] = dict(Title=title, Subject=subject)
        fig.savefig(output/(stem+"."+suffix), **options)
    plt.close(fig)


def write_latex(output):
    definition = (
        "A: fixed CSI with full-matrix LMMSE; "
        "B: fixed CSI with MMSE unfolding; "
        "C: iterative CSI refinement with full-matrix LMMSE; "
        "D: iterative CSI refinement with MMSE unfolding (ICR-MMSE-Net). "
    )
    nmse_note = ("The NMSE curves of A and B coincide. "
                 "Full-matrix NMSE is averaged in the linear domain before conversion to decibels.")
    snippets = ["% Choose the combined figure OR the two separate figures below.\n"
                "% PDF assets are in the same directory as main.tex.\n"]
    for stem, caption, label in (
        ("icr_ablation", "Ablation on the accepted compensated-CSI test sets: "
         "(a) NOF1 BER; (b) NCS1 BER; (c) NOF1 CSI NMSE; and (d) NCS1 CSI NMSE. "
         +definition+nmse_note, "fig:icr_ablation"),
        ("icr_ablation_ber", "BER of the four ablation variants on (a) NOF1 and (b) NCS1 "
         "using the accepted compensated-CSI test sets. "+definition, "fig:icr_ablation_ber"),
        ("icr_ablation_nmse", "Mean full-matrix CSI NMSE of the four ablation variants "
         "on (a) NOF1 and (b) NCS1. "+definition+nmse_note, "fig:icr_ablation_nmse"),
    ):
        snippets.append("\\begin{figure*}[!t]\n    \\centering\n"
                        f"    \\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n"
                        f"    \\caption{{{caption}}}\n    \\label{{{label}}}\n\\end{{figure*}}\n")
    (output/"icr_ablation_figures.tex").write_text("\n".join(snippets), encoding="utf-8")


def create_figures(run, output):
    reports = load_reports(run)
    output.mkdir(parents=True, exist_ok=True)
    font = select_font()
    with plt.rc_context({
        "font.family": font, "font.size": 9., "mathtext.fontset": "stix",
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": True, "figure.dpi": 160,
        "savefig.pad_inches": .02,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.2))
        fig.subplots_adjust(left=.085, right=.987, bottom=.092, top=.862, wspace=.285, hspace=.48)
        for row, metric in enumerate(("ber", "mean_nmse_db")):
            for col, channel in enumerate(CHANNELS):
                panel(axes[row, col], reports[channel], metric, "abcd"[row*2+col])
        legend(fig)
        export(fig, output, "icr_ablation", "A-D ablation: BER and CSI NMSE on NOF1 and NCS1")
        for metric, stem, title in (("ber", "icr_ablation_ber", "A-D ablation: BER"),
                                    ("mean_nmse_db", "icr_ablation_nmse", "A-D ablation: mean CSI NMSE")):
            fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.85))
            fig.subplots_adjust(left=.085, right=.987, bottom=.18, top=.772, wspace=.285)
            for col, channel in enumerate(CHANNELS):
                panel(axes[col], reports[channel], metric, "ab"[col])
            legend(fig)
            export(fig, output, stem, title)
    with (output/"icr_ablation_plot_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "group", "snr_db", "errors", "total_bits", "ber", "mean_full_matrix_nmse_linear", "mean_full_matrix_nmse_db"])
        for channel, report in reports.items():
            for group in GROUPS:
                row = report["curves"][group]
                for i, snr in enumerate(report["snr_db"]):
                    writer.writerow([channel, group, snr, row["errors"][i], row["total_bits"][i],
                                     row["ber"][i], row["mean_nmse_linear"][i], row["mean_nmse_db"][i]])
    write_latex(output)
    files = [output/(stem+"."+suffix) for stem in ("icr_ablation", "icr_ablation_ber", "icr_ablation_nmse")
             for suffix in ("pdf", "svg", "png")]
    manifest = dict(source_run=str(run.resolve()), split="test", input_view="compensated",
        source_reports={ch: dict(path=str((run/"test"/ch.lower()/"comparison.json").resolve()),
            sha256=sha256(run/"test"/ch.lower()/"comparison.json")) for ch in CHANNELS},
        font=font, width_inches=WIDTH, png_dpi=600, pdf_fonttype=42,
        data_policy="Exact source vertices, no smoothing, no offsets; NMSE averaged linearly before dB. "
                    "Alternating A/B NMSE markers expose their coincident curves.",
        output_files={path.name: dict(sha256=sha256(path), bytes=path.stat().st_size) for path in files})
    (output/"icr_ablation_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), font=font, files=list(manifest["output_files"])), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    create_figures(args.run, args.output)


if __name__ == "__main__":
    main()
