"""Plot completed validation diagnostics; no data files or test scores are read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRIALS = (("joint_lr1e-3", "Joint", "#2384b9"),
          ("joint_lr1e-4", "Joint", "#38a8a0"),
          ("snr30_lr1e-4", "30 dB only", "#c55454"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finish(figure, path):
    figure.savefig(path, dpi=170)
    plt.close(figure)


def trials(report):
    for name, scope, color in TRIALS:
        controls = report["trials"][name]["controls"]
        label = (f"{scope}, +{controls['epochs']} ep, "
                 f"LR={controls['learning_rate']:g}, B={controls['batch_size']}")
        yield name, label, color


def plot_channel(root, channel):
    run = root/channel.lower()
    report = read(run/"diagnostics.json")
    if not report.get("complete") or report["input_view"] != "compensated":
        raise ValueError("Only completed compensated-CSI diagnostic reports may be plotted")
    original = report["original_validation"]
    snr = [x["snr_db"] for x in original["by_snr"]]
    final_ber = lambda result: [x["ber"][-1] for x in result["by_snr"]]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, color in (("Full-matrix LMMSE", "#e07a13"), ("MMSE-PIC", "#26944c")):
        axis.semilogy(snr, [x["ber"] for x in original["baselines"][name]["by_snr"]],
                      "o-", color=color, lw=1.8, ms=4, label=name)
    axis.semilogy(snr, final_ber(original), "o-", color="#7650a4", lw=2, ms=4,
                  label="Original OAMP-Net2")
    for name, label, color in trials(report):
        result = report["trials"][name]["final"]
        if name.startswith("snr30"):
            # This is one warm-started SNR-specific model, not a complete bank
            # of independently trained models for every SNR.
            axis.semilogy([30], [final_ber(result)[-1]], "*", color=color, ms=12, label=label)
        else:
            axis.semilogy(snr, final_ber(result), ".--", color=color, lw=1.1, ms=4, label=label)
    axis.set(title=f"{channel} | fixed validation set, compensated CSI", xlabel="SNR (dB)",
             ylabel="BER", xticks=snr)
    axis.legend(fontsize=8)
    finish(figure, run/"ber_snr.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, label, color in trials(report):
        history = [x for x in read(run/name/"history.json") if "by_snr" in x]
        epochs = [0]+[x["additional_epoch"] for x in history]
        for axis, metric, title in zip(axes, ("ber", "vector_mse"), ("BER", "Squared vector error")):
            axis.plot(epochs, [original["by_snr"][-1][metric][-1]]
                      +[x["by_snr"][-1][metric][-1] for x in history], color=color, label=label)
            selected = report["trials"][name]
            axis.plot(selected["best_additional_epoch"], selected["final"]["by_snr"][-1][metric][-1],
                      "*", color=color, ms=10)
            axis.set(xlabel="Additional epochs (each trial's training scope)", ylabel=title)
    for axis, metric in zip(axes, ("ber", "vector_mse")):
        axis.axhline(original["by_snr"][-1][metric][-1], color="#7650a4", ls=":", label="Original OAMP-Net2")
        axis.axhline(original["baselines"]["MMSE-PIC"]["by_snr"][-1][metric],
                     color="#26944c", ls="--", label="MMSE-PIC")
    axes[0].legend(fontsize=7)
    figure.suptitle(f"{channel} | 30 dB validation; stars selected by validation MSE")
    finish(figure, run/"fine_tuning.png")

    csi = read(root/"_control/validation_nmse.json")[channel]
    figure, axis = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    axis.plot(csi["snr_db"], csi["mean_nmse_db"], "o-", color="#487fb2", label="Common fixed input CSI")
    axis.set(title=f"{channel} | validation, compensated CSI", xlabel="SNR (dB)",
             ylabel="Mean full-matrix NMSE (dB)", xticks=csi["snr_db"])
    axis.legend(fontsize=9)
    finish(figure, run/"nmse_snr.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--channel", choices=("NOF1", "NCS1"))
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22,
                         "axes.spines.top": False, "axes.spines.right": False})
    for channel in ([args.channel] if args.channel else ["NOF1", "NCS1"]):
        plot_channel(args.results, channel)


if __name__ == "__main__":
    main()
