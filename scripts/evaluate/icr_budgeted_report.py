"""Summarize a completed frozen budgeted-unfolding study without rerunning models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.experiment import file_sha256, write_json


CASES = ("fixed_ncs1", "fixed_nof1", "joint_ncs1", "joint_nof1")
TITLES = ("NCS1 · Fixed CSI", "NOF1 · Fixed CSI",
          "NCS1 · Iterative CSI correction", "NOF1 · Iterative CSI correction")
STYLES = (
    ("historical_lmmse", "Original LMMSE", "#202c39", "s", "--"),
    ("historical_unfolding", "Original unfolding", "#a7a7a7", "^", ":"),
    ("repaired", "Repaired unfolding", "#007a87", "o", "-"),
)


def summarize(root: Path):
    manifest = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or set(manifest["cases"]) != set(CASES):
        raise ValueError("The four-case frozen study must be complete")
    reports, results = {}, {}
    for name in CASES:
        path = root / name / "comparison.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (not report.get("complete") or report["split"] != manifest["split"]
                or report["checkpoint_sha256"] != manifest["checkpoint_sha256"][name]):
            raise ValueError(f"{name}: incomplete or mismatched frozen report")
        reports[name] = report
        models = report["models"]
        new, old = models["repaired"], models["historical_unfolding"]
        linear = models["historical_lmmse"]
        results[name] = dict(
            source_report=str(path), source_report_sha256=file_sha256(path),
            checkpoint_sha256=report["checkpoint_sha256"],
            acceptance=manifest["cases"][name]["acceptance"],
            unfolding_layers=dict(original=old["model_config"]["layers"],
                                  repaired=new["model_config"]["layers"]),
            mean_ber={key: value["mean_ber"] for key, value in models.items()},
            relative_ber_reduction_vs_historical_lmmse=1-new["mean_ber"]/linear["mean_ber"],
            total_milliseconds_per_block={key: 1000*value["seconds_per_block"]
                                          for key, value in models.items()},
            total_budget_ratio=new["seconds_per_block"]/old["seconds_per_block"],
            median_model_budget_ratio=(new["benchmark"]["median_seconds_per_block"]
                                       / old["benchmark"]["median_seconds_per_block"]),
            worse_snr_vs_historical_lmmse=[row["snr_db"] for i, row in enumerate(new["per_snr"])
                if row["ber"][-1] > linear["per_snr"][i]["ber"][-1]],
            learning_gain_supported_at_95_percent=(report["comparisons"]["initialized"]["overall"]["upper"] < 0),
            paired_mean_ber_differences={key: value["overall"]
                                         for key, value in report["comparisons"].items()},
            per_snr=[dict(snr_db=row["snr_db"], total_bits=row["total_bits"],
                          ber={key: value["per_snr"][i]["ber"][-1]
                               for key, value in models.items()},
                          nmse_db={key: value["per_snr"][i]["mean_nmse_db"][-1]
                                   for key, value in models.items()})
                     for i, row in enumerate(new["per_snr"])],
            historical_reproduction=report["historical_reproduction"],
        )
    result = dict(split=manifest["split"], detector_policy=manifest.get("detector_policy", "historical_mixed"),
        cases=results,
        environment=reports[CASES[0]]["environment"],
        timing=reports[CASES[0]]["timing"],
        limitations=[
            "One training seed; record bootstrap does not measure training-seed variability.",
            "Per-SNR BER and joint CSI NMSE need not improve, even when mean BER improves.",
            "Changed trainable parameters do not by themselves establish a gain over unlearned initialization; inspect the paired intervals.",
            "NOF1 original unfolding has 22 layers; repaired unfolding has 8.",
            "This budget-constrained repair study is not a matched-depth paper A-D ablation.",
            "Loaded LMMSE uses the learned detector's loading, not independently optimized loading.",
            "Historical reproduction allows absolute BER discrepancy <=1e-5 per SNR; actual differences are recorded.",
            "Per-block budget is steady state; loading and one-time frozen preparation are excluded.",
        ])
    write_json(root / "results_digest.json", result)
    return reports, result


def plot(reports, root: Path, split: str):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    for metric, label, filename in (("ber", "BER", "ber_overview.png"),
                                    ("mean_nmse_db", "Mean full-channel NMSE (dB)", "nmse_overview.png")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
        for ax, name, title in zip(axes.flat, CASES, TITLES):
            report = reports[name]
            for key, legend, color, marker, linestyle in STYLES:
                rows = report["models"][key]["per_snr"]
                ax.plot([row["snr_db"] for row in rows], [row[metric][-1] for row in rows],
                        label=legend, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.8, markersize=4.5)
            if metric == "ber":
                ax.set_yscale("log")
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_ylabel(label)
            ax.set_xticks(range(0, 31, 5))
            ax.grid(alpha=0.2, which="both")
        for ax in axes[-1]:
            ax.set_xlabel("SNR (dB)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.96), frameon=False)
        fig.suptitle(f"Frozen {split} · Compensated CSI · One training seed", y=0.995, fontsize=13)
        fig.text(0.5, 0.014, "Original/repaired depth: NCS1 8/8; NOF1 22/8. Fixed-CSI LMMSE uses one solve.",
                 ha="center", fontsize=9, color="#444444")
        fig.tight_layout(rect=(0, 0.035, 1, 0.91), h_pad=2)
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, name in zip(axes, ("joint_ncs1", "joint_nof1")):
        for key, legend, color, marker in (
                ("lmmse", "LMMSE", "#202c39", "s"),
                ("original", "Legacy detector", "#a7a7a7", "^"),
                ("initialized", "New detector, unlearned", "#bd7931", "d"),
                ("repaired", "New detector, learned", "#007a87", "o")):
            row = next(row for row in reports[name]["models"][key]["per_snr"]
                       if row["snr_db"] == 30)
            ax.semilogy(range(1, len(row["ber"])+1), row["ber"],
                        label=legend, color=color, marker=marker, linewidth=1.8, markersize=4.5)
        ax.set_title(name.split("_")[1].upper(), loc="left", fontweight="bold")
        ax.set_xlabel("Unfolding layer / CSI correction round")
        ax.set_ylabel("BER")
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.2, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=False)
    fig.suptitle(f"Frozen {split} · 30 dB · Parameter-learning control", y=0.995, fontsize=13)
    fig.text(0.5, 0.015, "All controls share the repaired CSI-corrector weights; feedback and CSI trajectories can differ.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.86))
    fig.savefig(root / "layerwise_30db.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "outputs/runs/icr_budgeted_unified_20261031/frozen_test_exact_legacy"))
    args = parser.parse_args()
    reports, result = summarize(args.root)
    plot(reports, args.root, result["split"])
    for name, row in result["cases"].items():
        ber = row["mean_ber"]
        print(f"{name}: BER {ber['historical_lmmse']:.8f} -> {ber['repaired']:.8f}; "
              f"total budget {row['total_budget_ratio']:.4f}; "
              f"warm model budget {row['median_model_budget_ratio']:.4f}")


if __name__ == "__main__":
    main()
