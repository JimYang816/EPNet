"""A-D factorial controls: CSI repair crossed with the detector implementation."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from src.common.experiment import file_sha256, write_json


VARIANTS = {
    "A": "A: Fixed CSI + LMMSE",
    "B": "B: Fixed CSI + MMSE unfolding",
    "C": "C: ICR + LMMSE",
    "D": "D: ICR + MMSE unfolding",
}


def compare_reports(reports, *, bootstrap_samples=2000):
    from src.evaluation.icr_mmse import record_interval

    if set(reports) != set(VARIANTS):
        raise ValueError("An A-D comparison requires all four variants")
    first = reports["A"]
    for name, report in reports.items():
        if (not report.get("complete") or report.get("split") != "test"
                or report.get("input_view") != "compensated"):
            raise ValueError("Expected completed compensated-CSI test reports")
        for key in ("channel", "snr_db", "data_identity", "code", "smoke", "operator_backend"):
            if report[key] != first[key]:
                raise ValueError("A-D report identity differs: " + key)
        expected_detector = "lmmse" if name in "AC" else "unfolded"
        expected_correction = "fixed" if name in "AB" else "iterative"
        if (report["effective_model_config"]["detector_mode"] != expected_detector
                or report["effective_model_config"]["correction_mode"] != expected_correction):
            raise ValueError("Wrong A-D detector/correction assignment")
        for key in ("precision", "noise_mode", "bandwidth", "use_gate", "use_uncertainty"):
            if report["model_config"][key] != first["model_config"][key]:
                raise ValueError("A-D nuisance configuration differs: " + key)
    if len({reports[name]["model_config"]["layers"] for name in "BCD"}) != 1:
        raise ValueError("B/C/D must use the same joint depth")
    rows = []
    for index, snr in enumerate(first["snr_db"]):
        groups = {name: report["per_snr"][index] for name, report in reports.items()}
        base = groups["A"]
        records = {name: {row["record"]: row for row in group["by_record"]}
                   for name, group in groups.items()}
        ids = sorted(records["A"])
        for name, group in groups.items():
            if any(group[key] != base[key] for key in ("snr_db", "records", "blocks", "total_bits")):
                raise ValueError("A-D test coverage differs")
            if sorted(records[name]) != ids or len(ids) != group["records"]:
                raise ValueError("A-D physical records differ")
            if any(records[name][r][key] != records["A"][r][key]
                   for r in ids for key in ("bits", "blocks")):
                raise ValueError("A-D per-record coverage differs")
            if not np.isclose(group["mean_nmse_linear"][0], base["mean_nmse_linear"][0], rtol=1e-6):
                raise ValueError("A-D input CSI differs")
        differences = {}
        for a, b in (("B", "A"), ("D", "C"), ("C", "A"), ("D", "B")):
            errors = [records[a][r]["bit_errors"][-1] - records[b][r]["bit_errors"][-1] for r in ids]
            differences[a+"_minus_"+b] = record_interval(
                errors, [records["A"][r]["bits"] for r in ids], resamples=bootstrap_samples)
        rows.append(dict(snr_db=float(snr), total_bits=base["total_bits"], blocks=base["blocks"],
            input_nmse_linear=base["mean_nmse_linear"][0],
            variants={name: dict(ber=group["ber"][-1], nmse_linear=group["mean_nmse_linear"][-1],
                                nmse_db=group["mean_nmse_db"][-1]) for name, group in groups.items()},
            paired_ber_differences=differences))
    return rows


def run_quartet(config, channel, seed, output, backend, *, resume=False,
                phase_callback=None, allow_smoke=False):
    from src.algorithms.icr_mmse.checkpoint import load_model
    from src.data.compact import CompactDataset
    from src.evaluation.icr_mmse import evaluate, plot_report
    from src.training.icr_mmse import train

    if config.get("input_view") != "compensated":
        raise ValueError("A-D ablations require accepted compensated CSI")
    data = Path(config["datasets"][channel])
    output = Path(output)/channel.lower()/f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    options = dict(config["training"])
    a_epochs, b_epochs, c_epochs = options.pop("stage_epochs")
    if min(a_epochs, b_epochs, c_epochs) < 1:
        raise ValueError("Specify positive common stage budgets; variants skip only inapplicable stages")
    if options.get("sample_order") != "snr_interleaved" or options.get("per_snr_acceptance") is None:
        raise ValueError("A-D controls require identical sampling and fixed validation acceptance")
    options.update(seed=seed, lambda_h=config["channel_options"][channel]["lambda_h"])
    model_config = dict(config["model"], layers=config["channel_options"][channel]["layers"])
    with CompactDataset(data/"train.h5") as dataset:
        if dataset.channel != channel:
            raise ValueError("Wrong channel dataset")
        if not allow_smoke and (dataset._config.get("smoke") or options.get("frame_limit") is not None):
            raise ValueError("Formal A-D experiments require all accepted training records")

    def phase(value):
        if phase_callback:
            phase_callback(value)

    checkpoints, payloads = {}, {}
    phase("A_preparation")
    checkpoints["A"] = train(data, output/"A", backend, resume=resume,
        model_config=dict(model_config, layers=1, detector_mode="lmmse", correction_mode="fixed"),
        stage_epochs=(0, 0, 0), **options)
    phase("shared_warmup_training")
    train(data, output/"shared_warmup", backend, resume=resume,
          model_config=dict(model_config, detector_mode="unfolded", correction_mode="fixed"),
          stage_epochs=(a_epochs, 0, 0), **options)
    warmup = output/"shared_warmup/warmup.pt"
    for name in "BCD":
        phase(name+"_training")
        checkpoints[name] = train(data, output/name, backend, resume=resume,
            model_config=dict(model_config, detector_mode="lmmse" if name == "C" else "unfolded",
                              correction_mode="fixed" if name == "B" else "iterative"),
            stage_epochs=(0 if name == "C" else a_epochs, 0 if name == "B" else b_epochs, c_epochs),
            warmup_checkpoint=warmup if name in "BD" else None, **options)
    # No test file is opened before all four models are finalized on validation.
    for name, path in checkpoints.items():
        _, payloads[name] = load_model(path)
    _, shared = load_model(warmup)
    for name in "BD":
        _, reused = load_model(output/name/"warmup.pt")
        if any(not torch.equal(value, reused["model"][key])
               for key, value in shared["model"].items() if key.startswith("equalizers.")):
            raise RuntimeError("B and D did not start from the same pretrained detector")
    schedules = {name: [(r["stage"], r["epoch"], r["sampling_seed"]) for r in p["history"]]
                 for name, p in payloads.items()}
    if schedules["C"] != schedules["D"]:
        raise RuntimeError("C/D correction training schedules differ")
    if schedules["B"] != [row for row in schedules["D"] if row[0] == "C"]:
        raise RuntimeError("B/D detector fine-tuning schedules differ")
    _, c_initial = load_model(output/"C/warmup.pt")
    _, d_initial = load_model(output/"D/warmup.pt")
    if any(not torch.equal(value, d_initial["model"][key]) for key, value in c_initial["model"].items()
           if key.startswith("corrector.") or key == "eta_logits"):
        raise RuntimeError("C and D did not start from identical correction parameters")
    write_json(output/"training_audit.json", dict(shared_warmup_sha256=file_sha256(warmup),
        identical_B_D_warmup=True, identical_C_D_initial_correction=True, schedules=schedules,
        detector_training_epochs=dict(A=0, B=a_epochs+c_epochs, C=0, D=a_epochs+c_epochs),
        correction_training_epochs=dict(A=0, B=0, C=b_epochs+c_epochs, D=b_epochs+c_epochs),
        test_access=False))
    reports, performance = {}, {}
    evaluation = dict(config["evaluation"])
    for name, path in checkpoints.items():
        phase(name+"_test_evaluation")
        report = evaluate(data/"test.h5", path, output/name/"evaluation", backend,
                          include_baselines=False, **evaluation)
        model, _ = load_model(path)
        active = sum(parameter.numel() for key, parameter in model.named_parameters()
                     if name in "CD" or key.startswith("equalizers."))
        report["performance"]["stored_parameters"] = report["performance"]["parameters"]
        report["performance"]["parameters"] = active
        report["ablation_variant"] = name
        report["curves"] = {VARIANTS[name]: next(iter(report["curves"].values()))}
        plot_report(report, output/name/"evaluation")
        write_json(output/name/"evaluation/metrics.json", report)
        reports[name], performance[name] = report, report["performance"]
    result = dict(channel=channel, seed=seed, complete=True, input_view="compensated",
        checkpoints={name: dict(path=str(path.resolve()), sha256=file_sha256(path)) for name, path in checkpoints.items()},
        validation_selection={name: p["per_snr_selection"] for name, p in payloads.items()},
        performance=performance, rows=compare_reports(reports,
            bootstrap_samples=evaluation.get("bootstrap_samples", 2000)))
    write_json(output/"quartet.json", result)
    return result


def save_comparison(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"comparison.json", dict(schema="icr-ad-factorial-v1", quartets=results,
        variants=VARIANTS, input_view="compensated",
        statistics="Paired whole-physical-record bootstrap within a seed; seeds are not extra channel records",
        selection="All A-D checkpoints selected using validation before test access",
        timing="Measured end-to-end latency, not compute-matched; LMMSE A performs one effective solve",
        nmse="Mean full-matrix per-block linear NMSE, then dB"))
    rows = [dict(channel=result["channel"], seed=result["seed"], variant=name, snr_db=row["snr_db"],
                 **values, parameters=result["performance"][name]["parameters"],
                 seconds_per_block=result["performance"][name]["seconds_per_block"])
            for result in results for row in result["rows"] for name, values in row["variants"].items()]
    with (output/"comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    channels = list(dict.fromkeys(result["channel"] for result in results))
    seed_summary = {}
    for channel in channels:
        selected = [r for r in results if r["channel"] == channel]
        seed_summary[channel] = dict(seeds=[r["seed"] for r in selected], per_snr=[])
        for index, row in enumerate(selected[0]["rows"]):
            entry = dict(snr_db=row["snr_db"], variants={})
            for name in VARIANTS:
                values = {metric: [r["rows"][index]["variants"][name][metric] for r in selected]
                          for metric in ("ber", "nmse_linear")}
                entry["variants"][name] = {
                    metric: dict(mean=float(np.mean(v)), seed_std=float(np.std(v, ddof=1)) if len(v)>1 else None)
                    for metric, v in values.items()}
            seed_summary[channel]["per_snr"].append(entry)
    write_json(output/"seed_summary.json", seed_summary)
    for metric in ("ber", "nmse_linear"):
        fig, axes = plt.subplots(1, len(channels), figsize=(6.5*len(channels), 4.7), squeeze=False, layout="constrained")
        for ax, channel in zip(axes[0], channels):
            selected = [r for r in results if r["channel"] == channel]
            snr = [r["snr_db"] for r in selected[0]["rows"]]
            for (name, label), style in zip(VARIANTS.items(), ("o--", "s--", "^-", "d-")):
                values = np.mean([[r["variants"][name][metric] for r in item["rows"]] for item in selected], axis=0)
                if metric == "ber":
                    floor = np.array([.5/r["total_bits"] for r in selected[0]["rows"]])
                    ax.semilogy(snr, np.maximum(values, floor), style, label=label)
                else:
                    ax.plot(snr, 10*np.log10(np.maximum(values, 1e-30)), style, label=label)
            if metric != "ber":
                ax.plot(snr, [10*np.log10(r["input_nmse_linear"]) for r in selected[0]["rows"]], ":", label="Input CSI")
            ax.set(title=f"{channel} ({len(selected)} paired seed(s))", xlabel="SNR (dB)",
                   ylabel="BER" if metric == "ber" else "Mean full-matrix NMSE (dB)", xticks=snr)
            ax.grid(True, which="both", alpha=.25)
            ax.legend(fontsize=9)
        stem = "ber" if metric == "ber" else "nmse"
        for extension in ("png", "svg"):
            fig.savefig(output/f"{stem}_snr.{extension}", dpi=180)
        plt.close(fig)
