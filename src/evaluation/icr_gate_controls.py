"""Paired gate/no-gate experiments with identical training and validation rules."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

from src.common.experiment import file_sha256, write_json
from src.training.icr_acceptance import assess


def comparison_rows(gate, no_gate, *, bootstrap_samples=2000):
    from src.evaluation.icr_mmse import record_interval

    for report in (gate, no_gate):
        if not report.get("complete") or report.get("input_view") != "compensated" or report.get("split") != "test":
            raise ValueError("Expected completed compensated-CSI test reports")
    for name in ("channel", "snr_db", "data_identity", "feedback_layout", "code", "smoke", "operator_backend"):
        if gate[name] != no_gate[name]:
            raise ValueError("Gate comparison identity differs: "+name)
    for name in ("model_config", "effective_model_config"):
        a, b = dict(gate[name]), dict(no_gate[name])
        if a.pop("use_gate") is not True or b.pop("use_gate") is not False or a != b:
            raise ValueError("Only use_gate may differ between paired model configurations")
    rows = []
    if len(gate["per_snr"]) != len(no_gate["per_snr"]):
        raise ValueError("Paired SNR coverage differs")
    for a, b in zip(gate["per_snr"], no_gate["per_snr"]):
        if any(a[k] != b[k] for k in ("snr_db", "records", "blocks", "total_bits")):
            raise ValueError("Paired test coverage differs")
        if not np.isclose(a["mean_nmse_linear"][0], b["mean_nmse_linear"][0], rtol=1e-6, atol=1e-10):
            raise ValueError("Paired input CSI differs")
        ar, br = ({r["record"]: r for r in x["by_record"]} for x in (a, b))
        if set(ar) != set(br) or len(ar) != a["records"]:
            raise ValueError("Paired physical records differ")
        ids = sorted(ar)
        if any(ar[k][v] != br[k][v] for k in ids for v in ("bits", "blocks")):
            raise ValueError("Paired record sample counts differ")
        delta = [ar[k]["bit_errors"][-1]-br[k]["bit_errors"][-1] for k in ids]
        interval = record_interval(delta, [ar[k]["bits"] for k in ids], resamples=bootstrap_samples)
        rows.append(dict(snr_db=a["snr_db"], total_bits=a["total_bits"], blocks=a["blocks"],
            gate_ber=a["ber"][-1], no_gate_ber=b["ber"][-1], ber_delta=interval["estimate"],
            gate_ber_relative_reduction_percent=100*(1-a["ber"][-1]/b["ber"][-1]) if b["ber"][-1] else None,
            paired_ber_interval=interval,
            input_nmse_linear=a["mean_nmse_linear"][0],
            gate_nmse_linear=a["mean_nmse_linear"][-1], no_gate_nmse_linear=b["mean_nmse_linear"][-1],
            nmse_delta_db=a["mean_nmse_db"][-1]-b["mean_nmse_db"][-1]))
    return rows


def run_pair(config, channel, seed, output, backend, *, resume=False, phase_callback=None, allow_smoke=False):
    from src.algorithms.icr_mmse.checkpoint import load_model
    from src.data.compact import CompactDataset
    from src.evaluation.icr_mmse import evaluate
    from src.training.icr_mmse import train

    if config.get("input_view") != "compensated":
        raise ValueError("Gate controls require the accepted compensated CSI")
    options = dict(config["training"])
    if options.get("sample_order") != "snr_interleaved" or options.get("per_snr_acceptance") is None:
        raise ValueError("Gate controls require interleaved training and per-SNR acceptance")
    channel_options = config["channel_options"][channel]
    options.update(seed=seed, lambda_h=channel_options["lambda_h"])
    model_config = dict(config["model"], layers=channel_options["layers"])
    output = Path(output)/channel.lower()/f"seed_{seed}"
    data = Path(config["datasets"][channel])
    with CompactDataset(data/"train.h5") as dataset:
        if dataset.channel != channel:
            raise ValueError("Configured channel does not match training data")
        if not allow_smoke and (dataset._config.get("smoke") or options.get("frame_limit") is not None):
            raise ValueError("Formal gate controls require the complete accepted dataset")
    checkpoints, payloads, reports = {}, {}, {}
    for name, enabled in (("gate", True), ("no_gate", False)):
        if phase_callback:
            phase_callback(name+"_training")
        warmup = output/"gate/warmup.pt" if not enabled else None
        checkpoints[name] = train(data, output/name, backend, model_config=dict(model_config, use_gate=enabled),
            warmup_checkpoint=warmup, resume=resume, **options)
        _, payloads[name] = load_model(checkpoints[name])
        for stage in ("B", "C"):
            selection = payloads[name]["per_snr_selection"][stage]
            if not assess(selection["selected_validation"], selection["limits"])["eligible"]:
                raise RuntimeError("Selected checkpoint violates per-SNR acceptance")
    # Both checkpoints have been selected without test access before scoring either.
    warmups = [load_model(output/name/"warmup.pt")[1] for name in ("gate", "no_gate")]
    if any(not torch.equal(value, warmups[1]["model"][key]) for key, value in warmups[0]["model"].items()):
        raise RuntimeError("Paired stage-A weights differ")
    schedules = [[(r["stage"], r["epoch"], r["sampling_seed"]) for r in payloads[name]["history"]
                  if r["stage"] in ("B", "C")] for name in ("gate", "no_gate")]
    if schedules[0] != schedules[1]:
        raise RuntimeError("Paired B/C sampling schedules differ")
    evaluation = dict(config["evaluation"])
    for name in checkpoints:
        if phase_callback:
            phase_callback(name+"_test_evaluation")
        reports[name] = evaluate(data/"test.h5", checkpoints[name], output/name/"evaluation", backend,
            include_baselines=False, **evaluation)
    result = dict(channel=channel, seed=seed, input_view="compensated", complete=True,
        shared_warmup_sha256=file_sha256(output/"gate/warmup.pt"), identical_warmup_weights=True,
        identical_sampling_schedule=True, sampling_schedule=schedules[0],
        checkpoints={name: dict(path=str(path.resolve()), sha256=file_sha256(path)) for name, path in checkpoints.items()},
        validation_selection={name: payloads[name]["per_snr_selection"] for name in payloads},
        rows=comparison_rows(reports["gate"], reports["no_gate"],
                             bootstrap_samples=evaluation.get("bootstrap_samples", 2000)))
    write_json(output/"pair.json", result)
    return result


def save_comparison(results, output):
    """Keep seed variability separate from within-seed physical-record intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"comparison.json", dict(schema="icr-paired-gate-controls-v1", input_view="compensated",
        pairs=results, nmse_definition="mean full-matrix per-block linear NMSE; average seeds in linear units before dB",
        statistics="BER intervals pair whole physical records within each seed; seed standard deviation is separate",
        selection="validation only; fixed per-SNR BER/NMSE bounds at B and C; test never selects a gate or seed"))
    rows = [dict(channel=result["channel"], seed=result["seed"],
                 **{k: v for k, v in row.items() if k != "paired_ber_interval"},
                 ber_delta_lower=row["paired_ber_interval"]["lower"],
                 ber_delta_upper=row["paired_ber_interval"]["upper"])
            for result in results for row in result["rows"]]
    with (output/"comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    channels = list(dict.fromkeys(result["channel"] for result in results))
    summaries = {}
    for channel in channels:
        pairs = [r for r in results if r["channel"] == channel]
        snrs = [r["snr_db"] for r in pairs[0]["rows"]]
        if any([r["snr_db"] for r in p["rows"]] != snrs for p in pairs):
            raise ValueError("SNR coverage differs across seeds")
        summaries[channel] = dict(seeds=[p["seed"] for p in pairs], per_snr=[])
        for i, snr in enumerate(snrs):
            item = dict(snr_db=snr)
            for metric in ("gate_ber", "no_gate_ber", "ber_delta", "gate_nmse_linear", "no_gate_nmse_linear"):
                values = [p["rows"][i][metric] for p in pairs]
                item[metric+"_mean"] = float(np.mean(values))
                item[metric+"_seed_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summaries[channel]["per_snr"].append(item)
    write_json(output/"seed_summary.json", summaries)
    for metric in ("ber", "nmse"):
        fig, axes = plt.subplots(1, len(channels), figsize=(6*len(channels), 4.5), squeeze=False, layout="constrained")
        for ax, channel in zip(axes[0], channels):
            pairs = [r for r in results if r["channel"] == channel]
            snr = [r["snr_db"] for r in pairs[0]["rows"]]
            for name, label, style in (("gate", "Gate", "o-"), ("no_gate", "No gate", "s--")):
                key = name+("_ber" if metric == "ber" else "_nmse_linear")
                values = np.array([[r[key] for r in p["rows"]] for p in pairs])
                mean = values.mean(0)
                if metric == "ber":
                    floor = np.array([1/r["total_bits"] for r in pairs[0]["rows"]])
                    ax.semilogy(snr, np.maximum(mean, floor), style, label=label)
                else:
                    ax.plot(snr, 10*np.log10(np.maximum(mean, 1e-30)), style, label=label)
            if metric == "nmse":
                ax.plot(snr, [10*math.log10(r["input_nmse_linear"]) for r in pairs[0]["rows"]], ":", label="Input CSI")
            ax.set(title=f"{channel} ({len(pairs)} paired seeds, mean)", xlabel="SNR (dB)",
                   ylabel="BER" if metric == "ber" else "Mean full-matrix NMSE (dB)", xticks=snr)
            ax.grid(True, which="both", alpha=.2)
            ax.legend()
        for extension in ("png", "svg"):
            fig.savefig(output/f"{metric}_snr.{extension}", dpi=180)
        plt.close(fig)
