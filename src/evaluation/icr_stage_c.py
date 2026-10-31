"""Stage-C-only learning-rate and validation-selection controls with unchanged inference."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.common.experiment import file_sha256, write_json
from src.evaluation.fingerprints import DataIdentity


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def comparison_rows(original, candidate):
    """Compare matched full test sets; NMSE is averaged in linear units first."""
    for report in (original, candidate):
        if (not report.get("complete") or report.get("input_view") != "compensated"
                or report.get("split") != "test"):
            raise ValueError("Comparison requires complete compensated-CSI test reports")
    for name in ("channel", "snr_db", "model_config", "effective_model_config", "data_identity",
                 "feedback_layout", "code", "smoke"):
        if original.get(name) != candidate.get(name):
            raise ValueError("Original/candidate comparison identity differs: " + name)
    if original["performance"]["parameters"] != candidate["performance"]["parameters"]:
        raise ValueError("The learning-rate control must preserve the model parameter count")
    if len(original["per_snr"]) != len(candidate["per_snr"]):
        raise ValueError("SNR coverage differs")
    rows = []
    for old, new in zip(original["per_snr"], candidate["per_snr"]):
        for name in ("snr_db", "blocks", "total_bits", "records"):
            if old[name] != new[name]:
                raise ValueError("Test coverage differs: " + name)
        old_records = [(r["record"], r["blocks"], r["bits"]) for r in old["by_record"]]
        new_records = [(r["record"], r["blocks"], r["bits"]) for r in new["by_record"]]
        if old_records != new_records:
            raise ValueError("Physical-record coverage differs")
        if not np.isclose(old["mean_nmse_linear"][0], new["mean_nmse_linear"][0], rtol=1e-6, atol=1e-10):
            raise ValueError("Initial CSI differs")
        ber_old = old["bit_errors"][-1] / old["total_bits"]
        ber_new = new["bit_errors"][-1] / new["total_bits"]
        nmse_old, nmse_new = old["mean_nmse_linear"][-1], new["mean_nmse_linear"][-1]
        rows.append(dict(channel=original["channel"], snr_db=old["snr_db"],
            total_bits=old["total_bits"], blocks=old["blocks"],
            original_ber=ber_old, candidate_ber=ber_new, ber_delta=ber_new-ber_old,
            ber_relative_reduction_percent=100*(1-ber_new/ber_old) if ber_old else None,
            initial_nmse_db=10*math.log10(old["mean_nmse_linear"][0]),
            original_nmse_linear=nmse_old, candidate_nmse_linear=nmse_new,
            original_nmse_db=10*math.log10(nmse_old), candidate_nmse_db=10*math.log10(nmse_new),
            nmse_delta_db=10*math.log10(nmse_new/nmse_old)))
    return rows


def save_comparison(results, output):
    """Write numerical results and both required SNR figures, without retraining."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"comparison.json", dict(schema="icr-stage-c-lr-comparison-v1",
        input_view="compensated", channels=results,
        nmse_definition="mean complete-matrix per-block linear NMSE, then 10log10",
        selection="validation BER with per-channel constraints; the stage-B starting weights remain a candidate",
        note="One training seed; test data are scored only after validation selection."))
    rows = [row for result in results.values() for row in result["rows"]]
    with (output/"comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}):
        for metric in ("ber", "nmse_db"):
            fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4.5),
                                     squeeze=False, layout="constrained")
            for ax, (channel, result) in zip(axes[0], results.items()):
                data = result["rows"]
                snr = [row["snr_db"] for row in data]
                curves = [("original", "Original selected model", "o--", "#63758B"),
                    ("candidate", result.get("candidate_label", "C lr=1e-4 (selected)"), "s-", "#147D92")]
                if "previous_ber" in data[0]:
                    curves.insert(1, ("previous", "C: both lr=1e-4", "^:", "#AD6CB6"))
                for name, label, style, color in curves:
                    values = [row[f"{name}_{metric}"] for row in data]
                    if metric == "ber":
                        values = [max(value, 1/row["total_bits"]) for value, row in zip(values, data)]
                        ax.semilogy(snr, values, style, color=color, label=label)
                    else:
                        ax.plot(snr, values, style, color=color, label=label)
                if metric == "nmse_db":
                    ax.plot(snr, [row["initial_nmse_db"] for row in data], ":", color="#A5AAB2", label="Input CSI")
                ax.set(title=channel, xlabel="SNR (dB)", ylabel="BER" if metric == "ber" else "Mean full-matrix NMSE (dB)", xticks=snr)
                ax.grid(True, which="both", alpha=.2)
                ax.legend(fontsize=8, frameon=False)
            name = "ber_snr" if metric == "ber" else "nmse_snr"
            for extension in ("png", "svg"):
                fig.savefig(output/f"{name}.{extension}", dpi=180)
            plt.close(fig)


def run_channel(source_root, output, channel, backend, *, learning_rate=1e-4,
                resume=False, phase_callback=None, allow_smoke=False,
                correction_learning_rate=None, nmse_max_relative_degradation=None, previous_root=None):
    """Inherit the actual selected run, skip A/B, run C and compare to its original final model."""
    from src.algorithms.icr_mmse.checkpoint import load_model
    from src.evaluation.icr_mmse import evaluate
    from src.training.icr_mmse import train

    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("Use a separate experiment output, outside the source run")
    if learning_rate != 1e-4:
        raise ValueError("This control is fixed to a stage-C learning rate of 1e-4")
    if correction_learning_rate is not None and not 0 < correction_learning_rate < learning_rate:
        raise ValueError("Correction learning rate must be positive and smaller than the equalizer learning rate")
    configuration = _read(source_root/"resolved_config.json")
    if configuration.get("input_view") != "compensated":
        raise ValueError("The source experiment must use compensated CSI")
    seeds = configuration.get("selection", {}).get("seeds", [configuration["training"]["seed"]])
    if len(seeds) != 1:
        raise ValueError("The source experiment must have exactly one training seed")
    source = source_root/channel.lower()/f"seed_{seeds[0]}"
    run = output/channel.lower()
    run.mkdir(parents=True, exist_ok=True)
    source_paths = dict(training=source/"training.json", stage_b=source/"correction_pretrained.pt",
        original_final=source/"icr_mmse.pt", original_report=source/"evaluation/metrics.json",
        source_configuration=source_root/"resolved_config.json")
    if previous_root is not None:
        previous_run = Path(previous_root).resolve()/channel.lower()
        source_paths.update(previous_report=previous_run/"evaluation/metrics.json",
                            previous_checkpoint=previous_run/"icr_mmse.pt")
    guard = DataIdentity(source_paths)
    training = _read(source/"training.json")
    original = _read(source/"evaluation/metrics.json")
    if (not training.get("complete") or training.get("input_view") != "compensated"
            or training.get("channel") != channel or original.get("channel") != channel
            or original.get("smoke") and not allow_smoke):
        raise ValueError("Expected the completed original experiment for this channel")
    if original.get("checkpoint_sha256") != guard.files["original_final"]["sha256"]:
        raise ValueError("Original evaluation does not match its final checkpoint")
    options = dict(training["controls"])
    model_config = options.pop("model")
    if options.pop("input_view") != "compensated":
        raise ValueError("Source training input view differs")
    if options.get("frame_limit") is not None and not allow_smoke:
        raise ValueError("Formal continuation requires all training frames")
    if options.get("warmup_checkpoint") or options.get("stage_b_checkpoint"):
        raise ValueError("Expected an original full A/B/C run")
    if options["learning_rate"] != 1e-3 or options["seed"] != seeds[0]:
        raise ValueError("Expected the original 1e-3 run and its configured seed")
    options.update(learning_rate=learning_rate, stage_b_checkpoint=str(source/"correction_pretrained.pt"))
    if correction_learning_rate is not None:
        options["stage_c_correction_learning_rate"] = correction_learning_rate
    if nmse_max_relative_degradation is not None:
        options["stage_c_nmse_max_relative_degradation"] = nmse_max_relative_degradation
    manifest = dict(channel=channel, source=str(source), source_identity=guard.files,
        model_config=model_config, training_options=options, executed_stages=["C"],
        stage_c_epochs=options["stage_epochs"][2], changes_from_source=dict(
            equalizer_learning_rate=learning_rate, correction_learning_rate=correction_learning_rate or learning_rate,
            validation_nmse_max_relative_degradation=nmse_max_relative_degradation))
    if (run/"continuation.json").exists() and _read(run/"continuation.json") != manifest:
        raise ValueError("Continuation source/configuration changed")
    write_json(run/"continuation.json", manifest)
    if phase_callback:
        phase_callback("stage_c_training")
    checkpoint = train(configuration["datasets"][channel], run, backend,
        model_config=model_config, resume=resume, **options)
    if phase_callback:
        phase_callback("test_evaluation")
    evaluation = configuration.get("evaluation", {})
    candidate = evaluate(Path(configuration["datasets"][channel])/"test.h5", checkpoint,
        run/"evaluation", backend, batch_size=evaluation.get("batch_size", 2),
        bootstrap_samples=evaluation.get("bootstrap_samples", 2000), include_baselines=False,
        operator_backend=original["operator_backend"])
    rows = comparison_rows(original, candidate)
    previous_selection = None
    if previous_root is not None:
        previous = _read(source_paths["previous_report"])
        if previous.get("checkpoint_sha256") != guard.files["previous_checkpoint"]["sha256"]:
            raise ValueError("Previous control evaluation does not match its checkpoint")
        paired = comparison_rows(previous, candidate)
        previous_selection = previous["selection"]
        for row, prior in zip(rows, paired):
            row.update(previous_ber=prior["original_ber"], previous_nmse_db=prior["original_nmse_db"],
                ber_delta_vs_previous=prior["ber_delta"], nmse_delta_db_vs_previous=prior["nmse_delta_db"])
    _, before = load_model(source/"correction_pretrained.pt")
    _, after = load_model(checkpoint)
    unchanged = all(torch.equal(before["model"][name], value) for name, value in after["model"].items())
    guard.assert_unchanged()
    result = dict(channel=channel, source_identity=guard.files, checkpoint_sha256=file_sha256(checkpoint),
        learning_rate=learning_rate, stage_c_epochs=options["stage_epochs"][2], model_config=model_config,
        lambda_h=options["lambda_h"], selected_stage_b_weights=unchanged,
        original_selection=original["selection"], candidate_selection=candidate["selection"], rows=rows,
        correction_learning_rate=correction_learning_rate or learning_rate,
        validation_nmse_max_relative_degradation=nmse_max_relative_degradation,
        stage_c_selection=after.get("stage_c_selection"), previous_selection=previous_selection,
        candidate_label=("C: grouped LR + NMSE bound" if nmse_max_relative_degradation is not None
                         else "C lr=1e-4 (selected)"))
    save_comparison({channel: result}, run/"comparison")
    return result
