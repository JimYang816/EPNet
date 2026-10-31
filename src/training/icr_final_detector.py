"""Bounded validation-only continuation of a completed paired gate experiment."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from src.algorithms.icr_mmse.checkpoint import load_model
from src.common.experiment import file_sha256, write_json
from src.evaluation.fingerprints import DataIdentity
from src.training.icr_acceptance import assess
from src.training.icr_mmse import train


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_pair(source_root, output, channel, backend, *, epochs=2, learning_rate=1e-4,
             resume=False, phase_callback=None, allow_smoke=False):
    """Skip A/B and all test evaluation; select against each variant's fixed B."""
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    if output == source_root or source_root in output.parents:
        raise ValueError("Use a separate output outside the source experiment")
    if (isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 2
            or isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("This diagnostic permits only one or two epochs and a positive learning rate")
    config_path = source_root/"_control/config.json"
    config = _read(config_path)
    if config.get("input_view") != "compensated" or len(config.get("seeds", [])) != 1:
        raise ValueError("Expected the compensated-CSI single-seed gate experiment")
    if channel not in config["channels"]:
        raise ValueError("Channel is absent from the source experiment")
    seed = config["seeds"][0]
    source = source_root/channel.lower()/f"seed_{seed}"
    guard = DataIdentity(dict(config=config_path, **{
        variant: source/variant/"correction_pretrained.pt" for variant in ("gate", "no_gate")}))
    payloads = {variant: load_model(source/variant/"correction_pretrained.pt")[1]
                for variant in ("gate", "no_gate")}
    first = payloads["gate"]
    for variant, payload in payloads.items():
        if (payload.get("stage") != "B" or payload.get("channel") != channel
                or payload.get("seed") != seed or payload.get("smoke") and not allow_smoke
                or payload["model_config"]["use_gate"] != (variant == "gate")):
            raise ValueError("Wrong stage/channel/seed/gate or smoke source checkpoint")
        if dict(payload["model_config"], use_gate=True) != first["model_config"]:
            raise ValueError("Paired source model configurations differ beyond the gate")
        for name, value in first["model"].items():
            if name.startswith("equalizers.") and not torch.equal(value, payload["model"][name]):
                raise ValueError("Paired source detector weights differ")
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    schedules = []
    for variant, before in payloads.items():
        if phase_callback:
            phase_callback(variant+"_validation_training")
        source_controls = before["identity"]["controls"]
        keys = ("lambda_h", "batch_size", "frame_limit", "operator_backend", "prefetch_batches",
                "reference_cache_mib", "compact_reference", "sample_order")
        options = {name: source_controls[name] for name in keys if name in source_controls}
        options.update(seed=seed, model_config=before["model_config"], stage_epochs=(1, 1, epochs),
            learning_rate=learning_rate, stage_c_final_equalizer_only=True,
            stage_b_checkpoint=str(source/variant/"correction_pretrained.pt"), per_snr_acceptance={})
        if options.get("frame_limit") is not None and not allow_smoke:
            raise ValueError("Formal validation requires every training/validation frame")
        if variant == "no_gate":
            paired = {key: value for key, value in options.items() if key not in ("model_config", "stage_b_checkpoint")}
            if paired != paired_options:
                raise ValueError("Paired training conditions differ")
        else:
            paired_options = {key: value for key, value in options.items()
                              if key not in ("model_config", "stage_b_checkpoint")}
        run = output/variant
        checkpoint = train(config["datasets"][channel], run, backend, resume=resume, **options)
        _, after = load_model(checkpoint)
        selection = after["per_snr_selection"]["C"]
        baseline, selected = selection["reference_validation"], selection["selected_validation"]
        prefix = f'equalizers.{before["model_config"]["layers"]-1}.'
        changed = [name for name, value in before["model"].items()
                   if not torch.equal(value, after["model"][name])]
        if any(not name.startswith(prefix) for name in changed):
            raise RuntimeError("A frozen parameter changed")
        if not assess(selected, selection["limits"])["eligible"]:
            raise RuntimeError("Selected candidate violates a fixed per-SNR bound")
        for entry in after["history"]:
            if entry["stage"] != "C":
                raise RuntimeError("The diagnostic must not restart A/B")
            for ref, actual in zip(baseline["per_snr"], entry["validation"]["per_snr"]):
                if any(ref[key] != actual[key] for key in
                       ("snr_db", "bits", "blocks", "initial_nmse", "corrected_nmse")):
                    raise RuntimeError("The frozen CSI trajectory or validation coverage changed")
        schedules.append([entry.get("sampling_seed") for entry in after["history"]])
        results[variant] = dict(selected_epoch=selection["selected_epoch"], baseline=baseline,
            selected=selected, history=after["history"], changed_parameters=changed,
            frozen_parameters_unchanged=True, every_epoch_validation_nmse_unchanged=True,
            checkpoint_sha256=file_sha256(checkpoint), model_config=before["model_config"])
        write_json(output/"partial.json", dict(split="validation", test_access=False, variants=results))
    if schedules[0] != schedules[1]:
        raise RuntimeError("Paired SNR/frame schedules differ")
    guard.assert_unchanged()
    result = dict(schema="icr-final-detector-validation-v1", channel=channel, seed=seed,
        input_view="compensated", split="validation", test_access=False, complete=True,
        source_identity=guard.files, epochs=epochs, learning_rate=learning_rate,
        variants=results, selection="Strict per-SNR BER/NMSE bounds relative to each stage-B checkpoint",
        note="Validation-only feasibility diagnostic; no new test result or cross-seed claim.")
    write_json(output/"summary.json", result)
    save_figures(result, output)
    (output/"partial.json").unlink(missing_ok=True)
    for variant in results:
        (output/variant/"latest.pt").unlink(missing_ok=True)
    return result


def save_figures(result, output):
    """The NMSE curve averages complete-matrix linear NMSE before conversion to dB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        for metric in ("ber", "corrected_nmse"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
            for ax, (variant, data) in zip(axes, result["variants"].items()):
                for name, style in (("baseline", "o--"), ("selected", "x-")):
                    rows = data[name]["per_snr"]
                    snr = [row["snr_db"] for row in rows]
                    values = [row[metric] for row in rows]
                    if metric == "ber":
                        ax.semilogy(snr, [max(v, 1/r["bits"]) for v, r in zip(values, rows)],
                                    style, label=name)
                    else:
                        ax.plot(snr, [10*math.log10(v) for v in values], style, label=name)
                ax.set(title=f'{result["channel"]}: {variant} (validation)', xlabel="SNR (dB)",
                       ylabel="BER" if metric == "ber" else "Mean full-matrix NMSE (dB)", xticks=snr)
                ax.grid(True, which="both", alpha=.2)
                ax.legend()
            name = "validation_ber_snr" if metric == "ber" else "validation_nmse_snr"
            fig.savefig(Path(output)/(name+".png"), dpi=160)
            plt.close(fig)
