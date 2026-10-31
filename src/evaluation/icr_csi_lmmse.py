"""Isolate dataset, frozen ICR-corrected and oracle CSI with one exact LMMSE."""
from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE, load_model, validate_dataset
from src.algorithms.interface import EqualizerBatch
from src.algorithms.likelihood import make_whitener
from src.algorithms.MMSE.full_matrix import equalize_batch
from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.icr_mmse import precision_dtypes
from src.evaluation.fingerprints import (DataIdentity, generation_origin, numerical_environment,
                                         source_fingerprint)
from src.evaluation.icr_validation_compare import validate_candidate
from src.evaluation.inputs import prepare_batch
from src.evaluation.result_cache import FileLock
from src.evaluation.scoring import score_batch


LOG = logging.getLogger(__name__)
VARIANTS = ("initial", "corrected", "true")
CODE = INFERENCE_CODE + ("src.evaluation.icr_csi_lmmse", "src.evaluation.icr_validation_compare",
    "src.algorithms.interface", "src.algorithms.likelihood", "src.algorithms.MMSE.full_matrix",
    "src.evaluation.scoring", "data_generation.src.transmitter.ofdm:decisions")


def model_inputs(dataset, batch, block, precision):
    """Cast the same receiver-only arrays used by input_batches, on CPU."""
    complex_dtype, real_dtype = precision_dtypes(precision)
    tensor = lambda value: torch.as_tensor(value, dtype=complex_dtype)
    masks = {name: torch.as_tensor(value, dtype=torch.bool) for name, value in dataset._masks.items()}
    data, pilot = masks["data"], masks["pilot"]
    return dict(initial_channel=tensor(batch.channel), observations=tensor(batch.raw_observations),
        known=tensor(dataset._known[block+1]).expand(len(batch.variances), -1),
        data_mask=data, pilot_mask=pilot, null_mask=~(data | pilot),
        carrier_positions=torch.arange(dataset._config["nfft"]),
        rf_variance=torch.as_tensor(batch.variances, dtype=real_dtype),
        unit_covariance=tensor(batch.noise_covariance))


def channel_batch(batch, channel, known):
    """Recompute pilot cancellation and start a fresh likelihood cache per CSI."""
    channel = np.asarray(channel, dtype=np.complex128)
    if channel.shape != batch.channel.shape or not np.isfinite(channel).all():
        raise ValueError("CSI must be a finite complete matrix with the original shape")
    residual = batch.raw_observations - channel @ np.asarray(known, dtype=np.complex128)
    return EqualizerBatch(channel, batch.raw_observations, residual, batch.variances,
                          batch.data_indices, batch.noise_covariance)


def mean_nmse(channel, reference):
    energy = np.sum(np.abs(reference)**2, axis=(-2, -1))
    if not np.isfinite(reference).all() or np.any(energy <= 0):
        raise ValueError("Reference CSI must have finite positive matrix energy")
    return np.sum(np.abs(channel-reference)**2, axis=(-2, -1)) / energy


def plot_report(report, run):
    run = Path(run)
    snr = np.asarray(report["snr_db"])
    styles = {"initial": ("--s", "#D97725"), "corrected": ("-o", "#2864C5"),
              "true": ("-.^", "#008577")}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        fig.subplots_adjust(left=.11, right=.97, bottom=.27, top=.81)
        fig.suptitle(f"{report['channel']} validation: CSI comparison with full-matrix LMMSE",
                     fontsize=13, y=.96)
        ax.set_title(f"{report['frame_count']} identical frames, D1/D2; full noise covariance; complex128",
                     fontsize=10, pad=12)
        any_zero = False
        for name in VARIANTS:
            curve = report["curves"][name]
            values = np.asarray(curve["ber"])
            any_zero |= bool(np.any(values == 0))
            style, color = styles[name]
            ax.semilogy(snr, np.maximum(values, 1/np.asarray(curve["total_bits"])), style,
                color=color, linewidth=2, markersize=5.5, markerfacecolor="white", label=curve["label"])
        ax.set(xlabel="SNR (dB)", ylabel="BER", xticks=snr)
        ax.grid(True, which="both", alpha=.2)
        ax.legend(frameon=False)
        fig.text(.11, .115, "Same observations, data columns and thermal-noise covariance; pilots cancelled with each CSI.\n"
                 "Corrected CSI: frozen candidate's final output; true CSI: oracle diagnostic.", fontsize=8.4)
        if any_zero:
            fig.text(.11, .035, "Zero observed errors are shown at one-bit resolution (1 / total bits).", fontsize=8)
        for suffix in (".png", ".svg"):
            fig.savefig(run / ("ber_snr"+suffix), dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.9), layout="constrained")
        for name in ("initial", "corrected"):
            style, color = styles[name]
            ax.plot(snr, report["curves"][name]["mean_nmse_db"], style, color=color,
                    linewidth=2, markersize=5.5, label=report["curves"][name]["label"])
        ax.set(title=f"{report['channel']} validation: full-matrix CSI error", xlabel="SNR (dB)",
               ylabel="Mean NMSE (dB)", xticks=snr)
        ax.grid(True, alpha=.23)
        ax.legend(frameon=False)
        ax.text(.02, .03, r"True CSI: NMSE = 0 ($-\infty$ dB)", transform=ax.transAxes,
                ha="left", color=styles["true"][1], fontsize=9)
        for suffix in (".png", ".svg"):
            fig.savefig(run / ("nmse_snr"+suffix), dpi=220)
        plt.close(fig)


def export_csv(report, run):
    fields = ["snr_db", "total_bits"]
    fields += [name+suffix for name in VARIANTS for suffix in
               ("_errors", "_ber", "_mean_nmse_linear", "_mean_nmse_db")]
    with (Path(run)/"metrics.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for si, snr in enumerate(report["snr_db"]):
            row = dict(snr_db=snr, total_bits=report["curves"]["initial"]["total_bits"][si])
            for name in VARIANTS:
                for metric in ("errors", "ber", "mean_nmse_linear", "mean_nmse_db"):
                    row[name+"_"+metric] = report["curves"][name][metric][si]
            writer.writerow(row)


def evaluate(dataset_path, candidate_report, run, *, batch_size=2, lambda_h=.01):
    """Read a frozen validation candidate; rerun correction, then three CPU LMMSEs.

    Learned correction retains checkpoint precision/noise settings. All three
    direct solves use complex128, full thermal covariance and fresh pilot
    cancellation. Reference CSI enters only the explicitly labelled oracle arm
    and scoring, after receiver-only predictions have completed.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not math.isfinite(lambda_h) or lambda_h <= 0:
        raise ValueError("lambda_h must be finite and positive")
    dataset_path, candidate_report, run = Path(dataset_path), Path(candidate_report), Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".csi_lmmse.lock"):
        candidate = json.loads(candidate_report.read_text(encoding="utf-8"))
        checkpoint = Path(candidate["checkpoint"])
        guard = DataIdentity({"validation.h5": dataset_path, "model.h5": dataset_path.parent/"model.h5",
                              "icr_report": candidate_report, "icr_checkpoint": checkpoint})
        code = source_fingerprint(CODE)
        backend = DenseBackend("cpu")
        started = time.perf_counter()
        with CompactDataset(dataset_path, input_view="compensated") as dataset, torch.inference_mode():
            bits = validate_candidate(candidate, dataset, guard)
            model, payload = load_model(checkpoint, "cpu")
            validate_dataset(payload, dataset)
            controls = payload.get("identity", {}).get("controls", {})
            if (not payload.get("complete") or payload.get("stage") != "C"
                    or not payload.get("inference_correction_enabled", True)
                    or controls.get("lambda_h") != lambda_h
                    or payload["model_config"] != candidate["model_config"]):
                raise ValueError("Requires the complete frozen candidate with matching lambda_h/model configuration")
            for name in ("validation.h5", "model.h5"):
                if payload.get("identity", {}).get("data", {}).get(name) != guard.files[name]:
                    raise ValueError("Candidate checkpoint data identity differs: "+name)
            whitener = make_whitener(dataset._shared["unit_rf_noise_covariance"][:], backend)
            count, nsnr = dataset.frame_count, len(dataset.snr_db)
            errors = np.zeros((3, nsnr, count, 2), dtype=np.int64)
            nmse = np.zeros((3, nsnr, count, 2), dtype=np.float64)
            records = dataset._h["windows/record"][:][dataset._h["samples/window_index"][:]]
            processed = 0
            for si, snr in enumerate(dataset.snr_db):
                for block in range(2):
                    known = dataset._known[block+1]
                    for start in range(0, count, batch_size):
                        frames = list(range(start, min(start+batch_size, count)))
                        batch = prepare_batch(dataset, si, frames, block, backend)
                        inputs = model_inputs(dataset, batch, block, model.config["precision"])
                        output = model(**inputs)
                        corrected = output.channels[-1].dense().cpu().numpy().astype(np.complex128)
                        del output, inputs
                        channels = [batch.channel, corrected]
                        predictions = [equalize_batch(channel_batch(batch, h, known), backend, whitener=whitener)
                                       for h in channels]
                        # First reference access: correction and both receiver-only solves are finished.
                        reference = np.stack([dataset.reference_csi(fi)[block] for fi in frames])
                        channels.append(reference)
                        predictions.append(equalize_batch(channel_batch(batch, reference, known), backend,
                                                          whitener=whitener))
                        for vi, (channel, prediction) in enumerate(zip(channels, predictions)):
                            errors[vi, si, frames, block] = score_batch(dataset, frames, block, prediction)[0]
                            nmse[vi, si, frames, block] = mean_nmse(channel, reference)
                        processed += len(frames)
                        write_json(run/"progress.json", dict(complete=False, phase="csi_lmmse_validation",
                            snr_db=float(snr), d_block=block+1, processed_blocks=processed,
                            total_blocks=nsnr*count*2, elapsed_seconds=time.perf_counter()-started))
                    LOG.info("%s CSI/LMMSE SNR=%g D%d complete (%d/%d blocks)",
                             dataset.channel, snr, block+1, processed, nsnr*count*2)
            totals = errors.sum(axis=(2, 3))
            means = nmse.mean(axis=(2, 3))
            labels = ("Dataset CSI", f"Corrected CSI (lambda_H={lambda_h:g})", "True CSI (oracle)")
            curves = {}
            for vi, name in enumerate(VARIANTS):
                curves[name] = dict(label=labels[vi], identifier="full_matrix_lmmse", noise_mode="full",
                    precision="complex128/float64", errors=totals[vi].tolist(),
                    total_bits=[bits]*nsnr, ber=(totals[vi]/bits).tolist(),
                    mean_nmse_linear=means[vi].tolist(),
                    mean_nmse_db=[float(10*np.log10(value)) if value > 0 else None for value in means[vi]])
            by_record = []
            for record in np.unique(records):
                selected = records == record
                by_record.append(dict(record=int(record), frames=int(selected.sum()),
                    total_bits_per_snr=int(4*selected.sum()*len(dataset._indices)),
                    errors={name: errors[vi][:, selected].sum(axis=(1, 2)).tolist()
                            for vi, name in enumerate(VARIANTS)}))
            report = dict(schema="icr-csi-full-matrix-lmmse-validation-v1", complete=True,
                channel=dataset.channel, split="validation", input_view="compensated", frame_count=count,
                smoke=bool(candidate.get("smoke")) or bool(dataset._config.get("smoke")),
                snr_db=dataset.snr_db.tolist(), dataset=str(dataset_path.resolve()), lambda_h=lambda_h,
                checkpoint=str(checkpoint.resolve()), checkpoint_sha256=candidate["checkpoint_sha256"],
                candidate_report=str(candidate_report.resolve()), data_identity=guard.files,
                model_config=dict(model.config), curves=curves,
                overall_ber={name: float(totals[vi].sum()/(bits*nsnr)) for vi, name in enumerate(VARIANTS)},
                by_record=by_record, block_errors={name: errors[vi].tolist() for vi, name in enumerate(VARIANTS)},
                nmse_definition="Mean per-D-block full-matrix linear NMSE, then 10log10; true CSI is zero (null dB = -infinity)",
                comparison_notes=["Identical complete validation frames, SNRs, D1/D2, observations and bit labels.",
                    "One exact full-matrix LMMSE implementation, all receive rows/data columns, full thermal covariance.",
                    "Pilot cancellation and likelihood are recomputed independently for each CSI.",
                    "Corrected CSI is the final full channel from the frozen candidate at its saved precision and noise mode.",
                    "True CSI is an oracle diagnostic; it never enters learned correction or either receiver-only solve."],
                fitting=False, test_access=False, gpu_used=False, correction_rerun=True,
                numerical_environment=dict(numerical_environment(backend), torch=str(torch.__version__)),
                origin=generation_origin(run, backend, batch_size), code=code,
                batch_size=batch_size, elapsed_seconds=time.perf_counter()-started)
            guard.assert_unchanged()
            if source_fingerprint(CODE) != code:
                raise RuntimeError("CSI comparison code changed while evaluating")
            plot_report(report, run)
            export_csv(report, run)
            write_json(run/"metrics.json", report)
            write_json(run/"progress.json", dict(complete=True, phase="finished", processed_blocks=processed,
                total_blocks=nsnr*count*2, elapsed_seconds=time.perf_counter()-started))
            return report
