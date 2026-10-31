"""Compare a frozen ICR validation report with the full-matrix LMMSE baseline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.MMSE.full_matrix import equalize_batch
from src.algorithms.likelihood import make_whitener
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.fingerprints import DataIdentity, numerical_environment, source_fingerprint
from src.evaluation.inputs import prepare_batch
from src.evaluation.result_cache import FileLock
from src.evaluation.scoring import score_batch


LOG = logging.getLogger(__name__)
CODE = ("src.evaluation.icr_validation_compare", "src.evaluation.inputs",
        "src.evaluation.scoring", "src.algorithms.MMSE.full_matrix",
        "src.algorithms.likelihood", "src.common.numerics", "src.data.compact:CompactDataset",
        "data_generation.src.receiver.dpss:DPSSModel")


def validate_candidate(report, dataset, identity):
    if (not report.get("complete") or report.get("split") != "validation"
            or report.get("input_view") != "compensated" or dataset.split != "validation"
            or dataset.input_view != "compensated" or report.get("channel") != dataset.channel):
        raise ValueError("Comparison requires a complete compensated-CSI validation report and matching channel")
    for name in ("validation.h5", "model.h5"):
        if report.get("data_identity", {}).get(name) != identity.files[name]:
            raise ValueError("Candidate and LMMSE data identity differ: " + name)
    if report.get("checkpoint_sha256") != identity.files["icr_checkpoint"]["sha256"]:
        raise ValueError("Candidate checkpoint changed after validation")
    if report.get("snr_db") != dataset.snr_db.tolist():
        raise ValueError("Candidate and LMMSE SNR grids differ")
    rows = report.get("per_snr", [])
    curve = report.get("curves", {}).get("ICR-MMSE-Net", {})
    bits = 4 * dataset.frame_count * len(dataset._indices)
    if (len(rows) != len(dataset.snr_db) or curve.get("total_bits") != [bits] * len(rows)
            or len(curve.get("errors", [])) != len(rows) or len(curve.get("ber", [])) != len(rows)):
        raise ValueError("Candidate must cover identical full validation frames and bit counts")
    for index, row in enumerate(rows):
        if (row["snr_db"] != float(dataset.snr_db[index]) or row["blocks"] != 2 * dataset.frame_count
                or row["total_bits"] != bits or row["bit_errors"][-1] != curve["errors"][index]
                or not np.isclose(curve["ber"][index], curve["errors"][index] / bits, rtol=0, atol=1e-15)):
            raise ValueError("Candidate must cover identical full validation frames and bit counts")
    return bits


def plot_report(report, run):
    run = Path(run)
    snr = np.asarray(report["snr_db"])
    colors = {"ICR-MMSE-Net": "#2864C5", "Full-matrix LMMSE": "#D97725"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(8.4, 5.5))
        fig.subplots_adjust(left=.105, right=.97, bottom=.24, top=.82)
        fig.suptitle(f"{report['channel']} validation: ICR-MMSE-Net vs full-matrix LMMSE", fontsize=14, y=.96)
        ax.set_title("Same validation frames, observations and initial dataset CSI", fontsize=10, pad=13)
        any_zero = False
        for name, style in (("ICR-MMSE-Net", "-o"), ("Full-matrix LMMSE", "--s")):
            curve = report["curves"][name]
            values = np.asarray(curve["ber"])
            floor = 1 / np.asarray(curve["total_bits"])
            any_zero |= bool(np.any(values == 0))
            ax.semilogy(snr, np.maximum(values, floor), style, color=colors[name],
                        linewidth=2, markersize=5.5, markerfacecolor="white", label=name)
        ax.set(xlabel="SNR (dB)", ylabel="BER", xticks=snr)
        ax.grid(True, which="major", alpha=.23)
        ax.legend(frameon=False)
        fig.text(.105, .12,
                 "ICR: complex64, diagonal noise approximation, learned CSI correction.\n"
                 "LMMSE: complex128, full noise covariance, original dataset CSI.", fontsize=8.5, color="#5D6878")
        if any_zero:
            fig.text(.105, .035, "Zero observed errors are shown at one-bit resolution (1 / total bits).", fontsize=8)
        for suffix in (".png", ".svg"):
            fig.savefig(run / ("ber_snr" + suffix), dpi=220)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
        ax.plot(snr, report["nmse"]["initial_mean_nmse_db"], "--o", color="#8390A0", label="Dataset CSI / LMMSE input")
        ax.plot(snr, report["nmse"]["corrected_mean_nmse_db"], "-s", color="#008577", label="ICR corrected CSI")
        ax.set(title=f"{report['channel']} validation CSI", xlabel="SNR (dB)",
               ylabel="Mean full-matrix NMSE (dB)", xticks=snr)
        ax.grid(True, alpha=.23)
        ax.legend(frameon=False)
        for suffix in (".png", ".svg"):
            fig.savefig(run / ("nmse_snr" + suffix), dpi=220)
        plt.close(fig)


def evaluate(dataset_path, candidate_report, run, *, batch_size=2):
    """CPU-only diagnostic; no fitting, test access, ICR rerun or shared cache writes."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be positive")
    dataset_path, candidate_report, run = Path(dataset_path), Path(candidate_report), Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / ".validation_comparison.lock"):
        candidate = json.loads(candidate_report.read_text(encoding="utf-8"))
        guard = DataIdentity({"validation.h5": dataset_path, "model.h5": dataset_path.parent / "model.h5",
                              "icr_report": candidate_report, "icr_checkpoint": candidate["checkpoint"]})
        code = source_fingerprint(CODE)
        backend = DenseBackend("cpu")
        started = time.perf_counter()
        with CompactDataset(dataset_path, input_view="compensated") as dataset:
            bits_per_snr = validate_candidate(candidate, dataset, guard)
            whitener = make_whitener(dataset._shared["unit_rf_noise_covariance"][:], backend)
            count, nsnr = dataset.frame_count, len(dataset.snr_db)
            errors = np.zeros((nsnr, count, 2), dtype=np.int64)
            records = dataset._h["windows/record"][:][dataset._h["samples/window_index"][:]]
            processed = 0
            for si, snr in enumerate(dataset.snr_db):
                for block in range(2):
                    for start in range(0, count, batch_size):
                        frames = list(range(start, min(start + batch_size, count)))
                        # Prediction uses receiver fields only; labels are read by score_batch afterward.
                        batch = prepare_batch(dataset, si, frames, block, backend)
                        result = equalize_batch(batch, backend, whitener=whitener)
                        batch_errors, _ = score_batch(dataset, frames, block, result)
                        errors[si, frames, block] = batch_errors
                        processed += len(frames)
                        write_json(run / "progress.json", dict(complete=False, phase="validation_lmmse",
                            snr_db=float(snr), d_block=block + 1, processed_blocks=processed,
                            total_blocks=nsnr * count * 2, elapsed_seconds=time.perf_counter() - started))
                    LOG.info("%s full-matrix LMMSE SNR=%g D%d complete (%d/%d blocks)",
                             dataset.channel, snr, block + 1, processed, nsnr * count * 2)
            totals = errors.sum(axis=(1, 2))
            curve = dict(identifier="full_matrix_lmmse", precision="complex128/float64", noise_mode="full",
                         errors=totals.tolist(), total_bits=[bits_per_snr] * nsnr,
                         ber=(totals / bits_per_snr).tolist())
            by_record = []
            for record in np.unique(records):
                selected = records == record
                by_record.append(dict(record=int(record), frames=int(selected.sum()),
                    errors=errors[:, selected].sum(axis=(1, 2)).tolist(),
                    total_bits=int(4 * selected.sum() * len(dataset._indices))))
            rows = candidate["per_snr"]
            result = dict(schema="icr-validation-lmmse-comparison-v1", complete=True,
                channel=dataset.channel, split="validation", input_view="compensated", frame_count=count,
                smoke=bool(candidate.get("smoke")) or bool(dataset._config.get("smoke")),
                snr_db=dataset.snr_db.tolist(), dataset=str(dataset_path.resolve()),
                candidate_report=str(candidate_report.resolve()), data_identity=guard.files,
                checkpoint_sha256=candidate["checkpoint_sha256"], icr_model_config=candidate["model_config"],
                curves={"ICR-MMSE-Net": candidate["curves"]["ICR-MMSE-Net"], "Full-matrix LMMSE": curve},
                overall_ber={"ICR-MMSE-Net": float(sum(candidate["curves"]["ICR-MMSE-Net"]["errors"]) /
                    (bits_per_snr * nsnr)), "Full-matrix LMMSE": float(totals.sum() / (bits_per_snr * nsnr))},
                nmse=dict(initial_mean_nmse_linear=[r["mean_nmse_linear"][0] for r in rows],
                    corrected_mean_nmse_linear=[r["mean_nmse_linear"][-1] for r in rows],
                    initial_mean_nmse_db=[r["mean_nmse_db"][0] for r in rows],
                    corrected_mean_nmse_db=[r["mean_nmse_db"][-1] for r in rows],
                    definition="Mean per-block full-matrix linear NMSE, then 10log10; reused frozen ICR report"),
                lmmse_by_record=by_record, lmmse_block_errors=errors.tolist(),
                comparison_notes=["Same full validation frames, D1/D2, SNR grid, observations and dataset CSI source.",
                    "ICR retains its trained CSI correction, precision and noise mode; LMMSE uses full colored noise.",
                    "End-to-end detector comparison; does not isolate iterative versus direct linear solution."],
                fitting=False, test_access=False, icr_report_reused=True,
                numerical_environment=numerical_environment(backend), code=code,
                batch_size=batch_size, elapsed_seconds=time.perf_counter() - started)
            guard.assert_unchanged()
            if source_fingerprint(CODE) != code:
                raise RuntimeError("Comparison code changed while evaluating")
            plot_report(result, run)
            write_json(run / "metrics.json", result)
            write_json(run / "progress.json", dict(complete=True, phase="finished", processed_blocks=processed,
                total_blocks=nsnr * count * 2, elapsed_seconds=time.perf_counter() - started))
            return result
