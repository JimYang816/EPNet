"""Independent on-disk acceptance; no equalizer, BER computation or training."""
from __future__ import annotations

import hashlib
import html
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from data_generation.src.config import digest
from data_generation.src.dataset.compact_protocol import (
    SCHEMA, SPLITS, SPLIT_CODES, validate_plan, physical_identities, sample_identities, target_nmse_db)
from data_generation.src.dataset.compact_math import coefficient_nmse, reconstruct, projection_power
from data_generation.src.dataset.compact_export import physical_hash
from data_generation.src.transmitter.ofdm import make_frame
from src.common.experiment import write_json, file_sha256
from src.data import CompactDataset, data_system

LOG = logging.getLogger(__name__)


def _assert_close(actual, expected, message, *, rtol=2e-10, atol=1e-12):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(message)


def accept(directory, run, c, *, plots=True, hash_files=True):
    directory, run = Path(directory), Path(run)
    counts = validate_plan(c)
    report = dict(protocol=SCHEMA, channel=c["channel"], smoke=bool(c.get("smoke")), complete=False,
                  reference_assisted_construction=True, config_hash=digest(c), counts=counts, splits={})
    with h5py.File(directory/"model.h5", "r") as shared:
        if "nmse_retarget_json" in shared.attrs:
            report["nmse_retarget"] = json.loads(shared.attrs["nmse_retarget_json"])
        calibration = shared["training_calibration"]
        if (calibration.attrs["source_split"] != "train"
                or json.loads(calibration.attrs["source_records_json"]) != c["learning_dataset"]["records"]["train"]):
            raise ValueError("Training calibration source leakage")
        train_numerator = np.zeros(len(c["snr_db"]))
        train_denominator = np.zeros_like(train_numerator)
        model_residual_sum = np.zeros((2, c["nfft"]))
        projection_residual_sum = np.zeros_like(model_residual_sum)
        blocks = shared["prior/blocks"]
        prior_trace = sum(np.trace(g["temporal"][:]).real*np.trace(g["delay"][:]).real for g in blocks.values())
        eigenvalues = shared["estimator/values"][:]
        column_energy = shared["estimator/gain_column_energy"][:]
        ridge = float(shared["estimator"].attrs["ridge"])
        noise_gain = np.trace(shared["unit_rf_noise_covariance"][:]).real/c["nfft"]
        for split in SPLITS:
            LOG.info("Disk acceptance %s %s", c["channel"], split)
            # Dataset acceptance audits the persisted historical compensation;
            # experiment readers choose their own (default raw) CSI view.
            with CompactDataset(directory/f"{split}.h5", input_view="compensated") as ds:
                h, model = ds._h, ds._model
                if h.attrs["config_hash"] != digest(c):
                    raise ValueError("Acceptance configuration mismatch")
                expected_windows, expected_samples = physical_identities(c, split), sample_identities(c, split)
                if (list(zip(h["windows/record"][:], h["windows/start_seconds"][:])) != expected_windows
                        or list(zip(h["samples/window_index"][:], h["samples/payload_repeat"][:], h["samples/noise_repeat"][:])) != expected_samples):
                    raise ValueError("Split count/window/repeat identities mismatch")
                ns, nf = len(c["snr_db"]), ds.frame_count
                recomputed = np.zeros((ns, nf, 2))
                alpha = h["compensation/alpha"][:]
                for wi, (record, start) in enumerate(expected_windows):
                    reference_stored = h["windows/reference_csi"][wi]
                    if hashlib.sha256(reference_stored.tobytes()).hexdigest().encode() != h["windows/reference_sha256"][wi]:
                        raise ValueError("Reference CSI changed during compensation")
                    reference = reference_stored.astype(complex)
                    theta = h["windows/reference_projection"][wi]
                    if split == "train":
                        power = projection_power(reference, reconstruct(model, theta), shared["known"][:], ds._indices)
                        _assert_close(power, h["windows/projection_residual_power"][wi], "Projection residual mismatch")
                        projection_residual_sum += power
                    payload_hashes, noise_hashes = set(), set()
                    for fi, (window, payload, noise) in enumerate(expected_samples):
                        if window != wi:
                            continue
                        clean = h["samples/clean_observations"][fi]
                        standard = h["samples/standard_noise"][fi]
                        bits = h["samples/bits"][fi]
                        variances = h["samples/rf_noise_variance"][:, fi]
                        if physical_hash(clean, standard, bits, variances).encode() != h["samples/physical_sha256"][fi]:
                            raise ValueError("Physical observations/labels/thermal noise changed during compensation")
                        expected = make_frame(c, record, start, SPLIT_CODES[split], payload_repeat=payload)
                        if not np.array_equal(expected.bits, bits):
                            raise ValueError("Payload seed/label mismatch")
                        if not np.array_equal(expected.seed, h["samples/payload_seed"][fi]):
                            raise ValueError("Payload stream identity mismatch")
                        expected_noise_seed = (c["seed"], SPLIT_CODES[split], record, round(start*1000), 1, noise)
                        if not np.array_equal(expected_noise_seed, h["samples/noise_seed"][fi]):
                            raise ValueError("Noise stream identity mismatch")
                        payload_hashes.add(hashlib.sha256(bits.tobytes()).hexdigest())
                        noise_hashes.add(hashlib.sha256(standard.tobytes()).hexdigest())
                        if hashlib.sha256(standard.tobytes()).hexdigest().encode() != h["samples/standard_noise_sha256"][fi]:
                            raise ValueError("Noise fingerprint mismatch")
                        powers = np.mean(abs(clean[[1, 2]])**2, axis=1)
                        _assert_close(variances, powers.mean()/noise_gain/10**(np.array(c["snr_db"])/10), "SNR variance mismatch")
                        realized = variances[:, None]*np.mean(abs(standard[[1, 2]])**2, axis=1)[None]
                        _assert_close(realized, h["samples/realized_rx_noise_power"][:, fi], "Realized noise mismatch")
                        _assert_close(10*np.log10(powers[None]/realized), h["samples/actual_snr_db"][:, fi], "Actual SNR mismatch")
                        raw = h["samples/raw_coefficients"][:, fi].astype(complex)
                        raw_nmse = coefficient_nmse(model, raw, reference)
                        _assert_close(raw_nmse, h["samples/raw_nmse_linear"][:, fi], "Persisted raw NMSE mismatch")
                        recomputed[:, fi] = raw_nmse*alpha[:, None]**2
                        residual = clean[[1, 2]]-np.einsum("dij,dj->di", reference, expected.x[[1, 2]], optimize=True)
                        _assert_close(residual, h["samples/model_residual"][fi], "Physical model residual mismatch")
                        if split == "train":
                            train_numerator += np.sum(abs(raw-theta[None])**2, axis=(1, 2))
                            variance = h["samples/effective_rf_variance"][:, fi]
                            train_denominator += prior_trace-np.sum(column_energy[None]/(eigenvalues[None]+variance[:, None]+ridge), axis=1)
                            model_residual_sum += abs(residual)**2
                    repeats = c["learning_dataset"]["repeats"][split]
                    if len(payload_hashes) != repeats or len(noise_hashes) != repeats:
                        raise ValueError("Repeated payloads/noise are not independent streams")
                means = 10*np.log10(recomputed.mean(axis=(1, 2)))
                target = target_nmse_db(c["channel"], c["snr_db"])
                _assert_close(h["compensation/target_nmse_db"][:], target, "Persisted NMSE target mismatch")
                _assert_close(h["compensation/raw_mean_nmse_linear"][:],
                              h["samples/raw_nmse_linear"][:].mean(axis=(1, 2)), "Raw mean NMSE mismatch")
                if np.max(abs(means-target)) > .02:
                    raise ValueError("On-disk mean NMSE misses target by more than 0.02 dB")
                differences = np.diff(means)
                if np.any(differences >= 0) or np.any(np.diff(differences) < -1e-8):
                    raise ValueError("NMSE must decrease monotonically with diminishing slope")
                if means.min() < target.min()-.02 or means.max() > target.max()+.02:
                    raise ValueError("NMSE outside target interval")
                # Explicit full-matrix reconstruction checks complement the exact Gram identity used for all samples.
                for si in range(ns):
                    for fi in sorted(set((0, nf-1))):
                        inputs = ds.model_input(si, fi)
                        audit = ds.audit(si, fi)
                        _assert_close(audit["true_nmse_linear"], recomputed[si, fi], "Direct reconstructed full CSI NMSE mismatch", rtol=2e-8)
                        forbidden = {"reference_csi", "raw_csi", "compensation_alpha", "true_nmse_linear", "true_error", "raw_coefficients"}
                        if forbidden & set(inputs) or forbidden & set(inputs["uncertainty"]):
                            raise ValueError("Audit fields leaked into default model inputs")
                        labels = ds.labels(fi)
                        known = inputs["known"][[1, 2]]
                        tx = known.copy()
                        tx[:, ds._indices] = labels["symbols"]
                        if not np.array_equal(labels["constellation"][labels["classes"]], labels["symbols"]):
                            raise ValueError("Class/constellation mapping mismatch")
                        y_data, h_data = data_system(inputs)
                        _assert_close(h_data, inputs["csi"][..., ds._indices], "Data column extraction mismatch")
                        _assert_close(y_data, inputs["observations"][[1, 2]]-np.einsum("dij,dj->di", inputs["csi"], known), "Pilot cancellation mismatch")
                        # Cancellation uses the compensated CSI, not stale raw-CSI derived inputs.
                        _assert_close(y_data-np.einsum("dij,dj->di", h_data, labels["symbols"]),
                                      inputs["observations"][[1, 2]]-np.einsum("dij,dj->di", inputs["csi"], tx),
                                      "Pilot/data decomposition mismatch")
                report["splits"][split] = dict(mean_nmse_db=means.tolist(), mean_nmse_db_display=[f"{v:.2f}" for v in means],
                         target_nmse_db=target.tolist(), maximum_target_error_db=float(np.max(abs(means-target))),
                         alpha=h["compensation/alpha"][:].tolist(), checked_frames=nf,
                         checked_d_blocks=ns*nf*2, full_matrix_spot_checks=ns*len(set((0, nf-1))))
        _assert_close(train_numerator, calibration["coefficient_error_energy_sum"][:], "Training numerator mismatch")
        _assert_close(train_denominator, calibration["proxy_trace_sum"][:], "Training trace denominator mismatch")
        _assert_close(train_numerator/train_denominator, calibration["gamma"][:], "Frozen gamma mismatch")
        _assert_close(model_residual_sum/counts["train"]["frames_per_snr"], calibration["model_residual_power"][:], "Training physical residual statistic mismatch")
        _assert_close(projection_residual_sum/counts["train"]["physical_windows"], calibration["projection_residual_power"][:], "Training CSI residual statistic mismatch")
        report["gamma"] = calibration["gamma"][:].tolist()
        report["training_statistics_source"] = dict(split="train", records=c["learning_dataset"]["records"]["train"])
        report["basis_sha256"] = hashlib.sha256(shared["basis"][:].tobytes()).hexdigest()
        report["delay_length"] = c["delay_length"]
        report["provenance"] = json.loads(shared.attrs["provenance_json"])
    files = {}
    for path in sorted(directory.glob("*.h5")):
        logical = []
        with h5py.File(path, "r") as h:
            h.visititems(lambda name, item: logical.append(item.size*item.dtype.itemsize) if isinstance(item, h5py.Dataset) else None)
        entry = dict(bytes=path.stat().st_size, uncompressed_array_bytes=sum(logical))
        if hash_files:
            entry["sha256"] = file_sha256(path)
        files[path.name] = entry
    report.update(complete=True, files=files, snr_db=c["snr_db"], total_file_bytes=sum(v["bytes"] for v in files.values()),
                  checks=dict(record_disjointness=True, nonoverlapping_windows=True, counts=True, independent_repeats=True,
                              full_csi_reconstruction=True, nmse_all_samples=True, physical_invariance=True,
                              labels_carriers_pilot_cancellation=True, training_only_statistics=True,
                              separate_model_and_csi_residuals=True, model_view_excludes_audit=True))
    write_json(run/"acceptance.json", report)
    write_json(directory/"manifest.json", report)
    if plots:
        plot_report(directory, run, c, report)
    return report


def plot_report(directory, run, c, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
    for split in SPLITS:
        ax.plot(c["snr_db"], report["splits"][split]["mean_nmse_db"], "o-", label=split)
    ax.plot(c["snr_db"], target_nmse_db(c["channel"], c["snr_db"]), "k--", label="target")
    ax.set(xlabel="SNR (dB)", ylabel="Mean NMSE (dB)", title=c["channel"]+" | mean(linear per-D NMSE)")
    ax.grid(True, alpha=.3)
    ax.legend()
    fig.savefig(run/"nmse_snr.png", dpi=160)
    plt.close(fig)
    figures = ["nmse_snr.png"]
    with CompactDataset(Path(directory)/"test.h5", input_view="compensated") as ds:
        for si, snr in enumerate(c["snr_db"]):
            audit = ds.audit(si, 0)
            fields = (audit["reference_csi"], audit["compensated_csi"], audit["true_error"])
            peak = float(np.max(abs(fields[0])**2))
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
            for col, (field, title) in enumerate(zip(fields, ("Reference CSI", "Compensated CSI", "CSI error"))):
                for d in range(2):
                    db = 10*np.log10(np.maximum(abs(field[d])**2/peak, 1e-8))
                    im = axes[d, col].imshow(db, origin="lower", vmin=-80, vmax=0, aspect="auto", cmap="viridis")
                    axes[d, col].set(title=f"D{d+1} | {title}", xlabel="TX carrier", ylabel="RX carrier")
            ident = audit["identity"]
            fig.suptitle(f"{c['channel']} record {ident['record']:03d} @ {ident['start_seconds']:g}s repeat 0 | SNR {snr} dB")
            fig.colorbar(im, ax=axes, label="10 log10(|H|² / fixed reference peak²), dB")
            name = f"csi_test_snr_{snr:02d}.png"
            fig.savefig(run/name, dpi=130)
            plt.close(fig)
            figures.append(name)
    lines = ["<!doctype html><html lang='zh'><meta charset='utf-8'><title>数据验收</title>",
             "<style>body{max-width:1100px;margin:2em auto;font-family:sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style>",
             f"<h1>{html.escape(c['channel'])} 数据验收 {'CPU smoke' if c.get('smoke') else '正式数据'}</h1>",
             "<p>协议 watermark-pddp-compact-v2；使用参考辅助 CSI 构造。协方差为接收机模型代理。未运行均衡器、训练或 BER。</p>",
             "<p>NMSE 为逐 D 块线性误差的算术平均再转 dB。所有热力图使用同一固定测试窗口的参考峰值，色标 −80 至 0 dB。</p>",
             "<pre>"+html.escape(json.dumps({k: report[k] for k in ("counts", "splits", "gamma", "checks", "total_file_bytes")}, indent=2, ensure_ascii=False))+"</pre>"]
    lines += [f"<img src='{name}' alt='{name}'>" for name in figures]
    (run/"report.html").write_text("\n".join(lines)+"</html>", encoding="utf-8")
