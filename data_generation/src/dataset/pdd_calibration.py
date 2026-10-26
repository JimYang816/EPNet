"""Independent PDDP prior fitting and validation of fixed shared DPSS parameters."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time

import h5py
import numpy as np

from data_generation.src.config import digest
from src.common.experiment import write_json
from data_generation.src.pdd import (CACHE_SCHEMA, PRIOR_SCHEMA, VALIDATION_SCHEMA, ESTIMATION_MODE,
                                     SYMBOL_COUNT, SYMBOL_LABELS, identities)
from data_generation.src.receiver.chain import full_frame_kernels
from data_generation.src.receiver.dpss import build_model, DPSSModel, FULL_FRAME_DOMAIN
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.transmitter.ofdm import make_frame, carrier_masks

LOG = logging.getLogger(__name__)


def collect(path, c, archive, chain, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum, n = c["delay_length"], c["nfft"]
    h = h5py.File(path, "r+" if path.exists() else "w")
    try:
        if "schema" in h.attrs:
            if (h.attrs["schema"] != CACHE_SCHEMA or h.attrs.get("estimation_mode") != ESTIMATION_MODE
                    or h.attrs.get("config_hash") != digest(c)
                    or h.attrs.get("archive_sha256") != manifest["archive_sha256"]):
                raise ValueError("Incompatible PDDP calibration cache; historical caches are forbidden")
        else:
            h.attrs.update(schema=CACHE_SCHEMA, estimation_mode=ESTIMATION_MODE, config_hash=digest(c),
                           archive_sha256=manifest["archive_sha256"], complete=False)
            for split in ("prior", "validation"):
                ids = identities(c, split)
                g = h.create_group(split)
                count = len(ids)
                g.create_dataset("record", data=[r for r, s in ids])
                g.create_dataset("start_seconds", data=[s for r, s in ids])
                g.create_dataset("complete", (count,), dtype=bool)
                g.create_dataset("taps", (count, SYMBOL_COUNT, n, maximum), dtype=complex,
                                 chunks=(1, 1, n, maximum), compression="gzip", compression_opts=1)
                if split == "prior":
                    g.create_dataset("full_frame_taps", (count, SYMBOL_COUNT*(n+c["cp"]), maximum), dtype=complex,
                                     chunks=(1, n, maximum), compression="gzip", compression_opts=1)
                for name in ("clean_y", "standard_noise"):
                    g.create_dataset(name, (count, SYMBOL_COUNT, n), dtype=complex)
                g.create_dataset("rf_base_variance", (count,), dtype=float)
                g.create_dataset("total_energy", (count, SYMBOL_COUNT), dtype=float)
                g.create_dataset("support_residuals", (count, SYMBOL_COUNT), dtype=float)
                g.create_dataset("diagnostics", (count,), dtype=h5py.string_dtype("utf-8"))
        for split, number in (("prior", 0), ("validation", 1)):
            g = h[split]
            ids = identities(c, split)
            if list(zip(g["record"][:], g["start_seconds"][:])) != ids:
                raise ValueError("PDDP calibration identities mismatch")
            for i, (record, start) in enumerate(ids):
                if g["complete"][i]:
                    continue
                LOG.info("PDDP calibration %s %d/%d %s_%03d @ %gs", split, i+1, len(ids),
                         c.get("channel", "NCS1"), record, start)
                frame = make_frame(c, record, start, number)
                sounding = archive.load(record)
                clean, y, taps, _, info = chain.frame(sounding, start, frame)
                standard, _, power = chain.noise(clean, record, start, number)
                energy = np.sum(abs(taps)**2, axis=(1, 2))
                if np.any(energy <= 0):
                    raise RuntimeError("Zero-energy calibration channel")
                g["taps"][i] = taps[..., :maximum]
                g["total_energy"][i] = energy
                g["support_residuals"][i] = np.sum(abs(taps[..., maximum:])**2, axis=(1, 2))/energy
                g["clean_y"][i], g["standard_noise"][i] = y, standard
                g["rf_base_variance"][i] = power/chain.noise_gain
                if split == "prior":
                    op, _ = chain.operators(sounding, start-c["guard_samples"]/c["fs_baseband"])
                    full = full_frame_kernels(op, c, maximum)
                    positions = np.concatenate([s*(n+c["cp"])+c["cp"]-c["fft_advance"]+np.arange(n) for s in range(SYMBOL_COUNT)])
                    sampled = full[positions].reshape(SYMBOL_COUNT, n, maximum)
                    target = taps[..., :maximum]
                    info["full_frame_window_model_error"] = float(np.sum(abs(sampled-target)**2)/max(np.sum(abs(target)**2), 1e-300))
                    if c.get("model_mismatch_policy", "reject") != "diagnose":
                        np.testing.assert_allclose(sampled, target, rtol=1e-12, atol=1e-15)
                    g["full_frame_taps"][i] = full
                g["diagnostics"][i] = json.dumps(info)
                h.flush()
                g["complete"][i] = True
                h.flush()
        h.attrs["complete"] = True
        h.flush()
    finally:
        h.close()
    return path


def save_pdd_prior(path, prior, c, validation, manifest):
    partial = Path(path).with_suffix(".partial.h5")
    with h5py.File(partial, "w") as h:
        h.attrs.update(complete=True, config_hash=digest(c), config_json=json.dumps(c),
                       schema=PRIOR_SCHEMA, estimation_mode=ESTIMATION_MODE, fit_domain=prior.fit_domain,
                       empirical_weight=prior.empirical_weight, scale=prior.scale,
                       regularizer=prior.regularizer, source_records_json=json.dumps(prior.source_records),
                       bandwidth_hz=prior.model.bandwidth_hz, tail_ratio=prior.model.tail_ratio,
                       length=prior.model.length, archive_sha256=manifest["archive_sha256"],
                       delay_block_size=prior.delay_block_size, interference_scale=prior.interference_scale,
                       feedback_iterations=prior.feedback_iterations, validation_json=json.dumps(validation))
        for name, value in (("basis", prior.model.basis), ("coefficients", prior.coefficients),
                            ("temporal_covariance", prior.temporal_covariance), ("delay_covariance", prior.delay_covariance)):
            h.create_dataset(name, data=value)
    partial.replace(path)


def load_pdd_prior(path, c, manifest):
    with h5py.File(path, "r") as h:
        if (h.attrs.get("schema") != PRIOR_SCHEMA or h.attrs.get("estimation_mode") != ESTIMATION_MODE
                or h.attrs.get("archive_sha256") != manifest["archive_sha256"]):
            raise ValueError("Incompatible PDDP frozen prior; historical priors cannot be reused")
        if not h.attrs.get("complete") or h.attrs["config_hash"] != digest(c):
            raise ValueError("Frozen prior configuration mismatch")
        if h.attrs.get("fit_domain") != FULL_FRAME_DOMAIN:
            raise ValueError("Frozen prior fit domain mismatch")
        model = DPSSModel(h["basis"][:], float(h.attrs["bandwidth_hz"]), float(h.attrs["tail_ratio"]), int(h.attrs["length"]), c)
        if len(model.basis) != SYMBOL_COUNT*(c["nfft"]+c["cp"]):
            raise ValueError("Frozen prior must cover the full PDDP frame")
        if (model.bandwidth_hz != c["dpss_half_bandwidth_hz"]
                or model.count != c["dpss_expected_count"] or model.length != c["delay_length"]
                or model.tail_ratio != c["dpss_tail_ratio"]
                or int(h.attrs["delay_block_size"]) != c["delay_block_size"]):
            raise ValueError("Frozen prior does not match the fixed shared DPSS controls")
        prior = FrozenPrior(model, h["coefficients"][:], h["temporal_covariance"][:], h["delay_covariance"][:],
                            float(h.attrs["empirical_weight"]), float(h.attrs["scale"]), float(h.attrs["regularizer"]),
                            tuple(json.loads(h.attrs["source_records_json"])), FULL_FRAME_DOMAIN,
                            int(h.attrs.get("delay_block_size", 0)), float(h.attrs.get("interference_scale", 1.0)),
                            int(h.attrs.get("feedback_iterations", 0)))
    if (prior.empirical_weight, prior.scale, prior.interference_scale, prior.feedback_iterations) != (0, 1, 1, 0):
        raise ValueError("PDDP frozen prior fixed controls mismatch")
    if tuple(c["prior_records"]) != prior.source_records or prior.fit_domain != FULL_FRAME_DOMAIN:
        raise ValueError("PDDP prior source/domain mismatch")
    return prior


def coefficient_nmse(model, coefficients, truth, energy):
    if coefficients.shape != (len(truth), model.count, model.length):
        raise ValueError("Expected shared full-frame batch x Q x L coefficients")
    values = []
    for s in (1, 2):
        predicted = model.reconstruct(coefficients, s)
        target = truth[:, s, :, :model.length]
        omitted = np.maximum(energy[:, s]-np.sum(abs(target)**2, axis=(1, 2)), 0)
        values.append((np.sum(abs(predicted-target)**2, axis=(1, 2))+omitted)/energy[:, s])
    return np.stack(values, axis=1)


def calibrate(path, run, c, archive, chain, backend, manifest):
    run = Path(run)
    frozen = run/"frozen_prior.h5"
    if frozen.exists():
        return load_pdd_prior(frozen, c, manifest)
    collect(path, c, archive, chain, manifest)
    with h5py.File(path, "r") as h:
        support = np.concatenate([h[s+"/support_residuals"][:] for s in ("prior", "validation")])
        maximum = float(support.max())
        train = h["prior/full_frame_taps"][:]
        valid = h["validation/taps"][:]
        energy = h["validation/total_energy"][:]
        variance = h["validation/rf_base_variance"][:]/10**(c["validation_snr_db"]/10)
        observations = h["validation/clean_y"][:]+np.sqrt(variance[:, None, None])*h["validation/standard_noise"][:]
    report = dict(schema=VALIDATION_SCHEMA, estimation_mode=ESTIMATION_MODE, config_hash=digest(c),
                  archive_sha256=manifest["archive_sha256"], complete=False,
                  maximum_support_residual=maximum, support_symbols=list(SYMBOL_LABELS),
                  support_exceeds_limit=bool(maximum > 10**(c["support_error_db"]/10)),
                  csi_model_error_ratio=c["csi_model_error_ratio"],
                  model_mismatch_policy=c["model_mismatch_policy"],
                  rule="Fixed shared W/tail/Q/L/B; independent prior refit and 30 dB validation, no reselection")
    report_path = run/"validation.json"
    write_json(report_path, report)
    if report["support_exceeds_limit"] and c["model_mismatch_policy"] != "diagnose":
        raise RuntimeError("Fixed PDDP support fails on prior/validation; stop without reselection")
    begin = time.perf_counter()
    model = build_model(c)
    prior = FrozenPrior.fit_coefficients(model, model.coefficients_full_frame(train), 0, 1, c["prior_records"],
                                         c["relative_regularizer"], delay_block_size=c["delay_block_size"])
    frame = make_frame(c, *identities(c, "validation")[0], split=1)
    masks = carrier_masks(c)
    prepared = JointEstimator(prior, frame.known, masks, chain.noise_covariance, backend)
    estimated = prepared.estimate_coefficients(observations, frame.known, masks, variance)
    values = coefficient_nmse(model, estimated, valid, energy)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Nonfinite validation NMSE")
    parameters = dict(bandwidth_hz=model.bandwidth_hz, tail_ratio=model.tail_ratio, count=model.count,
                      length=model.length, block_size=prior.delay_block_size,
                      basis_sha256=hashlib.sha256(model.basis.tobytes()).hexdigest())
    report.update(complete=True, fixed_parameters=parameters, mean_linear_nmse=float(values.mean()),
                  nmse_per_window_d=values.tolist(), mean_d1_d2=values.mean(axis=0).tolist(),
                  seconds=time.perf_counter()-begin, spectrum=prepared.diagnostics)
    LOG.info("Fixed PDDP validation %.6f dB in %.1fs", 10*np.log10(values.mean()), report["seconds"])
    write_json(report_path, report)
    write_json(run/"frozen_parameters.json", parameters)
    save_pdd_prior(frozen, prior, c, report, manifest)
    return prior
