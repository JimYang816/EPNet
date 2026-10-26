"""PDDP HDF5 export with explicit D1/D2 axes and strict resume identity."""
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
from data_generation.src.pdd import (DATA_SCHEMA, ESTIMATION_MODE, FRAME_STRUCTURE, SYMBOL_COUNT,
                                     SYMBOL_LABELS, DATA_SYMBOL_INDICES, TRAINING_SYMBOL_INDICES, identities)
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.transmitter.ofdm import make_frame, carrier_masks, overhead

LOG = logging.getLogger(__name__)


def initialize(path, c, chain, prior, provenance):
    ids = identities(c, "test")
    nsnr, count, n = len(c["snr_db"]), len(ids), c["nfft"]
    h = h5py.File(path, "w")
    h.attrs.update(schema=DATA_SCHEMA, estimation_mode=ESTIMATION_MODE, frame_structure=FRAME_STRUCTURE, complete=False,
                   symbol_labels_json=json.dumps(SYMBOL_LABELS),
                   channel=c.get("channel", "NCS1"),
                   config_hash=digest(c), config_json=json.dumps(c), provenance_json=json.dumps(provenance),
                   overhead_json=json.dumps(overhead(c)), rx_noise_gain=chain.noise_gain,
                   snr_definition="Mean power of D1/D2 useful RX windows / expected filtered continuous real-RF AWGN; CP excluded",
                   actual_snr_definition="Individual D useful-window signal power / realized filtered noise power",
                   mmse_noise_definition="RF AWGN variance times per-carrier diagonal of unit_rf_noise_covariance",
                   csi_noise_definition="Thermal RF variance plus csi_model_error_ratio times noise-subtracted RX training power, normalized by public P/D transmit energies and RX noise gain",
                   csi_dtype="complex64", construction_dtype="complex128",
                   prior_json=json.dumps(dict(count=prior.model.count, length=prior.model.length,
                       bandwidth_hz=prior.model.bandwidth_hz, tail_ratio=prior.model.tail_ratio,
                       basis_sha256=hashlib.sha256(prior.model.basis.tobytes()).hexdigest(),
                       bandwidth_parameterization="absolute_hz",
                       block_size=prior.delay_block_size, empirical_weight=0, scale=1, interference_scale=1,
                       regularizer=prior.regularizer, feedback_iterations=0, fit_domain=prior.fit_domain,
                        csi_model_error_ratio=c.get("csi_model_error_ratio", 0),
                       source_records=list(prior.source_records))))
    specs = {
        "observations": ((nsnr, count, SYMBOL_COUNT, n), "complex64"),
        "clean_observations": ((count, SYMBOL_COUNT, n), "complex64"),
        "reference_csi": ((count, 2, n, n), "complex64"),
        "estimated_csi": ((nsnr, count, 2, n, n), "complex64"),
        "tx_symbols": ((count, SYMBOL_COUNT, n), "complex64"),
        "data_bits": ((count, 2, c["data_count"], 2), "uint8"),
        "rf_noise_variance": ((nsnr, count), "float64"),
        "csi_effective_rf_variance": ((nsnr, count), "float64"),
        "signal_power": ((count, 2), "float64"),
        "actual_snr_db": ((nsnr, count, 2), "float64"),
        "realized_rx_noise_variance": ((nsnr, count, 2), "float64"),
        "standard_noise_sha256": ((count,), "S64"),
        "sample_complete": ((count,), "bool"),
        "nmse_linear": ((nsnr, count, 2), "float64"),
        "noiseless_nmse_linear": ((count, 2), "float64"),
        "projection_nmse_linear": ((count, 2), "float64"),
        "generation_seconds": ((count,), "float64"),
    }
    for name, (shape, dtype) in specs.items():
        options = {}
        if name in ("reference_csi", "estimated_csi"):
            options.update(chunks=(1,)*(len(shape)-2)+(n, n), compression="gzip", compression_opts=1)
        h.create_dataset(name, shape, dtype=dtype, **options)
    h.create_dataset("diagnostics_json", (count,), dtype=h5py.string_dtype("utf-8"))
    for name, values in (("record", [r for r, s in ids]), ("start_seconds", [s for r, s in ids]),
                          ("snr_db", c["snr_db"]), ("data_symbol_indices", DATA_SYMBOL_INDICES),
                          ("training_symbol_indices", TRAINING_SYMBOL_INDICES),
                          ("unit_rf_noise_covariance", chain.noise_covariance),
                          ("known_training", make_frame(c, *ids[0]).known)):
        h.create_dataset(name, data=values)
    group = h.create_group("masks")
    for name, mask in carrier_masks(c).items():
        group.create_dataset(name, data=mask)
    return h


def generate(path, run, c, archive, chain, prior, backend, provenance):
    path, run = Path(path), Path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame0 = make_frame(c, *identities(c, "test")[0])
    masks = carrier_masks(c)
    prepared = JointEstimator(prior, frame0.known, masks, chain.noise_covariance, backend)
    if path.exists():
        h = h5py.File(path, "r+")
        if (h.attrs.get("schema") != DATA_SCHEMA or h.attrs.get("estimation_mode") != ESTIMATION_MODE
                or h.attrs.get("config_hash") != digest(c)
                or json.loads(h.attrs["provenance_json"]) != provenance):
            h.close()
            raise ValueError("PDDP dataset provenance/schema mismatch; never reuse historical output")
    else:
        h = initialize(path, c, chain, prior, provenance)
    try:
        if list(zip(h["record"][:], h["start_seconds"][:])) != identities(c, "test"):
            raise ValueError("PDDP test identities mismatch")
        for i, (record, start) in enumerate(identities(c, "test")):
            if h["sample_complete"][i]:
                continue
            begin = time.perf_counter()
            LOG.info("PDDP generation %d/%d %s_%03d @ %gs", i+1, len(h["record"]),
                     c.get("channel", "NCS1"), record, start)
            frame = make_frame(c, record, start)
            clean, clean_y, taps, reference, info = chain.frame(archive.load(record), start, frame)
            energy = np.sum(abs(taps)**2, axis=(1, 2))
            residual = np.sum(abs(taps[..., prior.model.length:])**2, axis=(1, 2))/energy
            support_passed = bool(np.all(residual <= 10**(c["support_error_db"]/10)))
            if not support_passed and c.get("model_mismatch_policy", "reject") != "diagnose":
                raise RuntimeError("Frozen PDDP support fails on a test P1/D1/D2/P2; stop without reselection")
            standard, variances, _ = chain.noise(clean, record, start)
            observations = clean_y[None]+np.sqrt(variances[:, None, None])*standard[None]
            estimates = prepared.estimate(observations, frame.known, masks, variances)
            noiseless = prepared.estimate(clean_y, frame.known, masks, [0])[0]
            ref_energy = np.sum(abs(reference)**2, axis=(1, 2))
            nmse = np.sum(abs(estimates-reference[None])**2, axis=(2, 3))/ref_energy
            projection_coeff = prior.model.coefficients(taps)[0]
            projection = []
            for s in (1, 2):
                error = taps[s].copy()
                error[:, :prior.model.length] -= prior.model.reconstruct(projection_coeff, s)
                projection.append(float(np.sum(abs(error)**2)/ref_energy[s-1]))
            powers = np.mean(abs(clean_y[list(DATA_SYMBOL_INDICES)])**2, axis=1)
            realized_noise = variances[:, None]*np.mean(abs(standard[list(DATA_SYMBOL_INDICES)])**2, axis=1)[None]
            h["observations"][:, i] = observations
            h["clean_observations"][i] = clean_y
            h["reference_csi"][i] = reference
            h["estimated_csi"][:, i] = estimates
            h["tx_symbols"][i], h["data_bits"][i] = frame.x, frame.bits
            h["rf_noise_variance"][:, i] = variances
            h["csi_effective_rf_variance"][:, i] = prepared.effective_noise_variances(observations, variances)
            h["signal_power"][i] = powers
            h["realized_rx_noise_variance"][:, i] = realized_noise
            h["actual_snr_db"][:, i] = 10*np.log10(powers[None]/realized_noise)
            h["standard_noise_sha256"][i] = hashlib.sha256(standard.tobytes()).hexdigest().encode()
            h["nmse_linear"][:, i] = nmse
            h["noiseless_nmse_linear"][i] = np.sum(abs(noiseless-reference)**2, axis=(1, 2))/ref_energy
            h["projection_nmse_linear"][i] = projection
            info.update(support_residuals_p1_d1_d2_p2=residual.tolist(), support_symbols=list(SYMBOL_LABELS),
                        support_passed=support_passed,
                        model_mismatch_policy=c.get("model_mismatch_policy", "reject"), frame_seed=list(frame.seed),
                        selected_observations_per_symbol=np.diff(prepared.offsets).tolist(),
                        joint_observation_count=int(prepared.offsets[-1]), joint_spectrum=prepared.diagnostics)
            h["diagnostics_json"][i] = json.dumps(info)
            h["generation_seconds"][i] = time.perf_counter()-begin
            h.flush()
            h["sample_complete"][i] = True
            h.flush()
            write_json(run/"generation_progress.json", dict(frames=int(h["sample_complete"][:].sum()),
                       data_symbols=int(h["sample_complete"][:].sum())*2, total_frames=len(h["record"])))
        h.attrs["complete"] = True
        h.flush()
    except BaseException as exc:
        write_json(run/"generation_failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        h.close()
    return path
