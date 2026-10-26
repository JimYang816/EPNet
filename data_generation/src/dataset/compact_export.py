"""Resumable compact PDDP export, groupwise compensation and training-only statistics."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing

import h5py
import numpy as np

from data_generation.src.config import digest
from data_generation.src.dataset.compact_protocol import (
    SCHEMA, MODEL_SCHEMA, SPLITS, SPLIT_CODES, physical_identities, sample_identities, target_nmse_db)
from data_generation.src.dataset.compact_math import coefficient_nmse, reconstruct, projection_power
from data_generation.src.receiver.pdd import JointEstimator
from data_generation.src.transmitter.ofdm import make_frame, carrier_masks
from src.common.experiment import write_json
from src.data.uncertainty import blocks_from_prior
from data_generation.src.dataset.physical_windows import initialize_worker, window_results

LOG = logging.getLogger(__name__)


def physical_hash(clean, standard, bits, variances):
    result = hashlib.sha256()
    for array in (clean, standard, bits, variances):
        result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def _dataset(h, name, shape, dtype, *, chunk=False):
    options = dict(chunks=(1,)+tuple(shape[1:]), compression="lzf", shuffle=True) if chunk else {}
    return h.create_dataset(name, shape=shape, dtype=dtype, **options)


def prepare_model(path, c, chain, prior, backend, provenance):
    """Save and reuse the original estimator spectrum, not a dense QL by QL matrix."""
    known = make_frame(c, 1, 2).known
    masks = carrier_masks(c)
    identity = digest(dict(config=c, provenance=provenance))
    path = Path(path)
    if path.exists():
        with h5py.File(path, "r") as h:
            if (h.attrs.get("schema") != MODEL_SCHEMA or h.attrs.get("identity") != identity
                    or not h.attrs.get("prepared")):
                raise ValueError("Shared model identity/protocol mismatch")
            prepared = JointEstimator.__new__(JointEstimator)
            prepared.prior, prepared.backend = prior, backend
            prepared.known, prepared.masks = known, masks
            prepared.model_error_ratio = float(c["csi_model_error_ratio"])
            prepared.noise_gain = float(h["estimator"].attrs["noise_gain"])
            prepared.rows = [h[f"estimator/rows/{s}"][:] for s in range(4)]
            prepared.offsets = h["estimator/offsets"][:]
            prepared.ridge = float(h["estimator"].attrs["ridge"])
            prepared.values = backend.array(h["estimator/values"][:])
            prepared.projected = backend.array(h["estimator/projected"][:])
            prepared.gain = backend.array(h["estimator/gain"][:])
            prepared.diagnostics = json.loads(h.attrs["estimator_diagnostics"])
        return prepared, identity
    prepared = JointEstimator(prior, known, masks, chain.noise_covariance, backend)
    partial = path.with_suffix(".partial.h5")
    with h5py.File(partial, "w") as h:
        h.attrs.update(schema=MODEL_SCHEMA, identity=identity, prepared=False, complete=False,
                       config_json=json.dumps(c), provenance_json=json.dumps(provenance),
                       covariance_semantics="receiver_model_covariance_proxy",
                       reference_assisted_construction=True,
                       estimator_diagnostics=json.dumps(prepared.diagnostics))
        h.create_dataset("basis", data=prior.model.basis)
        h.create_dataset("known", data=known)
        h.create_dataset("unit_rf_noise_covariance", data=chain.noise_covariance)
        h.create_dataset("snr_db", data=c["snr_db"])
        bit_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.uint8)
        from data_generation.src.transmitter.ofdm import qpsk
        h.create_dataset("constellation/bits", data=bit_pairs)
        h.create_dataset("constellation/symbols", data=qpsk(bit_pairs))
        for name, mask in masks.items():
            h.create_dataset("masks/"+name, data=mask)
        for i, (start, stop, temporal, delay) in enumerate(blocks_from_prior(prior)):
            g = h.create_group(f"prior/blocks/{i:03d}")
            g.attrs.update(start=start, stop=stop)
            g.create_dataset("temporal", data=temporal)
            g.create_dataset("delay", data=delay)
        h["prior"].attrs["source_records_json"] = json.dumps(list(prior.source_records))
        h["prior"].attrs["fit_domain"] = prior.fit_domain
        for name, array in (("values", prepared.values), ("projected", prepared.projected), ("gain", prepared.gain)):
            value = backend.numpy(array)
            opts = dict(chunks=(min(1024, value.shape[0]), min(128, value.shape[1])), compression="lzf") if value.ndim == 2 else {}
            h.create_dataset("estimator/"+name, data=value, **opts)
        # This sufficient statistic makes per-sample trace evaluation O(observation size).
        gain = backend.numpy(prepared.gain)
        h.create_dataset("estimator/gain_column_energy", data=np.sum(abs(gain)**2, axis=0))
        h.create_dataset("estimator/offsets", data=prepared.offsets)
        for s, rows in enumerate(prepared.rows):
            h.create_dataset(f"estimator/rows/{s}", data=rows)
        h["estimator"].attrs["ridge"] = prepared.ridge
        h["estimator"].attrs["noise_gain"] = prepared.noise_gain
        h.attrs["prepared"] = True
    partial.replace(path)
    return prepared, identity


def initialize_split(path, c, split, identity):
    windows, samples = physical_identities(c, split), sample_identities(c, split)
    nw, nf, ns, n = len(windows), len(samples), len(c["snr_db"]), c["nfft"]
    q, length = c["dpss_expected_count"], c["delay_length"]
    h = h5py.File(path, "w")
    h.attrs.update(schema=SCHEMA, model_identity=identity, split=split, channel=c["channel"],
                   config_hash=digest(c), complete=False, raw_complete=False, compensated=False,
                   model_file="model.h5", reference_assisted_construction=True,
                   observation_reconstruction="clean_observations + sqrt(rf_noise_variance) * standard_noise",
                   compensation_formula="H_ref + alpha * (H_raw - H_ref)",
                   target_formula="a-b*(1-exp(-snr/10))/(1-exp(-3))",
                   construction_precision="complex128; stored full reference and raw coefficients complex64",
                   snr_definition="D1/D2 mean useful-window RX power / expected filtered real-RF thermal noise")
    for name, array in (("windows/record", [r for r, _ in windows]), ("windows/start_seconds", [t for _, t in windows]),
                        ("samples/window_index", [w for w, _, _ in samples]),
                        ("samples/payload_repeat", [r for _, r, _ in samples]),
                        ("samples/noise_repeat", [r for _, _, r in samples]), ("snr_db", c["snr_db"])):
        h.create_dataset(name, data=array)
    specs = {
        "windows/complete": ((nw,), bool, False),
        "windows/reference_sha256": ((nw,), "S64", False),
        "windows/reference_csi": ((nw, 2, n, n), "complex64", True),
        "windows/reference_projection": ((nw, q, length), "complex128", True),
        "windows/projection_residual_power": ((nw, 2, n), "float64", False),
        "samples/complete": ((nf,), bool, False),
        "samples/clean_observations": ((nf, 4, n), "complex128", True),
        "samples/standard_noise": ((nf, 4, n), "complex128", True),
        "samples/bits": ((nf, 2, c["data_count"], 2), "uint8", True),
        "samples/model_residual": ((nf, 2, n), "complex128", True),
        "samples/signal_power": ((nf, 2), "float64", False),
        "samples/rf_noise_variance": ((ns, nf), "float64", False),
        "samples/effective_rf_variance": ((ns, nf), "float64", False),
        "samples/actual_snr_db": ((ns, nf, 2), "float64", False),
        "samples/realized_rx_noise_power": ((ns, nf, 2), "float64", False),
        "samples/raw_coefficients": ((ns, nf, q, length), "complex64", False),
        "samples/raw_nmse_linear": ((ns, nf, 2), "float64", False),
        "samples/standard_noise_sha256": ((nf,), "S64", False),
        "samples/physical_sha256": ((nf,), "S64", False),
        "samples/payload_seed": ((nf, 6), "uint64", False),
        "samples/noise_seed": ((nf, 6), "uint64", False),
    }
    for name, (shape, dtype, chunk) in specs.items():
        if name == "samples/raw_coefficients":
            h.create_dataset(name, shape, dtype=dtype, chunks=(1, 1, q, length), compression="lzf", shuffle=True)
        else:
            _dataset(h, name, shape, dtype, chunk=chunk)
    h.create_dataset("windows/diagnostics_json", (nw,), dtype=h5py.string_dtype("utf-8"))
    return h


def generate_split(path, run, c, split, archive, chain, prepared, identity, *, stop_after=None, pool=None, prefetch=1):
    """Commit markers follow flushed writes; replay only incomplete sample transactions."""
    path = Path(path)
    h = h5py.File(path, "r+") if path.exists() else initialize_split(path, c, split, identity)
    try:
        if (h.attrs.get("schema") != SCHEMA or h.attrs.get("model_identity") != identity
                or h.attrs.get("config_hash") != digest(c) or h.attrs.get("split") != split):
            raise ValueError("Split identity/protocol mismatch; never overwrite historical data")
        if h.attrs.get("raw_complete"):
            return True
        model, code = prepared.prior.model, SPLIT_CODES[split]
        masks = carrier_masks(c)
        samples = sample_identities(c, split)
        generated = 0
        windows = physical_identities(c, split)
        jobs = []
        for wi, (record, start) in enumerate(windows):
            members = [(i, payload, noise) for i, (w, payload, noise) in enumerate(samples)
                       if w == wi and not h["samples/complete"][i]]
            if members:
                jobs.append((wi, split, record, start, members))
        for wi, (reference, taps, physical_samples) in window_results(jobs, c, archive, chain, pool=pool, prefetch=prefetch):
            record, start = windows[wi]
            for i, clean, y, diagnostics in physical_samples:
                _, payload, noise = samples[i]
                LOG.info("%s %s window %d/%d record %03d @ %gs repeat %d", c["channel"], split, wi+1,
                         len(physical_identities(c, split)), record, start, payload)
                frame = make_frame(c, record, start, code, payload_repeat=payload)
                if not h["windows/complete"][wi]:
                    # Save full coupling before any DPSS projection or compensation.
                    h["windows/reference_csi"][wi] = reference
                    theta = model.coefficients(taps)[0]
                    h["windows/reference_projection"][wi] = theta
                    persisted_reference = h["windows/reference_csi"][wi].astype(complex)
                    h["windows/reference_sha256"][wi] = hashlib.sha256(h["windows/reference_csi"][wi].tobytes()).hexdigest().encode()
                    projected = reconstruct(model, theta)
                    h["windows/projection_residual_power"][wi] = projection_power(
                        persisted_reference, projected, frame.known, np.flatnonzero(masks["data"]))
                    h["windows/diagnostics_json"][wi] = json.dumps(diagnostics)
                    h.flush()
                    h["windows/complete"][wi] = True
                    h.flush()
                reference = h["windows/reference_csi"][wi].astype(complex)
                standard, variances, _ = chain.noise(clean, record, start, code, noise_repeat=noise)
                observations = y[None]+np.sqrt(variances[:, None, None])*standard[None]
                coefficients = prepared.estimate_coefficients(observations, frame.known, masks, variances).astype("complex64")
                s = h["samples"]
                s["clean_observations"][i], s["standard_noise"][i] = y, standard
                s["raw_coefficients"][:, i] = coefficients
                s["raw_nmse_linear"][:, i] = coefficient_nmse(model, coefficients, reference)
                s["bits"][i] = frame.bits
                s["rf_noise_variance"][:, i] = variances
                s["effective_rf_variance"][:, i] = prepared.effective_noise_variances(observations, variances)
                powers = np.mean(abs(y[[1, 2]])**2, axis=1)
                realized = variances[:, None]*np.mean(abs(standard[[1, 2]])**2, axis=1)[None]
                s["signal_power"][i] = powers
                s["realized_rx_noise_power"][:, i] = realized
                s["actual_snr_db"][:, i] = 10*np.log10(powers[None]/realized)
                s["model_residual"][i] = y[[1, 2]]-np.einsum("dij,dj->di", reference, frame.x[[1, 2]], optimize=True)
                s["standard_noise_sha256"][i] = hashlib.sha256(standard.tobytes()).hexdigest().encode()
                s["physical_sha256"][i] = physical_hash(y, standard, frame.bits, variances).encode()
                s["payload_seed"][i] = frame.seed
                s["noise_seed"][i] = (c["seed"], code, record, round(start*1000), 1, noise)
                h.flush()
                s["complete"][i] = True
                h.flush()
                generated += 1
                write_json(Path(run)/f"{split}_progress.json", dict(complete_frames=int(s["complete"][:].sum()),
                           total_frames=len(samples), total_physical_windows=len(physical_identities(c, split))))
                if stop_after is not None and generated >= stop_after:
                    return False
        h.attrs["raw_complete"] = bool(np.all(h["samples/complete"][:]) and np.all(h["windows/complete"][:]))
        h.flush()
        return bool(h.attrs["raw_complete"])
    finally:
        h.close()


def calibrate_training(model_path, train_path, c):
    """Fit gamma and residual statistics using only the committed training split."""
    with h5py.File(model_path, "r+") as model, h5py.File(train_path, "r") as h:
        if not h.attrs.get("raw_complete") or h.attrs.get("split") != "train":
            raise ValueError("Calibration needs complete raw training data")
        if model.attrs.get("training_calibration_complete"):
            return
        ns, nf = h["samples/raw_coefficients"].shape[:2]
        numerator, denominator = np.zeros(ns), np.zeros(ns)
        prior_trace = sum(np.trace(g["temporal"][:]).real*np.trace(g["delay"][:]).real
                          for g in model["prior/blocks"].values())
        energies = model["estimator/gain_column_energy"][:]
        values = model["estimator/values"][:]
        ridge = model["estimator"].attrs["ridge"]
        for i in range(nf):
            wi = int(h["samples/window_index"][i])
            truth = h["windows/reference_projection"][wi]
            raw = h["samples/raw_coefficients"][:, i].astype(complex)
            numerator += np.sum(abs(raw-truth[None])**2, axis=(1, 2))
            variance = h["samples/effective_rf_variance"][:, i]
            traces = prior_trace-np.sum(energies[None]/(values[None]+variance[:, None]+ridge), axis=1)
            if np.any(traces <= 0) or not np.all(np.isfinite(traces)):
                raise ValueError("Nonpositive raw covariance proxy trace")
            denominator += traces
        if "training_calibration" in model:
            del model["training_calibration"]
        g = model.create_group("training_calibration")
        g.attrs.update(source_split="train", source_records_json=json.dumps(c["learning_dataset"]["records"]["train"]),
                       frame_count=nf, projection_domain="least_squares_all_four_fft_windows_same_full_frame_dpss",
                       semantics="receiver_model_proxy; CSI projection residual and physical model residual are separate")
        for name, value in (("gamma", numerator/denominator), ("coefficient_error_energy_sum", numerator),
                            ("proxy_trace_sum", denominator),
                            ("projection_residual_power", np.mean(h["windows/projection_residual_power"][:], axis=0)),
                            ("model_residual_power", np.mean(abs(h["samples/model_residual"][:])**2, axis=0)),
                            ("window_record", h["windows/record"][:]), ("window_start_seconds", h["windows/start_seconds"][:])):
            g.create_dataset(name, data=value)
        model.flush()
        model.attrs["training_calibration_complete"] = True
        model.flush()


def compensate(path, c):
    with h5py.File(path, "r+") as h:
        if not h.attrs.get("raw_complete"):
            raise ValueError("Cannot compensate incomplete raw data")
        target = target_nmse_db(c["channel"], c["snr_db"])
        if (h.attrs.get("compensated") and "compensation/target_nmse_db" in h
                and np.array_equal(h["compensation/target_nmse_db"][:], target)):
            return
        raw = h["samples/raw_nmse_linear"][:].mean(axis=(1, 2))
        if np.any(raw <= 0) or not np.all(np.isfinite(raw)):
            raise ValueError("Invalid raw group NMSE")
        alpha = np.sqrt(10**(target/10)/raw)
        h.attrs.update(compensated=False, complete=False)
        h.flush()
        if "compensation" in h:
            del h["compensation"]
        g = h.create_group("compensation")
        g.create_dataset("alpha", data=alpha)
        g.create_dataset("target_nmse_db", data=target)
        g.create_dataset("raw_mean_nmse_linear", data=raw)
        g.attrs.update(grouping="channel x split x snr; arithmetic mean of per-D linear NMSE",
                       reference_assisted=True, adjusted_using_ber=False)
        h.flush()
        h.attrs.update(compensated=True, complete=True)
        h.flush()


def generate(directory, run, c, archive, chain, prior, backend, provenance, *, workers=1, threads=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory/"model.h5"
    prepared, identity = prepare_model(model_path, c, chain, prior, backend, provenance)
    if not 1 <= workers <= 8:
        raise ValueError("Physical window workers must be 1..8")
    context = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=initialize_worker, initargs=(c, str(archive.path), threads))
               if workers > 1 else nullcontext(None))
    with context as pool:
        for split in SPLITS:
            path = directory/f"{split}.h5"
            generate_split(path, run, c, split, archive, chain, prepared, identity, pool=pool, prefetch=workers*2)
            if split == "train":
                calibrate_training(model_path, path, c)
            compensate(path, c)
    with h5py.File(model_path, "r+") as h:
        h.attrs["complete"] = True
    # The estimator is large; release it before the independent disk acceptance pass.
    del prepared
    return directory
