"""Four independent equalizers with content-addressed, per-detector reuse."""
from __future__ import annotations

from contextlib import ExitStack
import logging
from pathlib import Path
import time

import numpy as np

from src.algorithms.MMSE.single_tap import equalize_batch as single_tap_batch
from src.algorithms.robust_ep.real import robust_whitener
from src.algorithms.likelihood import likelihood_batch, make_whitener, backend_real_likelihood
from src.common.experiment import write_json
from src.data.compact import CompactDataset
from src.evaluation import fingerprints as fp
from src.evaluation.covariance import calibrate, training_count, dpss_predict, shrink_second_moment
from src.evaluation.inputs import prepare_batch
from src.evaluation.nmse import evaluate_nmse
from src.evaluation.comparison_plots import plot_ber, publish_report, replot
from src.evaluation.registry import ALGORITHMS, CURVES, resolve_parameters, resolve_force
from src.evaluation.result_cache import CacheEntry, DEFAULT_CACHE
from src.evaluation.scoring import score_batch

LOG = logging.getLogger(__name__)
SCHEMA = "compact-four-equalizers-ber-v2"


def test_batch(dataset, si, frames, d, backend):
    """Compatibility view of the old batch helper; evaluation uses prepare_batch."""
    batch = prepare_batch(dataset, si, frames, d, backend)
    result = single_tap_batch(batch, backend)
    return batch.channel[..., batch.data_indices], batch.cancelled_observations, batch.variances, result.symbols


def _covariance(ds, cache, data_identity, environment, origin, backend, batch_size):
    data = data_identity.files
    identity = dict(schema="train-covariance-v1", data={name: data[name] for name in ("train.h5", "model.h5")},
                    input_view=ds.input_view, snr_db=ds.snr_db.tolist(),
                    code=fp.source_fingerprint(fp.MODEL_CODE+("src.common.numerics", "src.evaluation.covariance",
                                                              "data_generation.src.transmitter.ofdm:qpsk")),
                    environment=environment)
    nsnr, n = len(ds.snr_db), ds._config["nfft"]
    count = training_count(ds)
    with CacheEntry(cache, "covariance", identity,
                    dict(error_covariance_unit=((nsnr, 2, n, n), "complex128"),
                         shrinkage=((nsnr, 2), "float64")),
                    (nsnr,), origin, limits={"shrinkage": (0, 1)}, data_guard=data_identity.assert_unchanged) as entry:
        if not np.all(entry.complete):
            calibrate(ds, entry, backend, batch_size)
        covariance, shrinkage = entry.read("error_covariance_unit"), entry.read("shrinkage")
        provenance = entry.provenance()
        entry.publish()
    return covariance, shrinkage, count, provenance


def _nmse(ds, cache, data_identity, environment, origin):
    data = data_identity.files
    identity = dict(schema="test-full-csi-linear-mean-nmse-v1", input_view=ds.input_view,
                    data={name: data[name] for name in ("test.h5", "model.h5")},
                    code=fp.source_fingerprint(fp.MODEL_CODE+("src.evaluation.nmse",
                        "data_generation.src.dataset.compact_math:coefficient_nmse",
                        "data_generation.src.dataset.compact_math:matrix_to_taps")),
                    environment=environment)
    nsnr = len(ds.snr_db)
    with CacheEntry(cache, "nmse", identity, dict(mean_nmse_db=((nsnr,), "float64")), (nsnr,), origin,
                    data_guard=data_identity.assert_unchanged) as entry:
        if not np.all(entry.complete):
            evaluate_nmse(ds, entry, data)
        values, provenance = entry.read("mean_nmse_db").tolist(), entry.provenance()
        entry.publish()
    return dict(split="test", mean_nmse_db=values,
                definition="10*log10(mean_frame,D(||H_est-H_ref||_F^2/||H_ref||_F^2))",
                provenance=provenance)


def evaluate(dataset_path, run, backend, *, batch_size=8, iterations=10, damping=.2, epsilon=5e-7,
             algorithm_parameters=None, force=(), cache_dir=None, input_view="compensated"):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    parameters = resolve_parameters(algorithm_parameters, iterations=iterations, damping=damping, epsilon=epsilon)
    forced = resolve_force(force)
    run, cache = Path(run), Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    run.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with CompactDataset(dataset_path, input_view=input_view) as ds:
        if ds.split != "test":
            raise ValueError("BER comparison requires the test split")
        nsnr, count, ndata = len(ds.snr_db), ds.frame_count, len(ds._indices)
        if not ds._config.get("smoke") and (count != 600 or ndata != 512):
            raise ValueError("Formal comparison requires all 600 test frames and 512 data carriers")
        LOG.info("%s verifying SHA256 of test, model and training data", ds.channel)
        identity = fp.DataIdentity({name: ds.path.parent/name for name in ("test.h5", "model.h5", "train.h5")})
        environment = fp.numerical_environment(backend)
        origin = fp.generation_origin(run, backend, batch_size)
        errors_cov, shrinkage, train_count, covariance_source = _covariance(
            ds, cache, identity, environment, origin, backend, batch_size)
        report_curves = {}
        rejected = {}
        with ExitStack() as stack:
            entries = {}
            shape = (nsnr, count, 2)
            for algorithm in ALGORITHMS:
                scientific_identity = dict(schema=SCHEMA, algorithm=algorithm.identifier, input_view=ds.input_view,
                    data={name: identity.files[name] for name in ("test.h5", "model.h5")},
                    parameters=parameters[algorithm.identifier],
                    code=fp.source_fingerprint(fp.INPUT_CODE+fp.SCORING_CODE+(algorithm.module,)+algorithm.helpers),
                    environment=environment)
                if algorithm.identifier == "robust_ep":
                    scientific_identity["covariance_key"] = covariance_source["cache_key"]
                entries[algorithm.identifier] = stack.enter_context(CacheEntry(
                    cache, "algorithms/"+algorithm.identifier, scientific_identity,
                    dict(bit_errors=(shape, "int64"), rejected_sites=(shape, "int64")), shape, origin,
                    force=algorithm.identifier in forced,
                    limits={"bit_errors": (0, 2*ndata), "rejected_sites": (0, None)},
                    data_guard=identity.assert_unchanged))
                entry = entries[algorithm.identifier]
                LOG.info("%s %s: %s (%d/%d existing blocks)", ds.channel, algorithm.label,
                         "reuse" if entry.reused else "compute/resume", entry.resumed_blocks, nsnr*count*2)
            thermal = None
            if any(not np.all(entries[a.identifier].complete) for a in ALGORITHMS[1:]):
                thermal = make_whitener(ds._shared["unit_rf_noise_covariance"][:], backend)
            for si, snr in enumerate(ds.snr_db):
                for d in range(2):
                    robust = None
                    for start in range(0, count, batch_size):
                        stop = min(start+batch_size, count)
                        pending = [(a, [fi for fi in range(start, stop)
                                        if not entries[a.identifier].complete[si, fi, d]]) for a in ALGORITHMS]
                        pending = [(a, frames) for a, frames in pending if frames]
                        if not pending:
                            continue
                        # Reuse input reconstruction when pending detectors need the same frames.
                        prepared = {}
                        for algorithm, frames in pending:
                            token = tuple(frames)
                            if token not in prepared:
                                prepared[token] = prepare_batch(ds, si, frames, d, backend)
                            batch = prepared[token]
                            extra = {}
                            if algorithm.identifier == "robust_ep":
                                if robust is None:
                                    noise = ds._shared["unit_rf_noise_covariance"][:]
                                    robust = robust_whitener(noise, errors_cov[si, d], snr, backend)
                                extra["whitener"] = robust
                            elif algorithm.identifier != "single_tap_mmse":
                                extra["whitener"] = thermal
                            result = algorithm.equalize(batch, backend, **extra, **parameters[algorithm.identifier])
                            errors, rejected_sites = score_batch(ds, frames, d, result)
                            for j, fi in enumerate(frames):
                                entries[algorithm.identifier].commit((si, fi, d),
                                    bit_errors=errors[j], rejected_sites=rejected_sites[j])
                    LOG.info("%s SNR=%g D%d finished", ds.channel, snr, d+1)
                write_json(run/"progress.json", dict(channel=ds.channel, snr_db=float(snr),
                    algorithms={a.identifier: dict(completed_blocks=int(entries[a.identifier].complete.sum()),
                        total_blocks=nsnr*count*2) for a in ALGORITHMS}, elapsed_seconds=time.perf_counter()-started))
            bits = count*2*ndata*2
            identity.assert_unchanged()
            for algorithm in ALGORITHMS:
                entry = entries[algorithm.identifier]
                totals = entry.read("bit_errors").sum(axis=(1, 2))
                report_curves[algorithm.label] = dict(identifier=algorithm.identifier, errors=totals.tolist(),
                    total_bits=[bits]*nsnr, ber=(totals/bits).tolist(), parameters=parameters[algorithm.identifier],
                    provenance=entry.provenance())
                rejected[algorithm.identifier] = int(entry.read("rejected_sites").sum())
                entry.publish()
        nmse = _nmse(ds, cache, identity, environment, origin)
        identity.assert_unchanged()
        report = dict(schema=SCHEMA, complete=True, channel=ds.channel, snr_db=ds.snr_db.tolist(),
            frame_count=count, bits_per_snr=bits, data_blocks_per_snr=count*2, curves=report_curves,
            parameters=parameters, dataset=str(Path(dataset_path).resolve()), data_identity=identity.files,
            nmse=nmse, report_generation=origin, numerical_environment=environment,
            input_view=ds.input_view, csi="raw receiver CSI" if ds.input_view == "raw" else "accepted compensated CSI",
            robust_covariance=dict(source="train only", frames=train_count, shrinkage=shrinkage.tolist(),
                method="gain-normalized residual second moment, diagonal shrinkage",
                model_residual_included=True, test_labels_used=False, provenance=covariance_source),
            single_tap="raw data-carrier observations; diagonal CSI; thermal noise",
            matrix_methods="all receive rows, all data-column ICI, estimated known-pilot cancellation",
            ep="independent ordinary and robust implementations of real-valued equations 23-34",
            paper_doi="10.1109/TVT.2017.2683527",
            rejected_sites=[rejected["ep"], rejected["robust_ep"]])
        publish_report(run, report)
        return report
