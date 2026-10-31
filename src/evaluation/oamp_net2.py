"""Complete OAMP-Net2 test scoring with independently reusable result blocks."""
from __future__ import annotations

from pathlib import Path
import logging

import h5py
import numpy as np
import torch

from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.likelihood import make_whitener
from src.algorithms.oamp_net2.checkpoint import load_model, validate_dataset, INFERENCE_CODE
from src.algorithms.oamp_net2.inference import equalize_batch
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.evaluation import comparison, fingerprints as fp
from src.evaluation.comparison_plots import publish_report
from src.evaluation.inputs import prepare_batch
from src.evaluation.result_cache import CacheEntry, DEFAULT_CACHE, FileLock
from src.evaluation.scoring import score_batch

LOG = logging.getLogger(__name__)


def score_oamp_net2(dataset, frames, d, result):
    errors, _ = score_batch(dataset, frames, d, result)
    truth = dataset._h["samples/bits"][frames, d]
    symbol_errors = (decisions(result.symbols) != truth).any(axis=-1).sum(axis=-1)
    return errors, symbol_errors


def _evaluate(dataset_path, checkpoint, run, backend, *, batch_size=8, cache_dir=None,
              algorithm_parameters=None, input_view="compensated", force=False):
    if (input_view != "compensated" or not isinstance(batch_size, int)
            or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError("OAMP-Net2 evaluation requires compensated CSI and a positive batch size")
    checkpoint, run = Path(checkpoint), Path(run)
    checkpoint_sha = file_sha256(checkpoint)
    model, payload = load_model(checkpoint, backend.device)
    with CompactDataset(dataset_path) as ds:
        validate_dataset(payload, ds)
        if ds.split != "test":
            raise ValueError("OAMP-Net2 formal evaluation requires the test split")
        if not ds._config.get("smoke") and (ds.frame_count != 600 or len(ds._indices) != 512):
            raise ValueError("Formal OAMP-Net2 evaluation requires all 600 frames and 512 data carriers")
        for split in ("train", "validation"):
            with h5py.File(ds.path.parent/(split+".h5"), "r") as other:
                if np.intersect1d(ds._h["windows/record"][:], other["windows/record"][:]).size:
                    raise ValueError("Physical-record leakage between dataset splits")
    baseline = comparison.evaluate(dataset_path, run/"baselines", backend, batch_size=batch_size,
        algorithm_parameters=algorithm_parameters, cache_dir=cache_dir, input_view=input_view)
    with CompactDataset(dataset_path) as ds:
        validate_dataset(payload, ds)
        guard = fp.DataIdentity({name: ds.path.parent/name for name in ("test.h5", "model.h5")})
        if payload["identity"]["data"]["model.h5"] != guard.files["model.h5"]:
            raise ValueError("OAMP-Net2 checkpoint shared model content changed since training")
        if baseline["data_identity"] != guard.files:
            raise RuntimeError("Baseline and OAMP-Net2 data identities disagree")
        shape = (len(ds.snr_db), ds.frame_count, 2)
        ndata = len(ds._indices)
        scientific_identity = dict(schema="oamp-net2-test-v1", input_view=input_view,
            data=guard.files, checkpoint_sha256=checkpoint_sha,
            code=fp.source_fingerprint(INFERENCE_CODE+fp.SCORING_CODE+("src.evaluation.oamp_net2:score_oamp_net2",)),
            environment=dict(fp.numerical_environment(backend), torch=str(torch.__version__)))
        origin = fp.generation_origin(run, backend, batch_size)
        with CacheEntry(Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE, "algorithms/oamp_net2", scientific_identity,
                dict(bit_errors=(shape, "int64"), symbol_errors=(shape, "int64")), shape, origin, force=force,
                limits={"bit_errors": (0, 2*ndata), "symbol_errors": (0, ndata)}, data_guard=guard.assert_unchanged) as entry:
            whitener = None
            if not np.all(entry.complete):
                whitener = make_whitener(ds._shared["unit_rf_noise_covariance"][:], backend)
            for si, snr in enumerate(ds.snr_db):
                for d in range(2):
                    for start in range(0, ds.frame_count, batch_size):
                        frames = [fi for fi in range(start, min(start+batch_size, ds.frame_count))
                                  if not entry.complete[si, fi, d]]
                        if not frames:
                            continue
                        batch = prepare_batch(ds, si, frames, d, backend)
                        result = equalize_batch(batch, backend, model=model, whitener=whitener)
                        errors, symbol_errors = score_oamp_net2(ds, frames, d, result)
                        for j, frame in enumerate(frames):
                            entry.commit((si, frame, d), bit_errors=errors[j], symbol_errors=symbol_errors[j])
                LOG.info("%s OAMP-Net2 test SNR=%g complete", ds.channel, snr)
                write_json(run/"progress.json", dict(channel=ds.channel, snr_db=float(snr),
                    completed_blocks=int(entry.complete.sum()), total_blocks=int(np.prod(shape))))
            errors = entry.read("bit_errors").sum(axis=(1, 2))
            symbol_errors = entry.read("symbol_errors").sum(axis=(1, 2))
            provenance = entry.provenance()
            entry.publish()
        guard.assert_unchanged()
        if file_sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError("OAMP-Net2 checkpoint changed during evaluation")
        bits = ds.frame_count*2*ndata*2
        report = dict(baseline, schema="oamp-net2-comparison-v1", smoke=bool(payload.get("smoke")), curves=dict(baseline["curves"]))
        report["curves"]["OAMP-Net2"] = dict(identifier="oamp_net2", precision="complex128/float64",
            errors=errors.tolist(), total_bits=[bits]*len(ds.snr_db), ber=(errors/bits).tolist(),
            symbol_errors=symbol_errors.tolist(), total_symbols=[bits//2]*len(ds.snr_db),
            ser=(symbol_errors/(bits//2)).tolist(), provenance=provenance)
        report["oamp_net2"] = dict(checkpoint=str(checkpoint.resolve()), sha256=checkpoint_sha,
            model_config=payload["model_config"], preprocessing=payload["preprocessing"],
            training_identity=payload["identity"], best_epoch=payload["epoch"],
            best_validation_loss=payload["best_loss"], training_scope=payload["training_scope"],
            parameters=payload["parameters"], paper_doi=payload["paper_doi"],
            selection="train/validation final-layer squared vector error; test excluded",
            csi="fixed compensated CSI; thermal covariance; detector only, no JCESD")
        publish_report(run, report)
        return report


def evaluate(dataset_path, checkpoint, run, backend, **kwargs):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".oamp_evaluation.lock"):
        return _evaluate(dataset_path, checkpoint, run, backend, **kwargs)
