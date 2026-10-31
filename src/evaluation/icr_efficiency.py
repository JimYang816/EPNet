"""Compensated-CSI efficiency comparison with a fresh process for each detector.

Workers prohibit reads of test supervision and report setup separately from
reconstruction plus inference.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy  # Initialize the scientific runtime before torch on Windows.
import h5py
import torch
from threadpoolctl import threadpool_limits

from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.contract import DEFAULT_INPUT_VIEW
from src.evaluation.fingerprints import (DataIdentity, ROOT,
                                         numerical_environment, source_fingerprint)
from src.evaluation.icr_mmse import process_peak_bytes
from src.evaluation.inputs import prepare_batch
from src.evaluation.registry import AVAILABLE_ALGORITHMS, resolve_parameters

AVAILABLE_IDENTIFIERS = tuple(a.identifier for a in AVAILABLE_ALGORITHMS) + ("udnet", "oamp_net2", "icr_mmse")


def _guard_truth_reads():
    original = h5py.Dataset.__getitem__
    def guarded(dataset, selection):
        # The accepted compensated CSI is constructed from saved reference_csi
        # and alpha. These are input construction, not model supervision. Labels
        # and other audit fields remain prohibited in timing workers.
        if any(value in dataset.name for value in ("reference_projection",
                "samples/bits", "actual_snr", "model_residual", "effective_rf", "training_calibration/")):
            raise AssertionError("Efficiency inference attempted to read supervision: " + dataset.name)
        return original(dataset, selection)
    h5py.Dataset.__getitem__ = guarded


def _worker(options):
    identifier, device = options["algorithm"], options["device"]
    if identifier not in AVAILABLE_IDENTIFIERS:
        raise ValueError("Unknown efficiency algorithm: " + identifier)
    torch.set_num_threads(options["threads"])
    _guard_truth_reads()
    backend = DenseBackend(device)
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with CompactDataset(options["dataset"], input_view=DEFAULT_INPUT_VIEW) as dataset, torch.inference_mode():
        if dataset.split not in ("validation", "test"):
            raise ValueError("Efficiency requires a validation/test input split")
        covariance = dataset._shared["unit_rf_noise_covariance"][:]
        model = preprocessor = thermal = None
        parameters = {}
        if identifier in ("udnet", "oamp_net2", "icr_mmse"):
            if identifier == "udnet":
                from src.algorithms.udnet.checkpoint import load_model, validate_dataset
                from src.algorithms.udnet.inference import equalize_batch
                from src.data.udnet import LocalPreprocessor, WindowPlan
            elif identifier == "oamp_net2":
                from src.algorithms.oamp_net2.checkpoint import load_model, validate_dataset
                from src.algorithms.oamp_net2.inference import equalize_batch
            else:
                from src.algorithms.icr_mmse.checkpoint import load_model, validate_dataset
            model, payload = load_model(options[identifier + "_checkpoint"], device)
            validate_dataset(payload, dataset)
            if identifier == "icr_mmse":
                from src.algorithms.icr_mmse.inference import prepare_frozen_inference
                prepare_frozen_inference(model)
            if identifier == "udnet":
                preprocessor = LocalPreprocessor(WindowPlan.build(dataset._config["nfft"], dataset._indices),
                                                 covariance, device)
        if identifier in ("full_matrix_lmmse", "mmse_dfe", "mmse_pic", "oamp_net2"):
            from src.algorithms.likelihood import make_whitener
            thermal = make_whitener(covariance, backend)
        algorithm = next((a for a in AVAILABLE_ALGORITHMS if a.identifier == identifier), None)
        if algorithm:
            importlib.import_module(algorithm.module)
            supplied = options["comparison"].get("baselines", options["comparison"]).get("parameters", {})
            parameters = resolve_parameters({identifier: supplied.get(identifier, {})})[identifier]
        backend.synchronize()
        setup_seconds = time.perf_counter() - started
        prediction_seconds, blocks = 0., 0
        if identifier == "icr_mmse":
            from src.data.icr_mmse import input_batches
            iterator = iter(input_batches(dataset, device=device, batch_size=options["batch_size"],
                                          frame_limit=options["frame_limit"],
                                          precision=model.config["precision"]))
            while True:
                backend.synchronize()
                tick = time.perf_counter()
                try:
                    inputs, keys = next(iterator)
                except StopIteration:
                    break
                output = model(**inputs, return_diagnostics=False)
                if not torch.isfinite(output.probabilities).all():
                    raise FloatingPointError("Nonfinite ICR probabilities")
                backend.synchronize()
                prediction_seconds += time.perf_counter() - tick
                blocks += len(keys)
        else:
            count = min(dataset.frame_count, options["frame_limit"] or dataset.frame_count)
            for si in range(len(dataset.snr_db)):
                for d in range(2):
                    for start in range(0, count, options["batch_size"]):
                        frames = list(range(start, min(start+options["batch_size"], count)))
                        backend.synchronize()
                        tick = time.perf_counter()
                        batch = prepare_batch(dataset, si, frames, d, backend)
                        if identifier == "udnet":
                            result = equalize_batch(batch, backend, model=model, preprocessor=preprocessor)
                        elif identifier == "oamp_net2":
                            result = equalize_batch(batch, backend, model=model, whitener=thermal)
                        else:
                            extra = {} if identifier == "single_tap_mmse" else dict(whitener=thermal)
                            result = algorithm.equalize(batch, backend, **extra, **parameters)
                        if not np.isfinite(result.symbols).all():
                            raise FloatingPointError("Nonfinite detector output")
                        backend.synchronize()
                        prediction_seconds += time.perf_counter() - tick
                        blocks += len(frames)
        total = setup_seconds + prediction_seconds
        environment = numerical_environment(backend)
        if identifier == "icr_mmse":
            environment["precision"] = (model.config["precision"] + "/" +
                ("float32" if model.config["precision"] == "complex64" else "float64"))
        elif identifier == "udnet":
            environment["precision"] = "complex128/float64_to_float32"
        return dict(algorithm=identifier, process_id=os.getpid(), input_view=dataset.input_view, blocks=blocks,
            detector_controls=parameters,
            snr_db=dataset.snr_db.tolist(), batch_size=options["batch_size"], frame_limit=options["frame_limit"],
            setup_seconds=setup_seconds, reconstruction_and_all_iterations_seconds=prediction_seconds,
            setup_and_prediction_seconds=total, amortized_seconds_per_block=total/blocks,
            parameters=sum(p.numel() for p in model.parameters()) if model else 0,
            peak_process_rss_bytes=process_peak_bytes(),
            numerical_environment=environment,
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()) if device == "cuda" else None)


def benchmark(dataset_path, comparison_report, checkpoint, udnet_checkpoint, *, output=None,
              device="cpu", batch_size=2, frame_limit=None, threads=4, algorithms=None, oamp_net2_checkpoint=None):
    if algorithms is None:
        algorithms = AVAILABLE_IDENTIFIERS
    if (device not in ("cpu", "cuda") or any(isinstance(x, bool) or not isinstance(x, int) or x < 1
        for x in (batch_size, threads)) or (frame_limit is not None and
        (isinstance(frame_limit, bool) or not isinstance(frame_limit, int) or frame_limit < 1))
        or not algorithms or len(set(algorithms)) != len(algorithms) or set(algorithms)-set(AVAILABLE_IDENTIFIERS)
        or ("oamp_net2" in algorithms and oamp_net2_checkpoint is None)):
        raise ValueError("Invalid efficiency benchmark controls")
    validation_started = time.perf_counter()
    dataset_path = Path(dataset_path).resolve()
    sources = {"dataset": dataset_path,
        "model.h5": dataset_path.parent/"model.h5",
        "icr_checkpoint": checkpoint, "udnet_checkpoint": udnet_checkpoint,
        "comparison_report": comparison_report}
    if oamp_net2_checkpoint is not None:
        sources["oamp_net2_checkpoint"] = oamp_net2_checkpoint
    guard = DataIdentity(sources)
    report = json.loads(Path(comparison_report).read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("input_view") != DEFAULT_INPUT_VIEW:
        raise ValueError("Efficiency requires a complete compensated CSI comparison report")
    initial_validation_seconds = time.perf_counter() - validation_started
    options = dict(dataset=str(Path(dataset_path).resolve()), comparison=report,
        icr_mmse_checkpoint=str(Path(checkpoint).resolve()), udnet_checkpoint=str(Path(udnet_checkpoint).resolve()),
        oamp_net2_checkpoint=str(Path(oamp_net2_checkpoint).resolve()) if oamp_net2_checkpoint is not None else None,
        device=device, batch_size=batch_size, frame_limit=frame_limit, threads=threads)
    rows = []
    for algorithm in algorithms:
        completed = subprocess.run([sys.executable, "-m", __name__, "--worker"], cwd=ROOT,
            input=json.dumps(dict(options, algorithm=algorithm)), text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"Efficiency worker {algorithm} failed:\n{completed.stderr}")
        rows.append(json.loads(completed.stdout))
    guard.assert_unchanged()
    checkpoints = dict(icr_mmse=options["icr_mmse_checkpoint"], udnet=options["udnet_checkpoint"])
    if oamp_net2_checkpoint is not None:
        checkpoints["oamp_net2"] = options["oamp_net2_checkpoint"]
    result = dict(schema="icr-isolated-efficiency-v1", input_view=DEFAULT_INPUT_VIEW, device=device, algorithms=rows,
        dataset=options["dataset"], comparison_report=str(Path(comparison_report).resolve()),
        checkpoints=checkpoints,
        data_identity=guard.files, code=source_fingerprint(("src.evaluation.icr_efficiency",)),
        initial_provenance_validation_seconds=initial_validation_seconds,
        timing="setup + disk reads, dataset CSI construction, preprocessing and every detector iteration; no labels/scoring",
        setup="dataset open, checkpoint load, thermal noise whitening and window plan setup",
        provenance_timing="data/checkpoint/report identity hashing is excluded from setup/amortized runtime and reported separately; checkpoint loader source verification remains included",
        memory="fresh subprocess per algorithm; OS peak RSS includes imports/native tensors; CUDA peaks include model and setup",
        filesystem_cache="OS file cache is not flushed; one cold process per detector",
        process_startup_and_import_seconds_included=False, test_labels_read=False,
        reference_assisted_csi_construction=True,
        reference_usage="Saved reference CSI and alpha only construct the accepted compensated input; never passed separately to detectors")
    if output:
        write_json(output, result)
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use python -m scripts.evaluate.icr_efficiency")
    json_options = json.load(sys.stdin)
    with threadpool_limits(json_options["threads"]):
        print(json.dumps(_worker(json_options)))
