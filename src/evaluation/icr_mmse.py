"""Full-matrix ICR-MMSE-Net scoring, with physical-record uncertainty estimates.

Prediction is completed before opening the independent supervision interface.
Reported CSI values average each full-matrix linear NMSE, then convert to dB.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import ctypes
import logging
from pathlib import Path
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches, supervision
from src.algorithms.icr_mmse.dpss import base_error
from src.evaluation.fingerprints import DataIdentity, numerical_environment, source_fingerprint
from src.evaluation.result_cache import FileLock

LOG = logging.getLogger(__name__)
SCHEMA = "icr-mmse-net-record-evaluation-v1"


def process_peak_bytes():
    """OS process-lifetime peak RSS, including native tensor allocations."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("Unable to read process peak working set")
        return int(counters.PeakWorkingSetSize)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def record_interval(numerators, denominators, *, seed=20261028, resamples=2000):
    """Bootstrap whole records; carrier/repeat counts never become sample size."""
    a, b = np.asarray(numerators, float), np.asarray(denominators, float)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or np.any(b <= 0):
        raise ValueError("Record totals require matching nonempty vectors and positive denominators")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or resamples < 1:
        raise ValueError("Invalid bootstrap totals or replicate count")
    estimate = float(a.sum()/b.sum())
    if len(a) < 2:
        return dict(estimate=estimate, lower=None, upper=None, records=len(a),
                    method="record bootstrap unavailable with fewer than two records")
    rng = np.random.default_rng(seed)
    # A bounded temporary also makes many-record experiments inexpensive.
    values = []
    for start in range(0, resamples, 128):
        selected = rng.integers(len(a), size=(min(128, resamples-start), len(a)))
        values.extend((a[selected].sum(axis=1)/b[selected].sum(axis=1)).tolist())
    lower, upper = np.quantile(values, (.025, .975))
    return dict(estimate=estimate, lower=float(lower), upper=float(upper), records=len(a),
                method="95% percentile bootstrap of physical records", resamples=resamples)


class RecordMetrics:
    """Sufficient statistics per (SNR, physical record), retaining layer order."""
    def __init__(self, layers):
        self.layers = layers
        self.rows = defaultdict(lambda: dict(blocks=0, bits=0, bit_errors=np.zeros(layers, np.int64),
            nmse=np.zeros(layers+1), csi_worse=np.zeros(layers, np.int64),
            feedback_worse=np.zeros(max(layers-1, 0), np.int64)))

    def add(self, snr_index, record, errors, bit_count, nmse):
        errors, nmse = np.asarray(errors, np.int64), np.asarray(nmse, float)
        if errors.shape != (self.layers,) or nmse.shape != (self.layers+1,):
            raise ValueError("Layer metric shape mismatch")
        if not np.isfinite(nmse).all() or np.any(nmse < 0) or bit_count < 1:
            raise ValueError("Invalid full-matrix NMSE or bit count")
        row = self.rows[(int(snr_index), int(record))]
        row["blocks"] += 1
        row["bits"] += int(bit_count)
        row["bit_errors"] += errors
        row["nmse"] += nmse
        row["csi_worse"] += nmse[1:] > nmse[:-1]
        row["feedback_worse"] += errors[1:] > errors[:-1]

    def report(self, snr_db, *, bootstrap_samples=2000):
        per_snr = []
        for si, snr in enumerate(snr_db):
            rows = sorted((record, value) for (index, record), value in self.rows.items() if index == si)
            if not rows:
                raise ValueError("Every requested SNR requires scored records")
            blocks = np.array([value["blocks"] for _, value in rows])
            bits = np.array([value["bits"] for _, value in rows])
            errors = np.stack([value["bit_errors"] for _, value in rows])
            nmse = np.stack([value["nmse"] for _, value in rows])
            mean_nmse = nmse.sum(axis=0)/blocks.sum()
            csi_worse = np.stack([value["csi_worse"] for _, value in rows])
            feedback_worse = np.stack([value["feedback_worse"] for _, value in rows])
            per_snr.append(dict(snr_db=float(snr), records=len(rows), blocks=int(blocks.sum()),
                total_bits=int(bits.sum()), bit_errors=errors.sum(axis=0).tolist(),
                ber=(errors.sum(axis=0)/bits.sum()).tolist(), mean_nmse_linear=mean_nmse.tolist(),
                mean_nmse_db=(10*np.log10(np.maximum(mean_nmse, np.finfo(float).tiny))).tolist(),
                csi_deterioration_fraction=(csi_worse.sum(axis=0)/blocks.sum()).tolist(),
                feedback_ber_deterioration_fraction=(feedback_worse.sum(axis=0)/blocks.sum()).tolist(),
                ber_intervals=[record_interval(errors[:, t], bits, resamples=bootstrap_samples)
                               for t in range(self.layers)],
                nmse_intervals=[record_interval(nmse[:, t], blocks, resamples=bootstrap_samples)
                                for t in range(self.layers+1)],
                by_record=[dict(record=record, blocks=value["blocks"], bits=value["bits"],
                    bit_errors=value["bit_errors"].tolist(), nmse_sum=value["nmse"].tolist(),
                    csi_worse=value["csi_worse"].tolist(), feedback_worse=value["feedback_worse"].tolist())
                    for record, value in rows]))
        return per_snr


def _initial_nmse(initial_channel, reference):
    energy = reference.abs().square().sum(dim=(-2, -1))
    if not torch.isfinite(reference).all() or (energy <= 0).any():
        raise ValueError("Reference CSI must be finite with positive matrix energy")
    return base_error(initial_channel, reference)/energy


def coverage(data_directory, run, *, bandwidths=(16,), batch_size=2, frame_limit=None,
             device="cpu", precision="complex64", operator_backend="dense", reference_cache_mib=512):
    """Measure local correction capacity on train/validation only, never test."""
    bandwidths = tuple(bandwidths)
    if not bandwidths or any(not isinstance(b, int) or isinstance(b, bool) or b < 0 for b in bandwidths):
        raise ValueError("Bandwidths must be nonnegative integers")
    directory, run = Path(data_directory), Path(run)
    run.mkdir(parents=True, exist_ok=True)
    result = dict(schema="icr-mmse-local-error-coverage-v1", input_view="compensated", precision=precision,
                  bandwidths=list(bandwidths),
                  definition="sum |H_init-H_ref|^2 inside |physical row-column|<=B / full error energy",
                  splits={})
    for split in ("train", "validation"):
        sums = defaultdict(lambda: np.zeros(len(bandwidths)+1))
        with ExitStack() as stack, torch.no_grad():
            dataset = stack.enter_context(CompactDataset(directory/(split+".h5")))
            if dataset.split != split:
                raise ValueError("Coverage requires train/validation split identities")
            if operator_backend == "dense_device":
                from src.data.icr_training import ReferenceCache, training_batches
                batches = stack.enter_context(training_batches(dataset, device=device, batch_size=batch_size,
                    frame_limit=frame_limit, precision=precision, operator_backend=operator_backend,
                    cache=ReferenceCache(reference_cache_mib*2**20), compact_reference=False))
            elif operator_backend == "dense":
                batches = ((inputs, supervision(dataset, keys, device=device, precision=precision), keys)
                    for inputs, keys in input_batches(dataset, device=device, batch_size=batch_size,
                        frame_limit=frame_limit, precision=precision))
            else:
                raise ValueError("Full-matrix coverage requires dense or dense_device")
            for inputs, target, keys in batches:
                error = (inputs["initial_channel"]-target["reference_channel"]).abs().square()
                positions = inputs["carrier_positions"]
                distance = (positions[:, None]-positions[None, :]).abs()
                values = [error.sum(dim=(-2, -1))]
                values.extend((error*(distance <= b)).sum(dim=(-2, -1)) for b in bandwidths)
                array = torch.stack(values, dim=-1).cpu().numpy()
                records = target["records"].cpu().numpy()
                for key, record, row in zip(keys, records, array):
                    sums[(key[0], int(record))] += row
            per_snr = []
            for si, snr in enumerate(dataset.snr_db):
                rows = [(record, value) for (index, record), value in sorted(sums.items()) if index == si]
                if not rows:
                    raise ValueError("Coverage cannot omit an SNR")
                total = np.stack([value for _, value in rows]).sum(axis=0)
                per_snr.append(dict(snr_db=float(snr), record_count=len(rows), error_energy=float(total[0]),
                    fractions=(total[1:]/total[0]).tolist() if total[0] else [None]*len(bandwidths),
                    by_record=[dict(record=record, error_energy=float(value[0]),
                        fractions=(value[1:]/value[0]).tolist() if value[0] else [None]*len(bandwidths))
                        for record, value in rows]))
            result["splits"][split] = dict(channel=dataset.channel, snr_db=dataset.snr_db.tolist(), per_snr=per_snr)
    write_json(run/"coverage.json", result)
    return result


def _record_disjoint(dataset):
    if dataset.split == "test":
        names = ("train", "validation")
    elif dataset.split == "validation":
        names = ("train",)
    else:
        raise ValueError("Model selection/scoring requires validation or test split")
    records = dataset._h["windows/record"][:]
    for name in names:
        with h5py.File(dataset.path.parent/(name+".h5"), "r") as other:
            if np.intersect1d(records, other["windows/record"][:]).size:
                raise ValueError("Physical-record leakage between dataset splits")


def evaluate(dataset_path, checkpoint, run, backend, *, batch_size=2, frame_limit=None,
             require_test=True, include_baselines=False, udnet_checkpoint=None, bootstrap_samples=2000,
             baseline_parameters=None, cache_dir=None, operator_backend="dense"):
    """Evaluate one checkpoint, recomputing preprocessing and every feedback step.

    Validation use must be explicit via require_test=False; test is never used
    by the search routine. Model timings include dataset CSI reconstruction and all
    layers, while supervision/scoring and optional baseline runs are excluded.
    """
    from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE, load_model, validate_dataset
    from src.algorithms.icr_mmse.inference import prepare_frozen_inference
    from src.algorithms.icr_mmse.model import csi_nmse
    run, checkpoint = Path(run), Path(checkpoint)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".icr_evaluation.lock"):
        model, payload = load_model(checkpoint, backend.device)
        prepare_frozen_inference(model)
        deployment_code = (("src.algorithms.icr_mmse.inference",)
                           if model.config["detector_mode"] == "budgeted" else ())
        precision = model.config["precision"]
        real_precision = "float32" if precision == "complex64" else "float64"
        checkpoint_hash = file_sha256(checkpoint)
        started = time.perf_counter()
        with CompactDataset(dataset_path) as dataset, torch.inference_mode():
            validate_dataset(payload, dataset)
            if require_test and dataset.split != "test":
                raise ValueError("Formal ICR-MMSE-Net scoring requires test split")
            _record_disjoint(dataset)
            if include_baselines and frame_limit is not None and frame_limit < dataset.frame_count:
                raise ValueError("Baseline comparison requires identical full frame counts; disable baselines for a frame-limited smoke test")
            guard = DataIdentity({name: dataset.path.parent/name for name in
                (dataset.path.name, "model.h5", "train.h5", "validation.h5")})
            accumulator = None
            preprocessing_and_prediction = 0.
            blocks, batches = 0, 0
            if backend.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            iterator = iter(input_batches(dataset, device=backend.device, batch_size=batch_size,
                                          frame_limit=frame_limit, precision=precision,
                                          operator_backend=operator_backend))
            while True:
                backend.synchronize()
                tick = time.perf_counter()
                try:
                    inputs, keys = next(iterator)
                except StopIteration:
                    break
                output = model(**inputs, return_diagnostics=False)
                backend.synchronize()
                preprocessing_and_prediction += time.perf_counter()-tick
                # This is the first access to references or labels for a batch.
                target = supervision(dataset, keys, device=backend.device, precision=precision)
                mask = inputs["data_mask"]
                predicted = output.symbols[..., mask]
                expected = target["symbols"][:, None, :]
                errors = ((predicted.real < 0) != (expected.real < 0)).sum(dim=-1)
                errors += ((predicted.imag < 0) != (expected.imag < 0)).sum(dim=-1)
                after = csi_nmse(output, target["reference_channel"])
                before = _initial_nmse(inputs["initial_channel"], target["reference_channel"])
                nmse = torch.cat((before[:, None], after), dim=-1).cpu().numpy()
                errors = errors.cpu().numpy()
                if accumulator is None:
                    accumulator = RecordMetrics(errors.shape[-1])
                records = target["records"].cpu().tolist()
                for i, ((si, _, _), record) in enumerate(zip(keys, records)):
                    accumulator.add(si, record, errors[i], 2*int(mask.sum()), nmse[i])
                blocks += len(keys)
                batches += 1
                if batches % 20 == 0:
                    write_json(run/"progress.json", dict(phase="evaluation", processed_blocks=blocks,
                        prediction_seconds=preprocessing_and_prediction))
            if accumulator is None:
                raise ValueError("Evaluation requires at least one batch")
            per_snr = accumulator.report(dataset.snr_db, bootstrap_samples=bootstrap_samples)
            total_bits = sum(row["total_bits"] for row in per_snr)
            initial_nmse = sum(row["mean_nmse_linear"][0]*row["blocks"] for row in per_snr)/blocks
            final_nmse = sum(row["mean_nmse_linear"][-1]*row["blocks"] for row in per_snr)/blocks
            elapsed = time.perf_counter()-started
            report = dict(schema=SCHEMA, complete=True, input_view="compensated", channel=dataset.channel,
                operator_backend=operator_backend,
                smoke=bool(payload.get("smoke")) or frame_limit is not None or bool(dataset._config.get("smoke")),
                split=dataset.split, dataset=str(Path(dataset_path).resolve()), snr_db=dataset.snr_db.tolist(),
                checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
                model_config=payload["model_config"], feedback_layout=payload["feedback_layout"],
                data_identity=guard.files,
                effective_model_config=dict(model.config),
                code=source_fingerprint(INFERENCE_CODE+("src.evaluation.icr_mmse",)+deployment_code),
                deployment="frozen scalar/window caches" if deployment_code else "checkpoint forward",
                numerical_environment=dict(numerical_environment(backend),
                    precision=precision+"/"+real_precision), per_snr=per_snr,
                selection=dict(ber=sum(row["bit_errors"][-1] for row in per_snr)/total_bits,
                    csi_improved=bool(final_nmse < initial_nmse), initial_nmse=initial_nmse, corrected_nmse=final_nmse),
                curves={"ICR-MMSE-Net":dict(identifier="icr_mmse", precision=precision+"/"+real_precision,
                    ber=[r["ber"][-1] for r in per_snr],
                    errors=[r["bit_errors"][-1] for r in per_snr], total_bits=[r["total_bits"] for r in per_snr])},
                performance=dict(preprocessing_and_all_iterations_seconds=preprocessing_and_prediction,
                    seconds_per_block=preprocessing_and_prediction/blocks, total_scoring_seconds=elapsed,
                    blocks=blocks, batches=batches, parameters=sum(p.numel() for p in model.parameters()),
                    peak_process_rss_bytes=process_peak_bytes(),
                    peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()) if backend.device == "cuda" else None,
                    peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()) if backend.device == "cuda" else None,
                    cpu_memory_definition="OS process-lifetime peak RSS; includes native tensors",
                    timing_definition="disk reads, dataset CSI reconstruction, input conversion, preprocessing and all iterations; excludes scoring/labels"),
                statistics=dict(unit="physical record", confidence="whole-record bootstrap",
                    nmse="mean of complete-matrix per-block linear NMSE, then 10log10",
                    csi_deterioration="fraction of blocks with NMSE_t > NMSE_(t-1), including input CSI at t=0",
                    feedback_deterioration="fraction of blocks with BER_t > BER_(t-1), t>=2"))
            guard.assert_unchanged()
            if file_sha256(checkpoint) != checkpoint_hash:
                raise RuntimeError("ICR-MMSE-Net checkpoint changed while evaluating")
        if include_baselines:
            if not require_test:
                raise ValueError("Baseline publication is supported only for formal test scoring")
            from src.evaluation.comparison import evaluate as baseline_evaluate
            baseline = baseline_evaluate(dataset_path, run/"baselines", backend, batch_size=batch_size,
                input_view="compensated", algorithm_parameters=baseline_parameters, cache_dir=cache_dir)
            report["curves"] = dict(baseline["curves"], **report["curves"])
            report["baselines"] = baseline
            report["comparison_precision"] = dict(icr_mmse=precision+"/"+real_precision,
                traditional_baselines=baseline["numerical_environment"]["precision"])
            if udnet_checkpoint is not None:
                from src.evaluation.udnet import evaluate as udnet_evaluate
                udnet = udnet_evaluate(dataset_path, udnet_checkpoint, run/"udnet", backend,
                    input_view="compensated", batch_size=batch_size, cache_dir=cache_dir,
                    algorithm_parameters=baseline_parameters)
                report["curves"]["UDNet"] = udnet["curves"]["UDNet"]
                report["udnet_checkpoint"] = str(Path(udnet_checkpoint).resolve())
                report["comparison_precision"]["udnet"] = "complex128/float64 preprocessing; float32 detector"
        plot_report(report, run)
        write_json(run/"metrics.json", report)
        return report


def plot_report(report, run):
    run = Path(run)
    snr = np.array(report["snr_db"])
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for name, curve in report["curves"].items():
            values = np.array(curve["ber"])
            shown = np.where(values > 0, values, .5/np.array(curve["total_bits"]))
            ax.semilogy(snr, shown, "o-", label=name)
        ax.set(xlabel="SNR (dB)", ylabel="BER", title=report["channel"]+" | compensated CSI")
        ax.grid(True, which="both", alpha=.25)
        ax.legend()
        fig.savefig(run/"ber_snr.png", dpi=180)
        plt.close(fig)
        nmse = np.array([row["mean_nmse_db"] for row in report["per_snr"]])
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        ax.plot(snr, nmse[:, 0], "o-", label="Dataset CSI")
        ax.plot(snr, nmse[:, -1], "s-", label="Corrected CSI")
        ax.set(xlabel="SNR (dB)", ylabel="Mean full-matrix NMSE (dB)", title=report["channel"])
        ax.grid(True, alpha=.25)
        ax.legend()
        fig.savefig(run/"nmse_snr.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
        for row in report["per_snr"]:
            label = f'{row["snr_db"]:g} dB'
            layers = np.arange(1, len(row["ber"])+1)
            axes[0].semilogy(layers, np.maximum(row["ber"], .5/row["total_bits"]), "o-", label=label)
            axes[1].plot(np.arange(len(row["mean_nmse_db"])), row["mean_nmse_db"], "o-", label=label)
            axes[2].plot(layers, row["csi_deterioration_fraction"], "o-", label=label)
        for ax, ylabel in zip(axes, ("BER", "Full-matrix NMSE (dB)", "CSI deterioration fraction")):
            ax.set(xlabel="Joint layer", ylabel=ylabel)
            ax.grid(True, alpha=.25)
            ax.legend(fontsize=8)
        fig.savefig(run/"layerwise.png", dpi=180)
        plt.close(fig)
