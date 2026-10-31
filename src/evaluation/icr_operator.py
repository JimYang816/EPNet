"""Same-checkpoint dense/DPSS equivalence, accuracy and isolated efficiency."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE, load_model, validate_dataset
from src.algorithms.icr_mmse.dpss import DPSSChannel
from src.algorithms.icr_mmse.model import ICRMMSEnet, csi_nmse, layerwise_loss
from src.algorithms.icr_mmse.operators import BandChannel
from src.common.experiment import file_sha256, write_json
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches, supervision
from src.evaluation.fingerprints import ROOT, DataIdentity, numerical_environment, source_fingerprint
from src.evaluation.icr_mmse import RecordMetrics, _initial_nmse, _record_disjoint, process_peak_bytes
from src.evaluation.result_cache import FileLock
from src.training.icr_mmse import configure_stage

MODES = ("dense", "dpss")


def _indices(dataset, limit):
    return np.linspace(0, dataset.frame_count-1, min(limit or dataset.frame_count, dataset.frame_count),
                       dtype=int).tolist()


def _paired_batches(dataset, options, *, gradients=False):
    common = dict(device=options["device"], batch_size=options["batch_size"],
                  precision=options["precision"], frame_indices=_indices(dataset, 1 if gradients else options["frame_limit"]))
    if gradients:
        common["snr_indices"] = sorted({0, len(dataset.snr_db)-1})
    iterators = [input_batches(dataset, operator_backend=mode, **common) for mode in MODES]
    for dense, fast in zip(*iterators, strict=True):
        if dense[1] != fast[1]:
            raise RuntimeError("Operator backends produced different sample keys")
        for name in ("observations", "known", "rf_variance", "unit_covariance"):
            if not torch.equal(dense[0][name], fast[0][name]):
                raise RuntimeError("Operator comparison changed receiver inputs: " + name)
        yield dense[0], fast[0], dense[1]


def _relative(actual, expected):
    return float(torch.linalg.vector_norm((actual-expected).reshape(-1)) /
                 torch.linalg.vector_norm(expected.reshape(-1)).clamp_min(1e-30))


def _gradient_worker(options):
    model, payload = load_model(options["checkpoint"], options["device"], input_view="raw")
    native_precision = model.config["precision"]
    converted = options.get("validation_precision") == "complex128" and native_precision != "complex128"
    if converted:
        precise = ICRMMSEnet(**dict(model.config, precision="complex128")).to(options["device"])
        precise.load_state_dict(model.state_dict(), strict=True)
        model = precise.eval().requires_grad_(False)
    options = dict(options, precision=model.config["precision"])
    rows = []
    with CompactDataset(options["validation"], input_view="raw") as dataset:
        validate_dataset(payload, dataset)
        if dataset.split != "validation":
            raise ValueError("Equivalence gradients require validation data, never test fitting")
        for dense, fast, keys in _paired_batches(dataset, options, gradients=True):
            # Supervision is independent of both complete inference input paths.
            targets = supervision(dataset, keys, device=options["device"], precision=options["precision"])
            raw, expected = fast["initial_channel"], dense["initial_channel"]
            probe = dense["observations"]
            weights = dense["unit_covariance"].diagonal().real.reciprocal()
            op = BandChannel(raw, torch.zeros(len(keys), expected.shape[-1], 2*model.config["bandwidth"]+1,
                                              dtype=expected.dtype, device=expected.device))
            exact = BandChannel(expected, op.bands)
            operators = dict(matvec=_relative(op.matvec(probe), exact.matvec(probe)),
                adjoint=_relative(op.adjoint(probe), exact.adjoint(probe)),
                local_entries=_relative(op.local_entries(), exact.local_entries()),
                weighted_column_energy=_relative(op.column_energy(weights), exact.column_energy(weights)))
            for stage in "ABC":
                configure_stage(model, stage)
                outputs, gradients, losses = [], [], []
                for inputs in (dense, fast):
                    model.zero_grad(set_to_none=True)
                    output = model(**inputs, correction_enabled=stage != "A", detach_feedback=stage == "B")
                    loss = layerwise_loss(output, targets["classes"], targets["reference_channel"],
                        inputs["data_mask"], lambda_h=.1 if stage != "A" else 0., csi_only=stage == "B")
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite paired equivalence loss")
                    loss.backward()
                    gradients.append(torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                                                 if p.requires_grad]))
                    if not torch.isfinite(gradients[-1]).all() or not torch.isfinite(output.probabilities).all():
                        raise FloatingPointError("Nonfinite paired equivalence outputs/gradients")
                    outputs.append(output)
                    losses.append(float(loss.detach()))
                first, second = outputs
                tolerance = 1e-9 if options["precision"] == "complex128" else 5e-4
                probability = float((first.probabilities-second.probabilities).abs().max())
                gradient = _relative(gradients[1], gradients[0])
                band = _relative(second.bands, first.bands)
                nmse = _relative(csi_nmse(second, targets["reference_channel"]),
                                 csi_nmse(first, targets["reference_channel"]))
                passed = (probability < tolerance and gradient < tolerance and band < tolerance
                          and nmse < tolerance and max(operators.values()) < tolerance)
                rows.append(dict(keys=keys, stage=stage, operator_relative_errors=operators,
                    probability_max_abs=probability, csi_increment_relative_error=band,
                    full_nmse_relative_error=nmse, parameter_gradient_relative_error=gradient,
                    losses=losses, tolerance=tolerance, passed=passed))
                del outputs, gradients, first, second, output, loss
    return dict(passed=bool(rows) and all(row["passed"] for row in rows), comparisons=rows,
                precision=options["precision"], checkpoint_precision=native_precision,
                weights_converted_to_double=converted,
                scope="one validation frame, both D blocks, lowest/highest SNR, stages A/B/C; no parameter updates")


def _accuracy_worker(options):
    model, payload = load_model(options["checkpoint"], options["device"], input_view="raw")
    options = dict(options, precision=model.config["precision"])
    accumulators = {mode: RecordMetrics(model.config["layers"]) for mode in MODES}
    blocks = disagreements = 0
    maximum_probability_difference = maximum_band_difference = 0.
    with CompactDataset(options["dataset"], input_view="raw") as dataset, torch.inference_mode():
        validate_dataset(payload, dataset)
        if dataset.split != "test":
            raise ValueError("Accuracy comparison requires the independent test split")
        _record_disjoint(dataset)
        for dense, fast, keys in _paired_batches(dataset, options):
            predictions = [model(**inputs) for inputs in (dense, fast)]
            maximum_probability_difference = max(maximum_probability_difference,
                float((predictions[0].probabilities-predictions[1].probabilities).abs().max()))
            maximum_band_difference = max(maximum_band_difference,
                float((predictions[0].bands-predictions[1].bands).abs().max()))
            target = supervision(dataset, keys, device=options["device"], precision=options["precision"])
            mask = dense["data_mask"]
            for mode, output, inputs in zip(MODES, predictions, (dense, fast)):
                symbols, expected = output.symbols[..., mask], target["symbols"][:, None]
                errors = ((symbols.real < 0) != (expected.real < 0)).sum(-1)
                errors += ((symbols.imag < 0) != (expected.imag < 0)).sum(-1)
                nmse = torch.cat((_initial_nmse(inputs["initial_channel"], target["reference_channel"])[:, None],
                                   csi_nmse(output, target["reference_channel"])), -1)
                for key, record, er, nh in zip(keys, target["records"].cpu().tolist(),
                                               errors.cpu().numpy(), nmse.cpu().numpy()):
                    accumulators[mode].add(key[0], record, er, 2*int(mask.sum()), nh)
            x, y = [out.symbols[:, -1, mask] for out in predictions]
            disagreements += int(((x.real < 0) != (y.real < 0)).sum() + ((x.imag < 0) != (y.imag < 0)).sum())
            blocks += len(keys)
            if blocks % 64 == 0:
                write_json(Path(options["run"])/"accuracy_progress.json", dict(blocks=blocks))
            del predictions, target, dense, fast, output, inputs, nmse, x, y
        return dict(channel=dataset.channel, blocks=blocks, selected_frame_indices=_indices(dataset, options["frame_limit"]),
                    snr_db=dataset.snr_db.tolist(), checkpoint_smoke=bool(payload.get("smoke")),
                    model_config=dict(model.config), precision=model.config["precision"],
                    dataset_smoke=bool(dataset._config.get("smoke")),
                    maximum_probability_difference=maximum_probability_difference,
                    maximum_csi_increment_abs_difference=maximum_band_difference,
                    final_bit_disagreements=disagreements,
                    modes={mode: dict(per_snr=acc.report(dataset.snr_db, bootstrap_samples=200))
                           for mode, acc in accumulators.items()})


def _timing_worker(options):
    # This process never reads supervision, including during CUDA peak capture.
    from src.evaluation.icr_efficiency import _guard_truth_reads
    _guard_truth_reads()
    backend = DenseBackend(options["device"])
    if options["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    model, payload = load_model(options["checkpoint"], options["device"], input_view="raw")
    with CompactDataset(options["dataset"], input_view="raw") as dataset, torch.inference_mode():
        validate_dataset(payload, dataset)
        backend.synchronize()
        setup = time.perf_counter()-tick
        blocks = batches = 0
        preparation = 0.
        times = np.zeros(options["timing_repeats"])
        iterator = iter(input_batches(dataset, device=options["device"], batch_size=options["batch_size"],
            operator_backend=options["operator_backend"], precision=model.config["precision"],
            frame_indices=_indices(dataset, options["frame_limit"])))
        cold = None
        while True:
            backend.synchronize()
            tick = time.perf_counter()
            try:
                inputs, keys = next(iterator)
            except StopIteration:
                break
            backend.synchronize()
            preparation += time.perf_counter()-tick
            raw = inputs["initial_channel"]
            if cold is None:
                tick = time.perf_counter()
                output = model(**inputs)
                backend.synchronize()
                cold = time.perf_counter()-tick
                del output
            for repeat in range(options["timing_repeats"]):
                if isinstance(raw, DPSSChannel):
                    raw.clear_sample_cache()
                backend.synchronize()
                tick = time.perf_counter()
                output = model(**inputs)
                backend.synchronize()
                times[repeat] += time.perf_counter()-tick
                if not torch.isfinite(output.probabilities).all():
                    raise FloatingPointError("Nonfinite timed probabilities")
                del output
            blocks += len(keys)
            batches += 1
            del inputs, raw
        inference = float(statistics.median(times))
        environment = numerical_environment(backend)
        environment["precision"] = model.config["precision"] + "/" + (
            "float32" if model.config["precision"] == "complex64" else "float64")
        environment["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        environment["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
        return dict(operator_backend=options["operator_backend"], process_id=os.getpid(), blocks=blocks,
            setup_seconds=setup, input_preparation_seconds=preparation,
            inference_seconds_by_repeat=times.tolist(), median_inference_seconds=inference,
            inference_seconds_per_block=inference/blocks,
            preprocessing_and_inference_seconds=preparation+inference,
            end_to_end_seconds_per_block=(preparation+inference)/blocks,
            cold_first_forward_seconds=cold, peak_process_rss_bytes=process_peak_bytes(),
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()) if backend.device == "cuda" else None,
            peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()) if backend.device == "cuda" else None,
            numerical_environment=environment, test_truth_reads=False)


def _subprocess(kind, options):
    completed = subprocess.run([sys.executable, "-m", __name__, "--worker", kind], cwd=ROOT,
        input=json.dumps(options), text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"DPSS {kind} worker failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


def _compare_operators(dataset, checkpoint, run, *, device="cpu", batch_size=2, frame_limit=16,
                      timing_repeats=3, threads=4, diagnose_native_drift=False):
    if (device not in ("cpu", "cuda") or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (batch_size, timing_repeats, threads))
            or (frame_limit is not None and (isinstance(frame_limit, bool) or not isinstance(frame_limit, int) or frame_limit < 1))):
        raise ValueError("Invalid operator comparison controls")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    options = dict(dataset=str(Path(dataset).resolve()), checkpoint=str(Path(checkpoint).resolve()),
        validation=str(Path(dataset).resolve().parent/"validation.h5"), run=str(run),
        device=device, batch_size=batch_size, frame_limit=frame_limit, timing_repeats=timing_repeats, threads=threads)
    computation_code = source_fingerprint(INFERENCE_CODE+(__name__, "src.training.icr_mmse:configure_stage"))
    guard = DataIdentity({name: path for name, path in (
        ("test.h5", options["dataset"]), ("validation.h5", options["validation"]),
        ("model.h5", Path(dataset).resolve().parent/"model.h5"), ("checkpoint", options["checkpoint"]))})
    write_json(run/"progress.json", dict(phase="validation_equivalence"))
    equivalence = _subprocess("gradients", options)
    write_json(run/"equivalence.json", equivalence)
    double_equivalence = None
    if not equivalence["passed"] and not diagnose_native_drift:
        raise RuntimeError("DPSS output/gradient equivalence failed; see equivalence.json")
    if diagnose_native_drift:
        double_equivalence = _subprocess("gradients", dict(options, validation_precision="complex128"))
        write_json(run/"equivalence_complex128.json", double_equivalence)
        if not double_equivalence["passed"]:
            raise RuntimeError("DPSS double-precision control failed; no diagnostic accuracy/timing run")
    write_json(run/"progress.json", dict(phase="paired_test_accuracy"))
    report = _subprocess("accuracy", options)
    report["equivalence"] = equivalence
    report["double_equivalence"] = double_equivalence
    for mode in MODES:
        write_json(run/"progress.json", dict(phase="isolated_efficiency", operator_backend=mode))
        report["modes"][mode]["performance"] = _subprocess("timing", dict(options, operator_backend=mode))
        if report["modes"][mode]["performance"]["blocks"] != report["blocks"]:
            raise RuntimeError("Timing and accuracy sample counts differ")
    for mode, value in report["modes"].items():
        rows = value["per_snr"]
        bits = sum(row["total_bits"] for row in rows)
        errors = sum(row["bit_errors"][-1] for row in rows)
        nmse = sum(row["mean_nmse_linear"][-1]*row["blocks"] for row in rows)/report["blocks"]
        value["aggregate"] = dict(ber=errors/bits, bit_errors=errors, total_bits=bits,
                                   mean_nmse_linear=nmse, mean_nmse_db=float(10*np.log10(max(nmse, 1e-300))))
    dense, fast = [report["modes"][mode] for mode in MODES]
    report.update(schema="icr-dense-dpss-comparison-v1", complete=True, input_view="raw",
        checkpoint=options["checkpoint"], checkpoint_sha256=file_sha256(options["checkpoint"]),
        dataset=options["dataset"], data_identity=guard.files, controls=options,
        limited_sample=frame_limit is not None, code=computation_code,
        diagnostic_native_drift=bool(diagnose_native_drift),
        native_strict_equivalence_passed=equivalence["passed"],
        validation_status=("native_strict_passed" if equivalence["passed"] else
                           "double_control_passed_native_strict_failed_diagnostic_only"),
        comparison="identical checkpoint/weights, noise mode, precision, observations, pilots, masks and sample keys; only channel representation differs",
        timing="separate fresh process per backend; warm inference median over repeats plus each backend's own measured input preparation; excludes supervision/scoring; sample caches cleared per repeat; shared basis/noise caches warmed; cold forward/setup separate",
        memory="isolated inference processes; OS lifetime peak RSS; CUDA peaks include setup/cold/input/forward and all shared kernels; supervision is prohibited",
        differences=dict(ber=fast["aggregate"]["ber"]-dense["aggregate"]["ber"],
            nmse_db=fast["aggregate"]["mean_nmse_db"]-dense["aggregate"]["mean_nmse_db"],
            inference_speedup=dense["performance"]["median_inference_seconds"]/fast["performance"]["median_inference_seconds"],
            end_to_end_speedup=dense["performance"]["preprocessing_and_inference_seconds"]/fast["performance"]["preprocessing_and_inference_seconds"]))
    guard.assert_unchanged()
    if computation_code != source_fingerprint(INFERENCE_CODE+(__name__, "src.training.icr_mmse:configure_stage")):
        raise RuntimeError("Computation source changed during operator comparison")
    write_json(run/"comparison.json", report)
    plot_comparison(report, run)
    write_json(run/"progress.json", dict(phase="complete"))
    return report


def compare_operators(dataset, checkpoint, run, **options):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".operator_comparison.lock"):
        return _compare_operators(dataset, checkpoint, run, **options)


def plot_comparison(report, run):
    for metric, ylabel, filename in (("ber", "BER", "ber_snr.png"),
                                      ("mean_nmse_db", "Mean full-matrix CSI NMSE (dB)", "nmse_snr.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for mode, value in report["modes"].items():
            rows = value["per_snr"]
            values = [row[metric][-1] for row in rows]
            if metric == "ber":
                values = [v if v > 0 else .5/row["total_bits"] for v, row in zip(values, rows)]
                ax.semilogy(report["snr_db"], values, "o-", label=mode)
            else:
                ax.plot(report["snr_db"], values, "o-", label=mode)
        if metric == "mean_nmse_db":
            ax.plot(report["snr_db"], [r[metric][0] for r in report["modes"]["dense"]["per_snr"]], "k--", label="raw CSI")
        ax.set(xlabel="SNR (dB)", ylabel=ylabel, title=report["channel"]+" same-checkpoint operators")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        if metric == "ber":
            fig.text(.5, .01, "Zero errors displayed at 0.5 / total bits", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, 1))
        fig.savefig(Path(run)/filename, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    modes = report["modes"]
    for offset, (key, label) in enumerate((("inference_seconds_per_block", "forward"),
                                          ("end_to_end_seconds_per_block", "input + forward"))):
        axes[0].bar(np.arange(2)+(offset-.5)*.35,
                    [1000*v["performance"][key] for v in modes.values()], .35, label=label)
    axes[0].set(xticks=np.arange(2), xticklabels=list(modes), ylabel="ms per D block")
    axes[0].legend()
    key = "peak_cuda_allocated_bytes" if report["controls"]["device"] == "cuda" else "peak_process_rss_bytes"
    axes[1].bar(list(modes), [v["performance"][key]/2**20 for v in modes.values()])
    axes[1].set(ylabel="CUDA allocated peak (MiB)" if "cuda" in key else "Process peak RSS (MiB)")
    fig.tight_layout()
    fig.savefig(Path(run)/"runtime_memory.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Use python -m scripts.evaluate.icr_operator")
    opts = json.load(sys.stdin)
    torch.set_num_threads(opts["threads"])
    with threadpool_limits(opts["threads"]):
        result = {"gradients": _gradient_worker, "accuracy": _accuracy_worker, "timing": _timing_worker}[sys.argv[2]](opts)
    print(json.dumps(result))
