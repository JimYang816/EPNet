"""Isolated training throughput/memory trials with paired numerical checks.

Only train/validation are read. Timed steps include loss, backward, clipping,
Adam updates, input preparation, supervision, and the original epoch metrics.
The dense reference path reproduces the previous synchronous training loop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from threading import Thread
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE, load_model, validate_dataset
from src.algorithms.icr_mmse.dpss import base_error, reference_energy
from src.algorithms.icr_mmse.model import ICRMMSEnet, csi_nmse, layerwise_loss
from src.common.experiment import file_sha256, write_json
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches, supervision
from src.data.icr_training import ReferenceCache, training_batches
from src.evaluation.fingerprints import DataIdentity, source_fingerprint
from src.evaluation.icr_mmse import process_peak_bytes
from src.training.icr_mmse import configure_stage, run_epoch

ROOT = Path(__file__).resolve().parents[2]
MODES = ("dense_reference", "dpss_full_reference", "compact", "prefetch")


def _modes(backend):
    return MODES if backend == "dpss" else ("dense_reference", "dense_cached", "dense_device", "prefetch")


def _frames(dataset, limit):
    return np.linspace(0, dataset.frame_count-1, min(limit, dataset.frame_count), dtype=int).tolist()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _model(options, dataset):
    torch.manual_seed(options["seed"])
    if options.get("checkpoint"):
        model, payload = load_model(options["checkpoint"], options["device"], input_view=dataset.input_view)
        validate_dataset(payload, dataset)
        if model.config != ICRMMSEnet(**options["model_config"]).config:
            raise ValueError("Benchmark checkpoint architecture differs from the requested model")
    else:
        model = ICRMMSEnet(**options["model_config"]).to(options["device"])
        # Nonzero repairs make equivalence checks exercise the learned path.
        with torch.no_grad():
            model.corrector.head.weight[:2*model.corrector.width].normal_(std=.001)
    return model


def _reference_epoch(model, dataset, device, *, stage, batch_size, frame_indices,
                     snr_indices, optimizer, operator_backend="dense"):
    """Frozen operation order of the original loop, for a measured baseline."""
    model.train()
    total_loss = nmse_total = raw_total = 0.
    count = errors = bits = nmse_count = 0
    by_snr = {float(value): [0, 0] for value in dataset.snr_db}
    for inputs, keys in input_batches(dataset, device=device, batch_size=batch_size,
            frame_indices=frame_indices, snr_indices=snr_indices,
            precision=model.config["precision"], operator_backend=operator_backend):
        result = model(**inputs, correction_enabled=stage != "A", detach_feedback=stage == "B",
                       reuse_base_energy=False)
        target = supervision(dataset, keys, device=device, include_reference=stage != "A",
                             precision=model.config["precision"])
        reference = target.get("reference_channel")
        loss = layerwise_loss(result, target["classes"], reference, inputs["data_mask"],
                              lambda_h=.1 if stage != "A" else 0., csi_only=stage == "B")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reference loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not parameters or any(not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Nonfinite reference gradient")
        torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach()) * len(keys)
        count += len(keys)
        prediction, labels = result.symbols[:, -1, inputs["data_mask"]].detach(), target["symbols"]
        per_example = ((prediction.real < 0) != (labels.real < 0)).long().sum(-1)
        per_example += ((prediction.imag < 0) != (labels.imag < 0)).long().sum(-1)
        errors += int(per_example.sum())
        bits += 2*labels.numel()
        for index, key in enumerate(keys):
            row = by_snr[float(dataset.snr_db[key[0]])]
            row[0] += int(per_example[index])
            row[1] += 2*labels.shape[-1]
        if reference is not None:
            values = csi_nmse(result, reference).detach()
            raw = base_error(inputs["initial_channel"], reference) / reference_energy(reference).clamp_min(1e-30)
            nmse_total += float(values[:, -1].sum())
            raw_total += float(raw.sum())
            nmse_count += len(keys)
    return dict(loss=total_loss/count, ber=errors/bits, errors=errors, bits=bits,
        initial_nmse=raw_total/nmse_count if nmse_count else None,
        corrected_nmse=nmse_total/nmse_count if nmse_count else None,
        per_snr=[dict(snr_db=snr, ber=row[0]/row[1], errors=row[0], bits=row[1])
                 for snr, row in by_snr.items() if row[1]])


class _Monitor:
    def __init__(self, enabled):
        self.process, self.samples = None, []
        if enabled:
            self.process = subprocess.Popen(["nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits",
                "--loop-ms=200"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = Thread(target=self._read, daemon=True)
            self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.samples.append((time.perf_counter(), *map(float, line.strip().split(","))))
            except ValueError:
                pass

    def finish(self, intervals):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.thread.join(timeout=10)
            self.process.stdout.close()
        values = [row[1:] for row in self.samples if any(a <= row[0] <= b for a, b in intervals)]
        return dict(samples=len(values),
            mean_gpu_utilization_percent=float(np.mean([v[0] for v in values])) if values else None,
            sampled_device_peak_mib=max(v[1] for v in values) if values else None,
            mean_power_w=float(np.mean([v[2] for v in values])) if values else None)


def _timing_worker(options):
    device, stage, mode = options["device"], options["stage"], options["mode"]
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    monitor = _Monitor(cuda)
    intervals, result = [], {}
    try:
        with CompactDataset(Path(options["data_directory"])/"train.h5", input_view="raw" if options["operator_backend"] == "dpss" else "compensated") as dataset:
            if dataset.split != "train":
                raise ValueError("Training performance trials require the train split")
            model = _model(options, dataset)
            configure_stage(model, stage)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
            frames, selected = _frames(dataset, options["frame_limit"]), [len(dataset.snr_db)//2]
            cache = ReferenceCache(options["reference_cache_mib"]*2**20)
            controls = dict(stage=stage, batch_size=options["batch_size"],
                            frame_indices=frames, snr_indices=selected, optimizer=optimizer)

            def epoch():
                if mode in ("dense_reference", "dpss_full_reference"):
                    return _reference_epoch(model, dataset, device, **controls,
                        operator_backend="dense" if mode == "dense_reference" else "dpss")
                selected_backend = "dense" if mode == "dense_cached" else options["operator_backend"]
                return run_epoch(model, dataset, device, **controls, operator_backend=selected_backend,
                    compact_reference=selected_backend == "dpss",
                    prefetch_batches=options["prefetch_batches"] if mode == "prefetch" else 0,
                    reference_cache=cache)

            _sync(device)
            started = time.perf_counter()
            epoch()  # Full cold pass: allocate Adam state, warm kernels, populate RAM cache.
            _sync(device)
            cold = time.perf_counter()-started
            elapsed = []
            for _ in range(options["repeats"]):
                _sync(device)
                started = time.perf_counter()
                metrics = epoch()
                _sync(device)
                ended = time.perf_counter()
                intervals.append((started, ended))
                elapsed.append(ended-started)
            median = statistics.median(elapsed)
            blocks = 2*len(frames)*len(selected)
            result = dict(status="ok", channel=dataset.channel, stage=stage, mode=mode,
                batch_size=options["batch_size"], blocks_per_repeat=blocks,
                full_batches_per_repeat=2*(len(frames)//options["batch_size"]),
                selected_frame_indices=frames, snr_db=[float(dataset.snr_db[i]) for i in selected],
                cold_epoch_seconds=cold, seconds_by_repeat=elapsed, median_seconds=median,
                blocks_per_second=blocks/median, milliseconds_per_block=1000*median/blocks,
                final_training_metrics=metrics, reference_cache_bytes=cache.bytes,
                reference_cache_hits=cache.hits, reference_cache_misses=cache.misses,
                optimizer_updates=True, checkpoint_saved=False)
    except torch.cuda.OutOfMemoryError as error:
        result = dict(status="oom", stage=stage, mode=mode, batch_size=options["batch_size"], reason=str(error))
    finally:
        result["monitor"] = monitor.finish(intervals)
    result.update(process_id=os.getpid(), peak_process_rss_bytes=process_peak_bytes(),
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None,
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()) if cuda else None,
        cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory if cuda else None,
        torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if cuda else None,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    return result


def _relative(actual, expected):
    return float((actual-expected).norm()/expected.norm().clamp_min(1e-30))


def _equivalence_worker(options):
    device = options["device"]
    cache = ReferenceCache(options["reference_cache_mib"]*2**20)
    gradients, curves = [], {"dense_reference": [], "prefetch": []}
    with CompactDataset(Path(options["data_directory"])/"validation.h5", input_view="raw" if options["operator_backend"] == "dpss" else "compensated") as dataset:
        if dataset.split != "validation":
            raise ValueError("Numerical/configuration checks require validation, never test")
        model = _model(options, dataset)
        precision = model.config["precision"]
        selected = sorted(set((0, len(dataset.snr_db)//2, len(dataset.snr_db)-1)))
        probe_frames = sorted(set((0, dataset.frame_count//2, dataset.frame_count-1)))
        batch_options = dict(device=device, batch_size=1, frame_indices=probe_frames,
                             snr_indices=selected, precision=precision)
        dense = input_batches(dataset, **batch_options)
        backend = options["operator_backend"]
        with training_batches(dataset, **batch_options, operator_backend=backend,
                compact_reference=backend == "dpss",
                bandwidth=model.config["bandwidth"], cache=cache, prefetch_batches=0) as fast:
            for (first, keys), (second, target, other_keys) in zip(dense, fast):
                if keys != other_keys:
                    raise AssertionError("Paired numerical check changed sample order")
                full = supervision(dataset, keys, device=device, precision=precision)
                for stage in "ABC":
                    configure_stage(model, stage)
                    captured = []
                    for inputs, reference, reuse, diagnostics in (
                            (first, full["reference_channel"], False, True),
                            (second, target["reference_channel"], True, False)):
                        model.zero_grad(set_to_none=True)
                        output = model(**inputs, correction_enabled=stage != "A", detach_feedback=stage == "B",
                                       reuse_base_energy=reuse, return_diagnostics=diagnostics)
                        loss = layerwise_loss(output, full["classes"], reference, inputs["data_mask"],
                            lambda_h=.1 if stage != "A" else 0., csi_only=stage == "B")
                        loss.backward()
                        gradient = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.requires_grad])
                        with torch.no_grad():
                            captured.append((output.probabilities.detach(), output.bands.detach(), loss.detach(),
                                gradient, csi_nmse(output, reference).detach(), output.symbols[:, -1].detach()))
                        del output, loss
                    a, b = captured
                    tolerance = 5e-4 if precision == "complex64" else 2e-9
                    difference = int(((a[-1].real < 0) != (b[-1].real < 0)).sum()
                                     + ((a[-1].imag < 0) != (b[-1].imag < 0)).sum())
                    errors = dict(probability_max_abs=float((a[0]-b[0]).abs().max()),
                        csi_increment_relative=_relative(b[1], a[1]), loss_relative=_relative(b[2], a[2]),
                        gradient_relative=_relative(b[3], a[3]), full_nmse_relative=_relative(b[4], a[4]))
                    if isinstance(second["initial_channel"], torch.Tensor):
                        errors["dense_input_relative"] = _relative(second["initial_channel"], first["initial_channel"])
                    gradients.append(dict(keys=keys, stage=stage, **errors, bit_disagreements=difference,
                        tolerance=tolerance, passed=max(errors.values()) < tolerance and difference == 0))
                    del captured, a, b, gradient
        model.eval().requires_grad_(False)
        frames = _frames(dataset, options["validation_frames"])
        rows = {mode: [[0, 0, 0., 0., 0] for _ in dataset.snr_db] for mode in curves}
        disagreements = 0
        batches = dict(device=device, batch_size=2, frame_indices=frames, precision=precision)
        dense = input_batches(dataset, **batches)
        with torch.no_grad(), training_batches(dataset, **batches, operator_backend=backend,
                compact_reference=backend == "dpss", bandwidth=model.config["bandwidth"],
                cache=cache, prefetch_batches=options["prefetch_batches"]) as fast:
            for (inputs, keys), (other, target, other_keys) in zip(dense, fast):
                if keys != other_keys:
                    raise AssertionError("Paired metric check changed sample order")
                full = supervision(dataset, keys, device=device, precision=precision)
                predictions = []
                for mode, prepared, reference in (("dense_reference", inputs, full["reference_channel"]),
                                                  ("prefetch", other, target["reference_channel"])):
                    output = model(**prepared, return_diagnostics=mode == "dense_reference",
                                   reuse_base_energy=mode != "dense_reference")
                    symbols = output.symbols[:, -1, prepared["data_mask"]]
                    expected = full["symbols"]
                    bit_errors = (((symbols.real < 0) != (expected.real < 0)).sum(-1)
                                  + ((symbols.imag < 0) != (expected.imag < 0)).sum(-1)).cpu().tolist()
                    nmse = csi_nmse(output, reference)[:, -1].cpu().tolist()
                    raw = (base_error(prepared["initial_channel"], reference)
                           / reference_energy(reference).clamp_min(1e-30)).cpu().tolist()
                    for key, error, value, initial in zip(keys, bit_errors, nmse, raw):
                        row = rows[mode][key[0]]
                        row[0] += error
                        row[1] += 2*expected.shape[-1]
                        row[2] += value
                        row[3] += initial
                        row[4] += 1
                    predictions.append(symbols.detach())
                    del output
                a, b = predictions
                disagreements += int(((a.real < 0) != (b.real < 0)).sum()
                                     + ((a.imag < 0) != (b.imag < 0)).sum())
                del predictions, full, a, b
        for mode in curves:
            for snr, (errors, bits, nmse, raw, blocks) in zip(dataset.snr_db, rows[mode]):
                curves[mode].append(dict(snr_db=float(snr), ber=errors/bits, errors=errors, bits=bits,
                    mean_nmse_linear=nmse/blocks, mean_nmse_db=float(10*np.log10(max(nmse/blocks, 1e-300))),
                    initial_nmse_linear=raw/blocks, blocks=blocks))
        curve_error = max(abs(a["mean_nmse_linear"]-b["mean_nmse_linear"])
                          / max(abs(a["mean_nmse_linear"]), 1e-30)
                          for a, b in zip(curves["dense_reference"], curves["prefetch"]))
        return dict(passed=all(row["passed"] for row in gradients) and disagreements == 0 and curve_error < 5e-4,
            channel=dataset.channel, model_config=model.config, gradients=gradients, curves=curves,
            bit_disagreements=disagreements, maximum_curve_nmse_relative_error=curve_error,
            validation_frame_indices=frames, checkpoint=options.get("checkpoint"),
            weights="same validated checkpoint" if options.get("checkpoint") else "same seeded diagnostic weights; no fitted accuracy claim",
            test_access=False, algorithm_changes=False, operator_backend=backend)


def choose_batch(trials, stages, *, memory_headroom_bytes=2*2**30, tolerance=.05):
    candidates = []
    for size in sorted({row["batch_size"] for row in trials if row["mode"] == "prefetch"}):
        rows = [row for row in trials if row["mode"] == "prefetch" and row["batch_size"] == size]
        if {row["stage"] for row in rows if row["status"] == "ok"} != set(stages):
            continue
        if any(row["cuda_total_bytes"] is not None and
               max(row["peak_cuda_reserved_bytes"], row["monitor"]["sampled_device_peak_mib"]*2**20
                   if row["monitor"]["sampled_device_peak_mib"] else 0)
               > row["cuda_total_bytes"]-memory_headroom_bytes for row in rows):
            continue
        candidates.append(dict(batch_size=size, combined_stage_seconds=sum(row["median_seconds"] for row in rows),
            peak_cuda_reserved_bytes=max((row["peak_cuda_reserved_bytes"] or 0) for row in rows)))
    if not candidates:
        raise RuntimeError("No tested batch is valid for all stages within the memory budget")
    fastest = min(row["combined_stage_seconds"] for row in candidates)
    selected = min((row for row in candidates if row["combined_stage_seconds"] <= fastest*(1+tolerance)),
                    key=lambda row: row["batch_size"])
    return dict(**selected, candidates=candidates, memory_headroom_bytes=memory_headroom_bytes,
                throughput_tolerance=tolerance,
                criterion="smallest batch within the configured tolerance of the fastest summed stage time, with memory headroom")


def _subprocess(kind, options):
    result = subprocess.run([sys.executable, "-m", "src.evaluation.icr_training_perf", "--worker", kind],
        cwd=ROOT, input=json.dumps(options), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ICR training {kind} worker failed:\n{result.stderr[-8000:]}")
    return json.loads(result.stdout)


def benchmark_training(data_directory, run, *, model_config, checkpoint=None, device="cuda",
        batch_sizes=(2, 4, 8, 16, 32, 64, 128), stages=("A", "B", "C"), frame_limit=128,
        validation_frames=4, repeats=3, threads=4, prefetch_batches=2, reference_cache_mib=512,
        seed=20261028, ablations=True, operator_backend="dense_device"):
    if (device not in ("cpu", "cuda") or not batch_sizes or len(set(batch_sizes)) != len(batch_sizes)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (*batch_sizes, frame_limit, validation_frames, repeats, threads))
            or not stages or len(set(stages)) != len(stages) or any(s not in "ABC" for s in stages)
            or operator_backend not in ("dense_device", "dpss")):
        raise ValueError("Invalid ICR training benchmark controls")
    directory, run = Path(data_directory).resolve(), Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    paths = {name: directory/name for name in ("train.h5", "validation.h5", "model.h5")}
    if checkpoint:
        paths["checkpoint"] = Path(checkpoint).resolve()
    identity = DataIdentity(paths)
    options = dict(data_directory=str(directory), run=str(run), model_config=model_config,
        checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None, device=device,
        frame_limit=frame_limit, validation_frames=validation_frames, repeats=repeats, threads=threads,
        prefetch_batches=prefetch_batches, reference_cache_mib=reference_cache_mib, seed=seed,
        operator_backend=operator_backend)
    write_json(run/"progress.json", dict(phase="paired_equivalence"))
    equivalence = _subprocess("equivalence", options)
    write_json(run/"equivalence.json", equivalence)
    if not equivalence["passed"]:
        raise RuntimeError("Training/metric equivalence failed; see equivalence.json")
    trials = []
    jobs = [(stage, "prefetch", size) for size in batch_sizes for stage in stages]
    if ablations:
        jobs = [(stage, mode, min(batch_sizes)) for mode in _modes(operator_backend)[:-1] for stage in stages] + jobs
    for stage, mode, size in jobs:
        write_json(run/"progress.json", dict(phase="timing", stage=stage, mode=mode, batch_size=size,
                                             completed=len(trials), total=len(jobs)))
        trial = _subprocess("timing", dict(options, stage=stage, mode=mode, batch_size=size))
        trials.append(trial)
        write_json(run/(mode+"_"+stage+"_b"+str(size)+".json"), trial)
    selection = choose_batch(trials, stages)
    report = dict(schema="icr-training-performance-v1", complete=True, options=options,
        channel=equivalence["channel"], equivalence=equivalence, trials=trials, selection=selection,
        data_identity=identity.files, code=source_fingerprint(INFERENCE_CODE+(
            "src.training.icr_mmse", "src.data.icr_training", "src.data.icr_dense_training",
            "src.evaluation.icr_training_perf")),
        checkpoint_sha256=file_sha256(checkpoint) if checkpoint else None,
        timing="fresh process per stage/mode/batch; one full cold epoch reported separately; median of warm complete epochs includes input/supervision/metrics/backward/clipping/Adam; both D blocks and identical evenly spaced train frames at middle SNR",
        memory="process lifetime peak RSS and CUDA allocated/reserved peaks include cold setup, reference preparation, backward and optimizer; sampled nvidia-smi device usage includes context/other device users",
        precision="original complex/real precision; no AMP, TF32 setting changes, truncation or gradient accumulation",
        test_access=False, learning_rate_scaling=False,
        limitation="batch changes alter optimizer trajectories; equivalence uses fixed weights; this benchmark does not establish converged training quality")
    identity.assert_unchanged()
    write_json(run/"comparison.json", report)
    _plot(report, run)
    write_json(run/"progress.json", dict(phase="complete", selection=selection))
    return report


def _plot(report, run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, ylabel, filename in (("ber", "BER", "ber_snr.png"),
                                 ("mean_nmse_db", "Mean full-matrix CSI NMSE (dB)", "nmse_snr.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for mode, rows in report["equivalence"]["curves"].items():
            values = [row[key] if key != "ber" or row[key] else .5/row["bits"] for row in rows]
            (ax.semilogy if key == "ber" else ax.plot)([row["snr_db"] for row in rows], values, "o-", label=mode)
        ax.set(xlabel="SNR (dB)", ylabel=ylabel, title=report["channel"]+" fixed weights / validation")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(run)/filename, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for stage in "ABC":
        rows = [r for r in report["trials"] if r["mode"] == "prefetch" and r["stage"] == stage and r["status"] == "ok"]
        sizes = [r["batch_size"] for r in rows]
        axes[0].plot(sizes, [r["blocks_per_second"] for r in rows], "o-", label=stage)
        axes[1].plot(sizes, [(r["peak_cuda_allocated_bytes"] or 0)/2**30 for r in rows], "o-", label=stage)
        axes[2].plot(sizes, [r["monitor"]["mean_gpu_utilization_percent"] or 0 for r in rows], "o-", label=stage)
    for ax, ylabel in zip(axes, ("D blocks / second", "Peak CUDA allocated (GiB)", "Mean GPU utilization (%)")):
        ax.set(xlabel="Batch size", ylabel=ylabel)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=.3)
        ax.legend(title="Stage")
    fig.tight_layout()
    fig.savefig(Path(run)/"batch_speed_memory.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Use python -m scripts.evaluate.icr_training_perf")
    options = json.load(sys.stdin)
    torch.set_num_threads(options["threads"])
    with threadpool_limits(options["threads"]):
        result = {"timing": _timing_worker, "equivalence": _equivalence_worker}[sys.argv[2]](options)
    print(json.dumps(result))
