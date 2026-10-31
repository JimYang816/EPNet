"""Compare exact compensated inputs, gradients and throughput with a frozen cache pipeline."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from unittest.mock import patch

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.algorithms.icr_mmse.checkpoint import INFERENCE_CODE
from src.algorithms.icr_mmse.model import ICRMMSEnet, csi_nmse, layerwise_loss
from src.common.experiment import write_json
from src.data.compact import CompactDataset
from src.data import icr_training as accelerated
from src.evaluation.fingerprints import normalized_source, source_fingerprint
from src.training import icr_mmse as training


PIPELINE_MODULES = ("src.data.icr_dense_training", "src.data.icr_training")
ROOT = Path(__file__).resolve().parents[2]


def frozen_pipeline(ref):
    modules, hashes = [], {}
    for name in PIPELINE_MODULES:
        path = name.replace(".", "/") + ".py"
        source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)
        hashes[name] = hashlib.sha256(normalized_source(source).encode()).hexdigest()
        module = types.ModuleType("_cache_baseline_" + name.rsplit(".", 1)[-1])
        sys.modules[module.__name__] = module
        exec(compile(source, f"{ref}:{path}", "exec"), module.__dict__)
        modules.append(module)
    dense, pipeline = modules
    pipeline.DenseCoefficients = dense.DenseCoefficients
    pipeline.dense_input_batches = dense.dense_input_batches
    return pipeline, hashes


def relative(actual, expected):
    return float((actual-expected).norm()/expected.norm().clamp_min(1e-30))


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def compare_gradients(model, inputs, targets):
    result = {}
    for stage in "ABC":
        training.configure_stage(model, stage)
        captured = []
        for batch, target in zip(inputs, targets):
            model.zero_grad(set_to_none=True)
            output = model(**batch, correction_enabled=stage != "A", detach_feedback=stage == "B",
                           return_diagnostics=False)
            loss = layerwise_loss(output, target["classes"], target["reference_channel"], batch["data_mask"],
                lambda_h=.01 if stage != "A" else 0., csi_only=stage == "B")
            loss.backward()
            gradient = torch.cat([p.grad.detach().flatten() for p in model.parameters()
                                  if p.requires_grad and p.grad is not None])
            with torch.no_grad():
                captured.append((loss.detach(), gradient, csi_nmse(output, target["reference_channel"]),
                    output.symbols[:, -1, batch["data_mask"]].detach()))
            del output, loss
        before, after = captured
        errors = dict(loss_relative=relative(after[0], before[0]),
            gradient_relative=relative(after[1], before[1]), nmse_relative=relative(after[2], before[2]),
            bit_disagreements=int(((before[3].real < 0) != (after[3].real < 0)).sum()
                                  + ((before[3].imag < 0) != (after[3].imag < 0)).sum()))
        if max(errors[name] for name in ("loss_relative", "gradient_relative", "nmse_relative")) > 5e-5 or errors["bit_disagreements"]:
            raise AssertionError(f"Cache optimization changed {stage} loss/gradient/NMSE/BER: {errors}")
        result[stage] = errors
    return result


def paired_inputs(directory, baseline, model, device):
    results = {}
    for split in ("train", "validation"):
        with ExitStack() as stack:
            dataset = stack.enter_context(CompactDataset(directory/(split+".h5")))
            frames = sorted(set((0, dataset.frame_count//2, dataset.frame_count-1)))
            options = dict(device=device, batch_size=len(frames), frame_indices=frames,
                precision=model.config["precision"], operator_backend="dense_device",
                include_reference=True, compact_reference=False, prefetch_batches=0)
            first = stack.enter_context(baseline.training_batches(dataset,
                cache=baseline.ReferenceCache(256*2**20), **options))
            second = stack.enter_context(accelerated.training_batches(dataset,
                cache=accelerated.ReferenceCache(256*2**20), **options))
            rows, blocks = [], 0
            for (a, ta, ka), (b, tb, kb) in zip(first, second, strict=True):
                if ka != kb:
                    raise AssertionError("Cache optimization changed sample ordering")
                for name in a:
                    torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
                for name in ta:
                    torch.testing.assert_close(ta[name], tb[name], rtol=0, atol=0)
                blocks += len(ka)
                if ka[0][0] in (0, len(dataset.snr_db)-1):
                    rows.append(dict(keys=ka, stages=compare_gradients(model, (a, b), (ta, tb))))
            results[split] = dict(blocks=blocks, all_snr=True, input_max_abs=0., target_max_abs=0., gradients=rows)
    return results


def timings(directory, baseline, original, options):
    results = []
    with CompactDataset(directory/"train.h5") as train, CompactDataset(directory/"validation.h5") as validation:
        frames = np.linspace(0, train.frame_count-1, min(options.frames, train.frame_count), dtype=int).tolist()
        val_frames = np.linspace(0, validation.frame_count-1, min(2, validation.frame_count), dtype=int).tolist()
        for stage in "ABC":
            rows = []
            for label, pipeline in (("baseline", baseline), ("cached", accelerated)):
                model = deepcopy(original)
                training.configure_stage(model, stage)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
                cache = pipeline.ReferenceCache(512*2**20)
                for repeat in range(options.repeats+1):
                    sync(options.device)
                    started = time.perf_counter()
                    with patch.object(training, "training_batches", pipeline.training_batches):
                        metrics = training.run_epoch(model, train, options.device, stage=stage,
                            batch_size=options.batch_size, frame_indices=frames, lambda_h=.01, optimizer=optimizer,
                            operator_backend="dense_device", prefetch_batches=2, reference_cache=cache)
                        validation_metrics = training.run_epoch(model, validation, options.device, stage=stage,
                            batch_size=options.batch_size, frame_indices=val_frames, lambda_h=.01,
                            operator_backend="dense_device", prefetch_batches=2, reference_cache=cache)
                    sync(options.device)
                    rows.append(dict(mode=label, warmup=repeat == 0, repeat=repeat,
                        seconds=time.perf_counter()-started, metrics=metrics, validation=validation_metrics,
                        cache_bytes=cache.bytes, cache_hits=cache.hits, cache_misses=cache.misses))
                del model, optimizer, cache
            before = [r for r in rows if r["mode"] == "baseline"]
            after = [r for r in rows if r["mode"] == "cached"]
            for a, b in zip(before, after, strict=True):
                for group in ("metrics", "validation"):
                    if a[group]["errors"] != b[group]["errors"] or a[group]["per_snr"] != b[group]["per_snr"]:
                        raise AssertionError("Cached training changed paired BER")
                    for field in ("loss", "initial_nmse", "corrected_nmse"):
                        if a[group][field] is not None and not np.isclose(a[group][field], b[group][field], rtol=5e-5, atol=1e-9):
                            raise AssertionError(f"Cached training changed {group} {field}")
            base_seconds = statistics.median(r["seconds"] for r in before if not r["warmup"])
            cached_seconds = statistics.median(r["seconds"] for r in after if not r["warmup"])
            results.append(dict(stage=stage, baseline_seconds=base_seconds, cached_seconds=cached_seconds,
                speedup=base_seconds/cached_seconds, paired_metrics_passed=True, trials=rows,
                training_frames=frames, validation_frames=val_frames))
            print(json.dumps(dict(channel=train.channel, **{k: v for k, v in results[-1].items()
                if k not in ("trials", "training_frames", "validation_frames")})), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/training/icr_mmse.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.frames, args.batch_size, args.repeats, args.threads) < 1:
        parser.error("Frame/batch/repeat/thread counts must be positive")
    # Continuous Adam trajectories amplify nondeterministic CUDA convolution
    # reductions even when repeating the baseline itself. Use deterministic
    # kernels only in this diagnostic; formal training settings stay unchanged.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    baseline, old_hashes = frozen_pipeline(args.baseline_ref)
    after = source_fingerprint(training.TRAINING_CODE)
    before = dict(after, **old_hashes)
    changed = sorted(k for k in before if before[k] != after[k])
    if changed != sorted(PIPELINE_MODULES):
        raise ValueError("Benchmark requires exactly the two cache-pipeline modules to change")
    report = dict(passed=False, input_view="compensated", training_code_before=before,
        training_code_after=after, inference_code=source_fingerprint(INFERENCE_CODE),
        baseline_commit=subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip(),
        changes=changed, device=args.device, threads=args.threads, channels={}, test_access=False,
        diagnostic_deterministic_algorithms=True, formal_training_settings_changed=False,
        scope="small train/validation subsets; timings may coexist with the original training service")
    with threadpool_limits(args.threads):
        for channel in ("NOF1", "NCS1"):
            directory = (ROOT/config["datasets"][channel]).resolve()
            torch.manual_seed(config["training"]["seed"])
            model = ICRMMSEnet(**config["model"]).to(args.device)
            with torch.no_grad():
                model.corrector.head.weight[:2*model.corrector.width].normal_(std=.001)
            paired = paired_inputs(directory, baseline, model, args.device)
            timed = timings(directory, baseline, model, args)
            report["channels"][channel] = dict(directory=str(directory), paired=paired, timings=timed)
            write_json(args.output, report)
            del model
    if source_fingerprint(training.TRAINING_CODE) != after:
        raise RuntimeError("Code changed during cache validation")
    report["passed"] = True
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
