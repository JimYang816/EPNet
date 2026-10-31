"""Validation-only sequential model selection, followed by fresh seed repeats."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from src.common.experiment import write_json


def sequential_select(model_config, lambda_h, assess, *, lambda_values=(.01, .1, 1.),
                      layers=(4, 8, 12, 22)):
    """Select lambda_H, then T while preserving the configured correction bandwidth.

    assess(model_config, lambda_h, candidate_name) must return validation
    selection metrics. It is deliberately decoupled from test scoring.
    """
    selected, selected_lambda = deepcopy(model_config), float(lambda_h)
    history = []
    for stage, candidates in (("lambda_h", lambda_values), ("layers", layers)):
        rows = []
        if not candidates:
            raise ValueError("Each sequential selection stage requires candidates")
        for value in candidates:
            candidate = deepcopy(selected)
            candidate_lambda = selected_lambda
            if stage == "lambda_h":
                candidate_lambda = float(value)
            else:
                candidate[stage] = int(value)
            name = stage+"_"+str(value).replace(".", "p")
            metrics = assess(candidate, candidate_lambda, name)
            row = dict(stage=stage, value=value, model_config=candidate, lambda_h=candidate_lambda,
                       selection=metrics)
            rows.append(row)
        eligible = [row for row in rows if row["selection"]["csi_improved"]]
        if not eligible:
            raise RuntimeError(f"No {stage} validation candidate improved CSI; test scoring is prohibited")
        winner = min(eligible, key=lambda row: (row["selection"]["ber"], row["selection"]["corrected_nmse"]))
        selected, selected_lambda = deepcopy(winner["model_config"]), winner["lambda_h"]
        history.append(dict(stage=stage, candidates=rows, selected=winner))
    return dict(model_config=selected, lambda_h=selected_lambda, stages=history,
                criterion="validation BER subject to mean full-matrix CSI NMSE improvement",
                test_used=False)


def match_once_budget(data_directory, checkpoint, backend, *, repeats=(1, 2, 3, 4, 6, 8),
                      tolerance=.1, batch_size=2, frame_limit=4, timing_repeats=3):
    """Match validation end-to-end latency by useful equalizer state updates.

    CSI is corrected exactly once. Additional equalizer substeps share each
    layer's parameters and preserve the warmup parameter shapes. Timing uses
    validation inputs only; labels and reference CSI are never loaded.
    """
    from src.algorithms.icr_mmse.checkpoint import load_model, validate_dataset
    from src.data.compact import CompactDataset
    from src.data.icr_mmse import input_batches
    if not repeats or any(not isinstance(r, int) or r < 1 for r in repeats) or timing_repeats < 1:
        raise ValueError("Budget matching requires positive equalizer and timing repeats")
    if not 0 < tolerance < 1:
        raise ValueError("Budget tolerance must be between zero and one")
    model, payload = load_model(checkpoint, backend.device)
    precision = model.config["precision"]
    with CompactDataset(Path(data_directory)/"validation.h5", input_view="compensated") as dataset:
        # load_model already verifies checkpoint metadata; validation metadata is
        # checked by the same function used by formal evaluation.
        validate_dataset(payload, dataset)
        first, _ = next(input_batches(dataset, device=backend.device, batch_size=batch_size,
                                     frame_limit=1, precision=precision))
        def measure():
            with torch.inference_mode():
                model(**first)
                elapsed, count = [], 0
                for _ in range(timing_repeats):
                    backend.synchronize()
                    started = time.perf_counter()
                    count = 0
                    for inputs, keys in input_batches(dataset, device=backend.device, batch_size=batch_size,
                                                     frame_limit=frame_limit, precision=precision):
                        model(**inputs)
                        count += len(keys)
                    backend.synchronize()
                    elapsed.append((time.perf_counter()-started)/count)
            return float(np.median(elapsed))
        target = measure()
        trials = []
        model.config.update(correction_mode="once", correction_passes=1)
        for value in repeats:
            model.config["equalizer_repeats"] = value
            measured = measure()
            trials.append(dict(equalizer_repeats=value, seconds_per_block=measured, ratio=measured/target))
    selected = min(trials, key=lambda row: abs(row["ratio"]-1))
    return dict(equalizer_repeats=selected["equalizer_repeats"], target_seconds_per_block=target,
                ratio=selected["ratio"], matched=abs(selected["ratio"]-1) <= tolerance,
                tolerance=tolerance, trials=trials, split="validation", frame_limit=frame_limit,
                timing_repeats=timing_repeats, correction_events=1, precision=precision,
                definition="median preprocessing + all inference iteration latency; no labels/reference access")


def _train_dataset_udnet(data_directory, run, backend, options, *, resume):
    from src.training.udnet import train
    from src.data.udnet_cache import remove_features
    controls = deepcopy(options.get("training", {}))
    controls["input_view"] = "compensated"
    checkpoint = train(data_directory, run, backend, model_config=options.get("model"),
                       resume=resume and (Path(run)/"latest.pt").exists(), **controls)
    features = Path(run)/"features"
    if features.exists() and not options.get("keep_feature_cache", False):
        for path in features.iterdir():
            if path.is_dir():
                remove_features(path, features)
            elif path.suffix == ".lock":
                path.unlink()
        if not any(features.iterdir()):
            features.rmdir()
    return checkpoint


def run_channel(data_directory, run, backend, config, *, resume=False, udnet_checkpoint=None):
    """Fixed-band coverage -> lambda/T selection -> configured seed training/test runs."""
    from src.evaluation.icr_mmse import coverage, evaluate
    from src.training.icr_mmse import train
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if config.get("input_view", "compensated") != "compensated":
        raise ValueError("ICR-MMSE-Net experiments require the fixed compensated CSI input view")
    model_config = deepcopy(config["model"])
    model_config.setdefault("precision", "complex64")
    model_config.setdefault("bandwidth", 16)
    training = deepcopy(config["training"])
    selection = deepcopy(config.get("selection", {}))
    if "bandwidth" in selection:
        raise ValueError("Correction bandwidth is fixed by model.bandwidth; remove selection.bandwidth")
    evaluation = deepcopy(config.get("evaluation", {}))
    seed = training.pop("seed", 20261028)
    lambda_h = training.pop("lambda_h", .1)
    seeds = selection.get("seeds", (seed,))
    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Final experiment requires a nonempty sequence of distinct integer random seeds")
    coverage(data_directory, run/"coverage", bandwidths=(model_config["bandwidth"],),
             batch_size=training.get("batch_size", 2), frame_limit=training.get("frame_limit"),
             device=backend.device, precision=model_config["precision"],
             operator_backend=training.get("operator_backend", "dense"),
             reference_cache_mib=training.get("reference_cache_mib", 512))

    def assess(candidate, weight, name):
        candidate_run = run/"selection"/name
        checkpoint = train(data_directory, candidate_run, backend, model_config=candidate,
                           lambda_h=weight, seed=seed, resume=resume, **training)
        report = evaluate(Path(data_directory)/"validation.h5", checkpoint, candidate_run/"validation", backend,
            require_test=False, batch_size=evaluation.get("batch_size", 2),
            frame_limit=training.get("frame_limit"), bootstrap_samples=evaluation.get("bootstrap_samples", 2000))
        return report["selection"]

    if selection.get("enabled", True):
        selected = sequential_select(model_config, lambda_h, assess,
            lambda_values=selection.get("lambda_h", (.01, .1, 1.)),
            layers=selection.get("layers", (4, 8, 12, 22)))
    else:
        selected = dict(model_config=model_config, lambda_h=lambda_h, stages=[],
                        criterion="prespecified configuration; no validation search", test_used=False)
    write_json(run/"selection.json", selected)
    if evaluation.get("include_baselines", True) and udnet_checkpoint is None:
        udnet_checkpoint = _train_dataset_udnet(data_directory, run/"udnet_compensated", backend,
            config.get("udnet", {}), resume=resume)
    reports = []
    for seed in seeds:
        seed_run = run/("seed_"+str(seed))
        checkpoint = train(data_directory, seed_run, backend, model_config=selected["model_config"],
                           lambda_h=selected["lambda_h"], seed=seed, resume=resume, **training)
        options = dict(evaluation)
        options.setdefault("include_baselines", True)
        options.setdefault("frame_limit", training.get("frame_limit"))
        report = evaluate(Path(data_directory)/"test.h5", checkpoint, seed_run/"evaluation", backend,
                          udnet_checkpoint=udnet_checkpoint, **options)
        fixed = seed_run/"warmup.pt"
        if fixed.exists():
            evaluate(Path(data_directory)/"test.h5", fixed, seed_run/"fixed_csi", backend,
                batch_size=evaluation.get("batch_size", 2), frame_limit=training.get("frame_limit"),
                bootstrap_samples=evaluation.get("bootstrap_samples", 2000))
        ablation_reports = {}
        for name in config.get("ablations", ("once", "no_uncertainty", "no_gate", "once_budget_matched")):
            ablation_config = deepcopy(selected["model_config"])
            budget = None
            if name in ("once", "once_budget_matched"):
                ablation_config.update(correction_mode="once", correction_passes=1)
            elif name == "no_uncertainty":
                ablation_config["use_uncertainty"] = False
            elif name == "no_gate":
                ablation_config["use_gate"] = False
            else:
                raise ValueError("Unknown ICR-MMSE-Net ablation: "+name)
            ablation_run = seed_run/"ablations"/name
            ablation_run.mkdir(parents=True, exist_ok=True)
            if name == "once_budget_matched":
                budget = match_once_budget(data_directory, checkpoint, backend, **config.get("budget_match", {}))
                write_json(ablation_run/"budget_match.json", budget)
                ablation_config["equalizer_repeats"] = budget["equalizer_repeats"]
            ablation_checkpoint = train(data_directory, ablation_run, backend, model_config=ablation_config,
                lambda_h=selected["lambda_h"], seed=seed, resume=resume, warmup_checkpoint=fixed, **training)
            ablation_report = evaluate(Path(data_directory)/"test.h5", ablation_checkpoint,
                ablation_run/"evaluation", backend, batch_size=evaluation.get("batch_size", 2),
                frame_limit=training.get("frame_limit"), bootstrap_samples=evaluation.get("bootstrap_samples", 2000))
            final_latency_ratio = (ablation_report["performance"]["seconds_per_block"] /
                                   report["performance"]["seconds_per_block"])
            ablation_reports[name] = dict(ber=ablation_report["curves"]["ICR-MMSE-Net"]["ber"],
                selection=ablation_report["selection"], budget_match=budget,
                test_latency_ratio=final_latency_ratio,
                test_latency_use="descriptive only; budget was selected on validation inputs",
                warmup_checkpoint=str(fixed.resolve()))
        reports.append(dict(seed=seed, checkpoint=str(Path(checkpoint).resolve()),
                            ber=report["curves"]["ICR-MMSE-Net"]["ber"], selection=report["selection"],
                            ablations=ablation_reports))
        if backend.device == "cuda":
            import torch
            torch.cuda.empty_cache()
        write_json(run/"seed_repeats.json", dict(complete=len(reports) == len(seeds), selected=selected,
            seeds=list(seeds), repeats=reports,
            statistics="Random seeds control optimization; physical records remain the channel sampling unit"))
    return reports
