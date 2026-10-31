"""Supplementary validation-only selection and once-correction budget diagnostics."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch



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
    from src.algorithms.icr_mmse.inference import prepare_frozen_inference
    from src.data.compact import CompactDataset
    from src.data.icr_mmse import input_batches
    if not repeats or any(not isinstance(r, int) or r < 1 for r in repeats) or timing_repeats < 1:
        raise ValueError("Budget matching requires positive equalizer and timing repeats")
    if not 0 < tolerance < 1:
        raise ValueError("Budget tolerance must be between zero and one")
    model, payload = load_model(checkpoint, backend.device)
    prepare_frozen_inference(model)
    precision = model.config["precision"]
    with CompactDataset(Path(data_directory)/"validation.h5", input_view="compensated") as dataset:
        # load_model already verifies checkpoint metadata; validation metadata is
        # checked by the same function used by formal evaluation.
        validate_dataset(payload, dataset)
        first, _ = next(input_batches(dataset, device=backend.device, batch_size=batch_size,
                                     frame_limit=1, precision=precision))
        def measure():
            with torch.inference_mode():
                model(**first, return_diagnostics=False)
                elapsed, count = [], 0
                for _ in range(timing_repeats):
                    backend.synchronize()
                    started = time.perf_counter()
                    count = 0
                    for inputs, keys in input_batches(dataset, device=backend.device, batch_size=batch_size,
                                                     frame_limit=frame_limit, precision=precision):
                        model(**inputs, return_diagnostics=False)
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
