"""Three-stage training on the accepted compensated CSI; test data never participate in fitting or selection."""
from __future__ import annotations

from contextlib import ExitStack
import logging
import math
from pathlib import Path

import numpy as np
import torch

from src.algorithms.icr_mmse.checkpoint import (SCHEMA, INPUT_VIEW, INFERENCE_CODE,
    atomic_save, cpu_state, load_model, validate_dataset)
from src.algorithms.icr_mmse.model import FEEDBACK_LAYOUT, ICRMMSEnet, csi_nmse, layerwise_loss
from src.algorithms.icr_mmse.dpss import base_error, reference_energy
from src.common.experiment import write_json
from src.data.compact import CompactDataset
from src.data.icr_mmse import input_batches, supervision
from src.data.icr_training import ReferenceCache, training_batches
from src.evaluation.fingerprints import DataIdentity, source_fingerprint, numerical_environment
from src.evaluation.result_cache import FileLock

LOG = logging.getLogger(__name__)
TRAINING_CODE = INFERENCE_CODE + ("src.training.icr_mmse", "src.evaluation.fingerprints",
                                 "src.common.experiment", "src.data.icr_training", "src.data.icr_dense_training")


def configure_stage(model, stage):
    """A: detector only; B: shared corrector/damping only; C: all parameters."""
    if stage not in ("A", "B", "C"):
        raise ValueError("Unknown ICR-MMSE-Net training stage")
    for name, parameter in model.named_parameters():
        correction = name.startswith("corrector.") or name == "eta_logits"
        parameter.requires_grad_(stage == "C" or (correction if stage == "B" else not correction))


def run_epoch(model, dataset, device, *, stage, batch_size=2, frame_limit=None,
              lambda_h=.1, optimizer=None, operator_backend="dense", prefetch_batches=0,
              compact_reference=None, reference_cache=None, snr_indices=None, frame_indices=None):
    """Every batch recomputes the entire current-H trajectory, including stage B."""
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    counts = torch.zeros(len(dataset.snr_db), 2, dtype=torch.long, device=device)
    count = bit_count = 0
    nmse_count = 0
    if compact_reference is None:
        compact_reference = operator_backend == "dpss"
    with torch.set_grad_enabled(training), training_batches(dataset, device=device, batch_size=batch_size,
            frame_limit=frame_limit, precision=model.config["precision"], operator_backend=operator_backend,
            include_reference=stage != "A" or not training, compact_reference=compact_reference,
            bandwidth=model.config["bandwidth"], cache=reference_cache, prefetch_batches=prefetch_batches,
            snr_indices=snr_indices, frame_indices=frame_indices) as batches:
        for inputs, targets, keys in batches:
            result = model(**inputs, correction_enabled=stage != "A", detach_feedback=stage == "B",
                           return_diagnostics=False)
            reference = targets.get("reference_channel")
            loss = layerwise_loss(result, targets["classes"], reference, inputs["data_mask"],
                                  lambda_h=lambda_h if stage != "A" else 0., csi_only=stage == "B")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ICR-MMSE-Net loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not parameters:
                    raise FloatingPointError("Missing ICR-MMSE-Net gradient")
                # One finite norm check covers every gradient; preserve fail-fast
                # clipping and avoid one CUDA synchronization per parameter.
                try:
                    torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
                except RuntimeError as error:
                    if "non-finite" in str(error):
                        raise FloatingPointError("Nonfinite ICR-MMSE-Net gradient") from error
                    raise
                optimizer.step()
            batch_count = len(keys)
            totals[0] += loss.detach().double() * batch_count
            count += batch_count
            prediction = result.symbols[:, -1, inputs["data_mask"]].detach()
            labels = targets["symbols"]
            per_example = ((prediction.real < 0) != (labels.real < 0)).long().sum(-1)
            per_example += ((prediction.imag < 0) != (labels.imag < 0)).long().sum(-1)
            bit_count += 2 * labels.numel()
            indices = targets["snr_indices"]
            counts[:, 0].scatter_add_(0, indices, per_example)
            counts[:, 1].scatter_add_(0, indices, torch.full_like(per_example, 2 * labels.shape[-1]))
            if reference is not None:
                with torch.no_grad():
                    # Metrics never need a second autograd graph.
                    values = csi_nmse(result, reference)
                    denominator = reference_energy(reference).clamp_min(1e-30)
                    raw = base_error(inputs["initial_channel"], reference) / denominator
                    totals[1] += values[:, -1].sum().double()
                    totals[2] += raw.sum().double()
                nmse_count += batch_count
            del result, loss, targets, inputs, reference
    if count == 0 or bit_count == 0:
        raise ValueError("No ICR-MMSE-Net examples")
    total_loss, nmse_total, raw_total = totals.cpu().tolist()
    by_snr = counts.cpu().tolist()
    errors = sum(row[0] for row in by_snr)
    return dict(loss=total_loss/count, ber=errors/bit_count, errors=errors, bits=bit_count,
                initial_nmse=raw_total/nmse_count if nmse_count else None,
                corrected_nmse=nmse_total/nmse_count if nmse_count else None,
                per_snr=[dict(snr_db=float(snr), ber=row[0]/row[1], errors=row[0], bits=row[1])
                         for snr, row in zip(dataset.snr_db, by_snr) if row[1]])


def _rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _selection_key(stage, metrics):
    if stage == "B":
        return (metrics["corrected_nmse"], metrics["ber"])
    if stage == "C":
        # Validation BER is primary among candidates with actual CSI improvement.
        improved = metrics["corrected_nmse"] < metrics["initial_nmse"]
        return (0 if improved else 1, metrics["ber"], metrics["corrected_nmse"])
    return (metrics["ber"], metrics["loss"])


def _train(data_directory, run, backend, *, model_config=None, stage_epochs=(10, 10, 10),
           lambda_h=.1, seed=20261028, batch_size=2, learning_rate=1e-3, frame_limit=None,
           resume=False, stop_after_epochs=None, warmup_checkpoint=None, operator_backend="dense",
           prefetch_batches=2, reference_cache_mib=512, compact_reference=None):
    if (len(stage_epochs) != 3 or any(isinstance(e, bool) or not isinstance(e, int) or e < 1 for e in stage_epochs)
            or not math.isfinite(lambda_h) or lambda_h <= 0
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or (frame_limit is not None and (isinstance(frame_limit, bool) or not isinstance(frame_limit, int) or frame_limit < 1))
            or (stop_after_epochs is not None and (isinstance(stop_after_epochs, bool)
                or not isinstance(stop_after_epochs, int) or stop_after_epochs < 1))):
        raise ValueError("Invalid ICR-MMSE-Net training controls")
    if operator_backend not in ("dense", "dpss", "dense_device"):
        raise ValueError("Unknown ICR operator backend")
    if (isinstance(prefetch_batches, bool) or not isinstance(prefetch_batches, int) or prefetch_batches < 0
            or isinstance(reference_cache_mib, bool) or not isinstance(reference_cache_mib, int) or reference_cache_mib < 0):
        raise ValueError("Invalid ICR data pipeline controls")
    if compact_reference is None:
        compact_reference = operator_backend == "dpss"
    directory, run = Path(data_directory), Path(run)
    final = run / "icr_mmse.pt"
    latest = run / "latest.pt"
    configuration = ICRMMSEnet(**(model_config or {})).config
    controls = dict(model=configuration, stage_epochs=list(stage_epochs), lambda_h=lambda_h, seed=seed,
                    batch_size=batch_size, learning_rate=learning_rate, frame_limit=frame_limit,
                    warmup_checkpoint=str(Path(warmup_checkpoint).resolve()) if warmup_checkpoint else None,
                    input_view=INPUT_VIEW, operator_backend=operator_backend,
                    prefetch_batches=prefetch_batches, reference_cache_mib=reference_cache_mib,
                    compact_reference=compact_reference)
    if not resume and any(path.exists() for path in (latest, final, run / "warmup.pt")):
        raise FileExistsError("Training output exists; use resume or a new run directory")
    torch.manual_seed(seed)
    with ExitStack() as stack:
        train_ds = stack.enter_context(CompactDataset(directory / "train.h5", input_view=INPUT_VIEW))
        validation_ds = stack.enter_context(CompactDataset(directory / "validation.h5", input_view=INPUT_VIEW))
        pipeline = dict(operator_backend=operator_backend, prefetch_batches=prefetch_batches,
                        compact_reference=compact_reference,
                        reference_cache=ReferenceCache(reference_cache_mib * 2**20))
        if (train_ds.split != "train" or validation_ds.split != "validation"
                or train_ds.channel != validation_ds.channel
                or train_ds._h.attrs["model_identity"] != validation_ds._h.attrs["model_identity"]
                or not np.array_equal(train_ds.snr_db, validation_ds.snr_db)
                or np.intersect1d(train_ds._h["windows/record"][:], validation_ds._h["windows/record"][:]).size):
            raise ValueError("Training requires disjoint train/validation physical records and one data model")
        paths = {name: directory / name for name in ("train.h5", "validation.h5", "model.h5")}
        if warmup_checkpoint:
            paths["warmup_checkpoint"] = Path(warmup_checkpoint)
        guard = DataIdentity(paths)
        identity = dict(data=guard.files, code=source_fingerprint(TRAINING_CODE), controls=controls,
                        environment=dict(numerical_environment(backend), torch=str(torch.__version__),
                                         precision=configuration["precision"] + "/" +
                                         ("float32" if configuration["precision"] == "complex64" else "float64")))
        model = ICRMMSEnet(**configuration).to(backend.device)
        history = []
        stage_index = 0
        epoch = 0
        selected = None
        selected_key = None
        updater_state = None
        if resume and latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=True)
            if state["identity"] != identity:
                raise ValueError("Resume identity changed: CSI view, data, code or training controls")
            if state.get("finished"):
                _, completed = load_model(final, backend.device)
                if completed.get("identity") != identity:
                    raise ValueError("Completed training identity changed")
                validate_dataset(completed, train_ds)
                return final
            model.load_state_dict(state["model"])
            stage_index, epoch = state["stage_index"], state["epoch"]
            history, selected, selected_key = state["history"], state["selected"], state["selected_key"]
            updater_state = state["updater"]
            _restore_rng(state["rng"])
        elif resume and final.exists():
            _, payload = load_model(final, backend.device)
            if payload.get("identity") != identity:
                raise ValueError("Completed training identity changed")
            return final
        elif warmup_checkpoint:
            _, warmup = load_model(warmup_checkpoint, backend.device)
            validate_dataset(warmup, train_ds)
            if warmup.get("stage") != "A" or warmup.get("inference_correction_enabled", True):
                raise ValueError("Warmup checkpoint must be stage A of the same channel/data model")
            if warmup["model_config"].get("precision") != configuration["precision"]:
                raise ValueError("Warmup checkpoint precision differs")
            warmup_data = warmup.get("identity", {}).get("data", {})
            if any(warmup_data.get(name) != guard.files[name]
                   for name in ("train.h5", "validation.h5", "model.h5")):
                raise ValueError("Warmup checkpoint was trained on different records/data")
            # Reuse the detector alone: B changes corrector shape, T changes layer count.
            detector = {k: v for k, v in warmup["model"].items() if k.startswith("equalizers.")}
            own_detector = {k: v for k, v in model.state_dict().items() if k.startswith("equalizers.")}
            if detector.keys() != own_detector.keys() or any(detector[k].shape != own_detector[k].shape for k in detector):
                raise ValueError("Warmup detector architecture differs")
            model.load_state_dict(dict(model.state_dict(), **detector))
            stage_index = 1

        def payload(stage, correction_enabled, finished=False):
            return dict(schema=SCHEMA, algorithm="ICR-MMSE-Net", input_view=INPUT_VIEW,
                        feedback_layout=FEEDBACK_LAYOUT,
                        channel=train_ds.channel, data_schema=str(train_ds._h.attrs["schema"]),
                        model_identity=str(train_ds._h.attrs["model_identity"]),
                        data_indices=train_ds._indices.tolist(), snr_db=train_ds.snr_db.tolist(),
                        training_scope="one_channel_all_snr_D1_D2", model_config=configuration,
                        inference_code=source_fingerprint(INFERENCE_CODE), model=cpu_state(model),
                        inference_correction_enabled=correction_enabled, stage=stage, complete=finished,
                        identity=identity, history=history, seed=seed,
                        smoke=frame_limit is not None or bool(train_ds._config.get("smoke")))

        if warmup_checkpoint and stage_index == 1 and not (run / "warmup.pt").exists():
            atomic_save(run / "warmup.pt", payload("A", False))
        completed_this_call = 0
        for index in range(stage_index, 3):
            stage = "ABC"[index]
            configure_stage(model, stage)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
            if index == stage_index and updater_state is not None:
                optimizer.load_state_dict(updater_state)
            if selected is None:
                metrics = run_epoch(model, validation_ds, backend.device, stage=stage, batch_size=batch_size,
                                    frame_limit=frame_limit, lambda_h=lambda_h, **pipeline)
                selected, selected_key = cpu_state(model), _selection_key(stage, metrics)
            for current in range(epoch if index == stage_index else 0, stage_epochs[index]):
                training = run_epoch(model, train_ds, backend.device, stage=stage, batch_size=batch_size,
                                     frame_limit=frame_limit, lambda_h=lambda_h,
                                     optimizer=None if stage == "B" and configuration["correction_mode"] == "fixed" else optimizer,
                                     **pipeline)
                metrics = run_epoch(model, validation_ds, backend.device, stage=stage, batch_size=batch_size,
                                    frame_limit=frame_limit, lambda_h=lambda_h, **pipeline)
                key = _selection_key(stage, metrics)
                if key < selected_key:
                    selected, selected_key = cpu_state(model), key
                history.append(dict(stage=stage, epoch=current+1, train=training, validation=metrics))
                guard.assert_unchanged()
                stage_finished = current+1 == stage_epochs[index]
                if stage_finished:
                    model.load_state_dict(selected)
                    name = ("warmup.pt", "correction_pretrained.pt", "icr_mmse.pt")[index]
                    atomic_save(run / name, payload(stage, stage != "A", finished=index == 2))
                atomic_save(latest, dict(identity=identity, stage_index=index+1 if stage_finished else index,
                    epoch=0 if stage_finished else current+1, model=cpu_state(model), history=history,
                    selected=None if stage_finished else selected, selected_key=None if stage_finished else selected_key,
                    updater=None if stage_finished else optimizer.state_dict(), rng=_rng_state(),
                    finished=stage_finished and index == 2))
                write_json(run / "progress.json", dict(channel=train_ds.channel, stage=stage,
                    epoch=current+1, epochs=stage_epochs[index], validation=metrics, input_view=INPUT_VIEW))
                LOG.info("%s stage=%s epoch=%d BER=%.6g NMSE=%s", train_ds.channel, stage, current+1,
                         metrics["ber"], metrics["corrected_nmse"])
                completed_this_call += 1
                if (stop_after_epochs is not None and completed_this_call >= stop_after_epochs
                        and not (stage_finished and index == 2)):
                    return final if final.exists() else latest
            epoch = 0
            selected = selected_key = updater_state = None
        guard.assert_unchanged()
        write_json(run / "training.json", dict(algorithm="ICR-MMSE-Net", input_view=INPUT_VIEW,
            channel=train_ds.channel, controls=controls, history=history, complete=True,
            selection="validation BER with CSI improvement; stage B uses validation full-matrix NMSE",
            test_access=False))
        return final


def train(data_directory, run, backend, **options):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / ".training.lock"):
        return _train(data_directory, run, backend, **options)
