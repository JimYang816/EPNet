"""Three-stage training on the accepted compensated CSI; test data never participate in fitting or selection."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
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
from src.data.icr_sampling import epoch_sampling_seed
from src.training.icr_acceptance import acceptance_controls, make_limits, assess
from src.evaluation.fingerprints import DataIdentity, source_fingerprint, numerical_environment
from src.evaluation.result_cache import FileLock

LOG = logging.getLogger(__name__)
TRAINING_CODE = INFERENCE_CODE + ("src.training.icr_mmse", "src.evaluation.fingerprints",
                                 "src.common.experiment", "src.data.icr_training", "src.data.icr_dense_training",
                                 "src.data.icr_sampling", "src.training.icr_acceptance")


def configure_stage(model, stage, *, final_equalizer_only=False, detector_parameters="all"):
    """Optionally restrict C to the detector after the final CSI update."""
    if stage not in ("A", "B", "C"):
        raise ValueError("Unknown ICR-MMSE-Net training stage")
    if final_equalizer_only and stage != "C":
        raise ValueError("Final-equalizer-only training requires stage C")
    if detector_parameters not in ("all", "scalars"):
        raise ValueError("Unknown detector parameter selection")
    final_prefix = f"equalizers.{len(model.equalizers)-1}."
    for name, parameter in model.named_parameters():
        correction = name.startswith("corrector.") or name == "eta_logits"
        parameter.requires_grad_(name.startswith(final_prefix) if final_equalizer_only
            else stage == "C" or (correction if stage == "B" else not correction))
        if correction and model.config["correction_mode"] == "fixed":
            parameter.requires_grad_(False)
        if (detector_parameters == "scalars" and name.startswith("equalizers.")
                and (".hidden." in name or ".classifier." in name)):
            parameter.requires_grad_(False)


def run_epoch(model, dataset, device, *, stage, batch_size=2, frame_limit=None,
              lambda_h=.1, optimizer=None, operator_backend="dense", prefetch_batches=0,
              compact_reference=None, reference_cache=None, snr_indices=None, frame_indices=None,
              sampling_seed=None, final_equalizer_only=False, loss_options=None):
    """Every batch recomputes the entire current-H trajectory, including stage B."""
    if final_equalizer_only and stage != "C":
        raise ValueError("Final-equalizer-only loss requires stage C")
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    counts = torch.zeros(len(dataset.snr_db), 2, dtype=torch.long, device=device)
    snr_nmse = torch.zeros(len(dataset.snr_db), 2, dtype=torch.float64, device=device)
    snr_blocks = torch.zeros(len(dataset.snr_db), dtype=torch.long, device=device)
    count = bit_count = 0
    nmse_count = 0
    if compact_reference is None:
        compact_reference = operator_backend == "dpss"
    with torch.set_grad_enabled(training), training_batches(dataset, device=device, batch_size=batch_size,
            frame_limit=frame_limit, precision=model.config["precision"], operator_backend=operator_backend,
            include_reference=stage != "A" or not training, compact_reference=compact_reference,
            bandwidth=model.config["bandwidth"], cache=reference_cache, prefetch_batches=prefetch_batches,
            snr_indices=snr_indices, frame_indices=frame_indices, sampling_seed=sampling_seed) as batches:
        for inputs, targets, keys in batches:
            result = model(**inputs, correction_enabled=stage != "A", detach_feedback=stage == "B",
                           return_diagnostics=False)
            reference = targets.get("reference_channel")
            # Earlier outputs are frozen in this mode. Avoid diluting the only
            # trainable detector's CE by the number of unrolled layers.
            loss_output = (replace(result, probabilities=result.probabilities[:, -1:],
                                   channels=result.channels[-1:],
                                   log_probabilities=result.log_probabilities[:, -1:]) if final_equalizer_only else result)
            loss = layerwise_loss(loss_output, targets["classes"], reference, inputs["data_mask"],
                                  lambda_h=lambda_h if stage != "A" else 0., csi_only=stage == "B",
                                  **(loss_options or {}))
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
            snr_blocks.scatter_add_(0, indices, torch.ones_like(indices))
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
                    snr_nmse[:, 0].scatter_add_(0, indices, raw.double())
                    snr_nmse[:, 1].scatter_add_(0, indices, values[:, -1].double())
                nmse_count += batch_count
            del result, loss_output, loss, targets, inputs, reference
    if count == 0 or bit_count == 0:
        raise ValueError("No ICR-MMSE-Net examples")
    total_loss, nmse_total, raw_total = totals.cpu().tolist()
    by_snr = counts.cpu().tolist()
    nmse_by_snr, blocks_by_snr = snr_nmse.cpu().tolist(), snr_blocks.cpu().tolist()
    errors = sum(row[0] for row in by_snr)
    return dict(loss=total_loss/count, ber=errors/bit_count, errors=errors, bits=bit_count,
                initial_nmse=raw_total/nmse_count if nmse_count else None,
                corrected_nmse=nmse_total/nmse_count if nmse_count else None,
                per_snr=[dict(snr_db=float(snr), ber=row[0]/row[1], errors=row[0], bits=row[1], blocks=blocks,
                              initial_nmse=nmse[0]/blocks if nmse_count else None,
                              corrected_nmse=nmse[1]/blocks if nmse_count else None)
                         for snr, row, nmse, blocks in zip(dataset.snr_db, by_snr, nmse_by_snr, blocks_by_snr) if row[1]])


def _rng_state():
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _selection_key(stage, metrics, *, stage_c_nmse_limit=None, per_snr_limits=None):
    prefix = (0 if assess(metrics, per_snr_limits)["eligible"] else 1,) if per_snr_limits is not None else ()
    if stage == "B":
        return prefix + (metrics["corrected_nmse"], metrics["ber"])
    if stage == "C":
        # Validation BER is primary among candidates with actual CSI improvement.
        improved = metrics["corrected_nmse"] < metrics["initial_nmse"]
        if stage_c_nmse_limit is not None:
            improved = improved and metrics["corrected_nmse"] <= stage_c_nmse_limit
        return prefix + (0 if improved else 1, metrics["ber"], metrics["corrected_nmse"])
    return (metrics["ber"], metrics["loss"])


def _optimizer(model, stage, learning_rate, correction_learning_rate=None, neural_learning_rate_scale=1.):
    if not any(p.requires_grad for p in model.parameters()):
        return None
    if neural_learning_rate_scale != 1.:
        groups = {"physical": [], "neural": [], "correction": []}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            correction = name.startswith("corrector.") or name == "eta_logits"
            neural = name.startswith("equalizers.") and (".hidden." in name or ".classifier." in name)
            groups["correction" if correction else "neural" if neural else "physical"].append(parameter)
        rates = dict(physical=learning_rate, neural=learning_rate*neural_learning_rate_scale,
                     correction=correction_learning_rate if stage == "C" and correction_learning_rate is not None
                     else learning_rate)
        return torch.optim.Adam([dict(params=params, lr=rates[name], name=name)
                                 for name, params in groups.items() if params])
    if stage != "C" or correction_learning_rate is None:
        return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    groups = {"equalizer": [], "correction": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            correction = name.startswith("corrector.") or name == "eta_logits"
            groups["correction" if correction else "equalizer"].append(parameter)
    return torch.optim.Adam([
        dict(params=groups["equalizer"], lr=learning_rate, name="equalizer"),
        dict(params=groups["correction"], lr=correction_learning_rate, name="correction")])


def _train(data_directory, run, backend, *, model_config=None, stage_epochs=(10, 10, 10),
           lambda_h=.1, seed=20261028, batch_size=2, learning_rate=1e-3, frame_limit=None,
           resume=False, stop_after_epochs=None, warmup_checkpoint=None, operator_backend="dense",
           prefetch_batches=2, reference_cache_mib=512, compact_reference=None,
           stage_b_checkpoint=None, stage_c_correction_learning_rate=None,
           stage_c_nmse_max_relative_degradation=None, stage_c_learning_rate=None,
           sample_order="sequential", per_snr_acceptance=None,
           stage_c_final_equalizer_only=False, loss_options=None, detector_parameters="all",
           neural_learning_rate_scale=1.):
    if (len(stage_epochs) != 3 or any(isinstance(e, bool) or not isinstance(e, int) or e < 0 for e in stage_epochs)
            or not math.isfinite(lambda_h) or lambda_h <= 0
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or (frame_limit is not None and (isinstance(frame_limit, bool) or not isinstance(frame_limit, int) or frame_limit < 1))
            or (stop_after_epochs is not None and (isinstance(stop_after_epochs, bool)
                or not isinstance(stop_after_epochs, int) or stop_after_epochs < 1))):
        raise ValueError("Invalid ICR-MMSE-Net training controls")
    if operator_backend not in ("dense", "dpss", "dense_device"):
        raise ValueError("Unknown ICR operator backend")
    if (sample_order not in ("sequential", "snr_interleaved")
            or sample_order == "snr_interleaved" and operator_backend != "dense_device"):
        raise ValueError("SNR-interleaved training requires dense_device")
    if per_snr_acceptance is not None:
        per_snr_acceptance = acceptance_controls(per_snr_acceptance)
    if not isinstance(stage_c_final_equalizer_only, bool):
        raise ValueError("Stage-C final-equalizer-only control must be boolean")
    if detector_parameters not in ("all", "scalars"):
        raise ValueError("Unknown detector parameter selection")
    if (isinstance(neural_learning_rate_scale, bool) or not math.isfinite(neural_learning_rate_scale)
            or neural_learning_rate_scale <= 0):
        raise ValueError("Neural learning-rate scale must be positive and finite")
    if detector_parameters == "scalars" and neural_learning_rate_scale != 1.:
        raise ValueError("Neural learning-rate scale requires learning all detector parameters")
    if loss_options is not None and (not isinstance(loss_options, dict)
            or set(loss_options)-{"objective", "final_weight"}):
        raise ValueError("Unknown loss controls")
    if stage_c_final_equalizer_only and stage_c_correction_learning_rate is not None:
        raise ValueError("Frozen CSI correction cannot have a stage-C correction learning rate")
    if stage_c_learning_rate is not None and (
            isinstance(stage_c_learning_rate, bool) or not math.isfinite(stage_c_learning_rate)
            or stage_c_learning_rate <= 0):
        raise ValueError("Stage-C learning rate must be positive and finite")
    if (isinstance(prefetch_batches, bool) or not isinstance(prefetch_batches, int) or prefetch_batches < 0
            or isinstance(reference_cache_mib, bool) or not isinstance(reference_cache_mib, int) or reference_cache_mib < 0):
        raise ValueError("Invalid ICR data pipeline controls")
    if compact_reference is None:
        compact_reference = operator_backend == "dpss"
    if warmup_checkpoint and stage_b_checkpoint:
        raise ValueError("Choose either a stage-A warmup or a stage-B checkpoint")
    if stage_c_correction_learning_rate is not None and (
            isinstance(stage_c_correction_learning_rate, bool)
            or not math.isfinite(stage_c_correction_learning_rate) or stage_c_correction_learning_rate <= 0):
        raise ValueError("Stage-C correction learning rate must be positive and finite")
    if stage_c_nmse_max_relative_degradation is not None and (
            isinstance(stage_c_nmse_max_relative_degradation, bool)
            or not math.isfinite(stage_c_nmse_max_relative_degradation)
            or stage_c_nmse_max_relative_degradation < 0 or not stage_b_checkpoint):
        raise ValueError("Stage-C NMSE bound requires a stage-B checkpoint and nonnegative finite tolerance")
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
    if stage_b_checkpoint:
        controls["stage_b_checkpoint"] = str(Path(stage_b_checkpoint).resolve())
    if stage_c_correction_learning_rate is not None:
        controls["stage_c_correction_learning_rate"] = stage_c_correction_learning_rate
    if stage_c_nmse_max_relative_degradation is not None:
        controls["stage_c_nmse_max_relative_degradation"] = stage_c_nmse_max_relative_degradation
    if stage_c_learning_rate is not None:
        controls["stage_c_learning_rate"] = stage_c_learning_rate
    if stage_c_final_equalizer_only:
        controls["stage_c_final_equalizer_only"] = True
    if sample_order != "sequential":
        controls.update(sample_order=sample_order, sampler="snr-rounds-stage-epoch-v1")
    if per_snr_acceptance is not None:
        controls["per_snr_acceptance"] = per_snr_acceptance
    if loss_options:
        controls["loss_options"] = loss_options
    if detector_parameters != "all":
        controls["detector_parameters"] = detector_parameters
    if neural_learning_rate_scale != 1.:
        controls["neural_learning_rate_scale"] = neural_learning_rate_scale
    if not resume and any(path.exists() for path in (latest, final, run / "warmup.pt")):
        raise FileExistsError("Training output exists; use resume or a new run directory")
    torch.manual_seed(seed)
    with ExitStack() as stack:
        train_ds = stack.enter_context(CompactDataset(directory / "train.h5", input_view=INPUT_VIEW))
        validation_ds = stack.enter_context(CompactDataset(directory / "validation.h5", input_view=INPUT_VIEW))
        pipeline = dict(operator_backend=operator_backend, prefetch_batches=prefetch_batches,
                        compact_reference=compact_reference,
                        reference_cache=ReferenceCache(reference_cache_mib * 2**20), loss_options=loss_options)
        if (train_ds.split != "train" or validation_ds.split != "validation"
                or train_ds.channel != validation_ds.channel
                or train_ds._h.attrs["model_identity"] != validation_ds._h.attrs["model_identity"]
                or not np.array_equal(train_ds.snr_db, validation_ds.snr_db)
                or np.intersect1d(train_ds._h["windows/record"][:], validation_ds._h["windows/record"][:]).size):
            raise ValueError("Training requires disjoint train/validation physical records and one data model")
        paths = {name: directory / name for name in ("train.h5", "validation.h5", "model.h5")}
        if warmup_checkpoint:
            paths["warmup_checkpoint"] = Path(warmup_checkpoint)
        if stage_b_checkpoint:
            paths["stage_b_checkpoint"] = Path(stage_b_checkpoint)
        guard = DataIdentity(paths)
        identity = dict(data=guard.files, code=source_fingerprint(TRAINING_CODE), controls=controls,
                        environment=dict(numerical_environment(backend), torch=str(torch.__version__),
                                         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                         precision=configuration["precision"] + "/" +
                                         ("float32" if configuration["precision"] == "complex64" else "float64")))
        model = ICRMMSEnet(**configuration).to(backend.device)
        history = []
        stage_index = 0
        epoch = 0
        selected = None
        selected_key = None
        updater_state = None
        stage_c_selection = None
        per_snr_selection = {}
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
            stage_c_selection = state.get("stage_c_selection")
            per_snr_selection = state.get("per_snr_selection", {})
            if stage_c_nmse_max_relative_degradation is not None and stage_c_selection is None:
                raise ValueError("Resume is missing its fixed stage-B validation NMSE bound")
            _restore_rng(state["rng"])
        elif resume and final.exists():
            _, payload = load_model(final, backend.device)
            if payload.get("identity") != identity:
                raise ValueError("Completed training identity changed")
            return final
        elif stage_b_checkpoint:
            _, pretrained = load_model(stage_b_checkpoint, backend.device)
            validate_dataset(pretrained, train_ds)
            if pretrained.get("stage") != "B" or not pretrained.get("inference_correction_enabled"):
                raise ValueError("Stage-C initialization requires a stage-B correction checkpoint")
            if pretrained["model_config"] != configuration:
                raise ValueError("Stage-B checkpoint architecture/configuration differs")
            source_data = pretrained.get("identity", {}).get("data", {})
            if any(source_data.get(name) != guard.files[name]
                   for name in ("train.h5", "validation.h5", "model.h5")):
                raise ValueError("Stage-B checkpoint was trained on different records/data")
            model.load_state_dict(pretrained["model"], strict=True)
            stage_index = 2
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
                        stage_c_selection=stage_c_selection,
                        per_snr_selection=per_snr_selection,
                        smoke=frame_limit is not None or bool(train_ds._config.get("smoke")))

        if warmup_checkpoint and stage_index == 1 and not (run / "warmup.pt").exists():
            atomic_save(run / "warmup.pt", payload("A", False))
        completed_this_call = 0
        for index in range(stage_index, 3):
            stage = "ABC"[index]
            final_only = stage == "C" and stage_c_final_equalizer_only
            configure_stage(model, stage, final_equalizer_only=final_only,
                            detector_parameters=detector_parameters)
            stage_pipeline = dict(pipeline, final_equalizer_only=final_only)
            rate = stage_c_learning_rate if stage == "C" and stage_c_learning_rate is not None else learning_rate
            optimizer = _optimizer(model, stage, rate, stage_c_correction_learning_rate,
                                   neural_learning_rate_scale)
            if (stage_epochs[index] and optimizer is None
                    and not (stage == "B" and configuration["correction_mode"] == "fixed")):
                raise ValueError(f"Stage {stage} has an epoch budget but no trainable parameters")
            if index == stage_index and updater_state is not None:
                optimizer.load_state_dict(updater_state)
            if selected is None:
                metrics = run_epoch(model, validation_ds, backend.device, stage=stage, batch_size=batch_size,
                                    frame_limit=frame_limit, lambda_h=lambda_h, **stage_pipeline)
                if stage == "C" and stage_c_nmse_max_relative_degradation is not None:
                    reference_nmse = metrics["corrected_nmse"]
                    stage_c_selection = dict(scope="validation: full-matrix linear NMSE averaged over all blocks/SNRs",
                        reference="starting stage-B checkpoint", reference_validation=metrics,
                        max_relative_degradation=stage_c_nmse_max_relative_degradation,
                        nmse_limit=reference_nmse*(1+stage_c_nmse_max_relative_degradation),
                        selected_epoch=0, selected_validation=metrics)
                    LOG.info("%s stage=C equalizer_lr=%g correction_lr=%g validation_NMSE_limit=%.12g",
                        train_ds.channel, rate,
                        stage_c_correction_learning_rate or rate, stage_c_selection["nmse_limit"])
                if per_snr_acceptance is not None and stage in ("B", "C"):
                    per_snr_selection[stage] = dict(reference="starting stage-"+stage+" weights",
                        reference_validation=metrics, limits=make_limits(metrics, per_snr_acceptance),
                        selected_epoch=0, selected_validation=metrics)
                selected, selected_key = cpu_state(model), _selection_key(stage, metrics,
                    stage_c_nmse_limit=stage_c_selection["nmse_limit"] if stage_c_selection else None,
                    per_snr_limits=per_snr_selection[stage]["limits"] if stage in per_snr_selection else None)
            if per_snr_acceptance is not None and stage in ("B", "C") and stage not in per_snr_selection:
                raise ValueError("Resume is missing fixed per-SNR validation bounds")
            if stage_epochs[index] == 0:
                # A fixed detector has no A training; a fixed CSI has no B
                # training. Persist the boundary without fake optimizer steps.
                model.load_state_dict(selected)
                name = ("warmup.pt", "correction_pretrained.pt", "icr_mmse.pt")[index]
                atomic_save(run / name, payload(stage, stage != "A", finished=index == 2))
                atomic_save(latest, dict(identity=identity, stage_index=index+1, epoch=0,
                    model=cpu_state(model), history=history, selected=None, selected_key=None,
                    updater=None, rng=_rng_state(), stage_c_selection=stage_c_selection,
                    per_snr_selection=per_snr_selection, finished=index == 2))
                LOG.info("%s stage=%s skipped (zero training budget)", train_ds.channel, stage)
                epoch = 0
                selected = selected_key = updater_state = None
                continue
            for current in range(epoch if index == stage_index else 0, stage_epochs[index]):
                sampling_seed = epoch_sampling_seed(seed, stage, current+1) if sample_order == "snr_interleaved" else None
                training = run_epoch(model, train_ds, backend.device, stage=stage, batch_size=batch_size,
                                     frame_limit=frame_limit, lambda_h=lambda_h,
                                     optimizer=None if stage == "B" and configuration["correction_mode"] == "fixed" else optimizer,
                                     sampling_seed=sampling_seed, **stage_pipeline)
                metrics = run_epoch(model, validation_ds, backend.device, stage=stage, batch_size=batch_size,
                                    frame_limit=frame_limit, lambda_h=lambda_h, **stage_pipeline)
                key = _selection_key(stage, metrics,
                    stage_c_nmse_limit=stage_c_selection["nmse_limit"] if stage_c_selection else None,
                    per_snr_limits=per_snr_selection[stage]["limits"] if stage in per_snr_selection else None)
                if key < selected_key:
                    selected, selected_key = cpu_state(model), key
                    if stage_c_selection:
                        stage_c_selection.update(selected_epoch=current+1, selected_validation=metrics)
                    if stage in per_snr_selection:
                        per_snr_selection[stage].update(selected_epoch=current+1, selected_validation=metrics)
                entry = dict(stage=stage, epoch=current+1, train=training, validation=metrics)
                if sampling_seed is not None:
                    entry["sampling_seed"] = sampling_seed
                if stage in per_snr_selection:
                    entry["per_snr_acceptance"] = assess(metrics, per_snr_selection[stage]["limits"])
                if stage_c_selection:
                    entry["nmse_bound_eligible"] = (metrics["corrected_nmse"] < metrics["initial_nmse"]
                        and metrics["corrected_nmse"] <= stage_c_selection["nmse_limit"])
                history.append(entry)
                guard.assert_unchanged()
                stage_finished = current+1 == stage_epochs[index]
                if stage_finished:
                    model.load_state_dict(selected)
                    name = ("warmup.pt", "correction_pretrained.pt", "icr_mmse.pt")[index]
                    atomic_save(run / name, payload(stage, stage != "A", finished=index == 2))
                atomic_save(latest, dict(identity=identity, stage_index=index+1 if stage_finished else index,
                    epoch=0 if stage_finished else current+1, model=cpu_state(model), history=history,
                    selected=None if stage_finished else selected, selected_key=None if stage_finished else selected_key,
                    updater=None if stage_finished or optimizer is None else optimizer.state_dict(), rng=_rng_state(),
                    stage_c_selection=stage_c_selection,
                    per_snr_selection=per_snr_selection,
                    finished=stage_finished and index == 2))
                write_json(run / "progress.json", dict(channel=train_ds.channel, stage=stage,
                    epoch=current+1, epochs=stage_epochs[index], validation=metrics, input_view=INPUT_VIEW,
                    stage_c_selection=stage_c_selection, per_snr_selection=per_snr_selection,
                    per_snr_acceptance=entry.get("per_snr_acceptance"), sampling_seed=sampling_seed))
                if stage in per_snr_selection:
                    LOG.info("%s stage=%s per_SNR_eligible=%s violations=%s selected_epoch=%d", train_ds.channel,
                        stage, entry["per_snr_acceptance"]["eligible"], entry["per_snr_acceptance"]["violations"],
                        per_snr_selection[stage]["selected_epoch"])
                LOG.info("%s stage=%s epoch=%d BER=%.6g NMSE=%s", train_ds.channel, stage, current+1,
                         metrics["ber"], metrics["corrected_nmse"])
                if stage_c_selection:
                    LOG.info("%s stage=C NMSE_bound_eligible=%s selected_epoch=%d",
                        train_ds.channel, entry["nmse_bound_eligible"], stage_c_selection["selected_epoch"])
                completed_this_call += 1
                if (stop_after_epochs is not None and completed_this_call >= stop_after_epochs
                        and not (stage_finished and index == 2)):
                    return final if final.exists() else latest
            epoch = 0
            selected = selected_key = updater_state = None
        guard.assert_unchanged()
        write_json(run / "training.json", dict(algorithm="ICR-MMSE-Net", input_view=INPUT_VIEW,
            channel=train_ds.channel, controls=controls, history=history, complete=True,
            selection=("fixed per-SNR BER/NMSE bounds: B relative to A, C relative to B; starting weights remain candidates"
                       if per_snr_acceptance is not None else
                       "validation BER with CSI improvement and a fixed NMSE bound relative to stage B"
                       if stage_c_selection else
                       "validation BER with CSI improvement; stage B uses validation full-matrix NMSE"),
            stage_c_selection=stage_c_selection,
            per_snr_selection=per_snr_selection,
            test_access=False))
        return final


def train(data_directory, run, backend, **options):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / ".training.lock"):
        return _train(data_directory, run, backend, **options)
