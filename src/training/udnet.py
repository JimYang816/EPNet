"""Joint-SNR Adam training, independent channel models, exact epoch recovery."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import logging
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from src.algorithms.udnet.checkpoint import (SCHEMA, PAPER_DOI, INFERENCE_CODE, atomic_save,
    cpu_state, random_state, restore_random_state)
from src.algorithms.udnet.model import UDNet
from src.common.experiment import write_json
from src.data.compact import CompactDataset
from src.data.udnet import FEATURES, PREPROCESSING
from src.data.udnet_cache import CACHE_CODE, build_features
from src.evaluation.fingerprints import DataIdentity, source_fingerprint, numerical_environment
from src.evaluation.result_cache import FileLock
from src.training.udnet_supervision import multilayer_cross_entropy
from src.training.udnet_diagnostics import ValidationMetrics, training_health

LOG = logging.getLogger(__name__)
TRAINING_CODE = INFERENCE_CODE+CACHE_CODE+("src.training.udnet", "src.common.experiment",
                                         "src.evaluation.fingerprints", "src.evaluation.result_cache",
                                         "src.training.udnet_diagnostics")


def choose_batch_size(model, cache, device, learning_rate):
    """Measure all four candidates on real training data, including Adam state."""
    if device == "cpu":
        return min(64, cache.count), dict(device="cpu", selected=min(64, cache.count), probes=[])
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    budget = int(.8*free)
    trials = []
    saved_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    for size in (2000, 1024, 512, 256):
        trial = dict(batch_size=size, accepted=False)
        probe = optimizer = batch = probabilities = loss = None
        try:
            torch.cuda.empty_cache()
            probe = copy.deepcopy(model).train()
            optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
            batch = cache.batch(np.arange(size) % cache.count, device)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                probabilities = probe(*(batch[name] for name in FEATURES))
                loss = multilayer_cross_entropy(probabilities, batch["labels"], batch["data_mask"])
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            probabilities = probe(*(batch[name] for name in FEATURES))
            loss = multilayer_cross_entropy(probabilities, batch["labels"], batch["data_mask"])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            trial.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=peak,
                         step_seconds=time.perf_counter()-started, accepted=peak <= budget)
        except torch.cuda.OutOfMemoryError:
            trial["error"] = "CUDA out of memory"
        finally:
            del probe, optimizer, batch, probabilities, loss
            torch.cuda.empty_cache()
        trials.append(trial)
        LOG.info("UDNet real-batch probe: %s", trial)
    torch.set_rng_state(saved_rng[0])
    torch.cuda.set_rng_state_all(saved_rng[1])
    accepted = [trial["batch_size"] for trial in trials if trial["accepted"]]
    if not accepted:
        raise RuntimeError("No requested batch size fits within 80% of available GPU memory")
    selected = max(accepted)
    return selected, dict(device=torch.cuda.get_device_name(), free_bytes=free, total_bytes=total,
                          memory_budget_bytes=budget, selected=selected, probes=trials)


def run_epoch(model, cache, device, batch_size, *, optimizer=None, generator=None, metrics=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(cache.count, generator=generator).numpy() if training else np.arange(cache.count)
    total, labels = 0., 0
    with torch.set_grad_enabled(training):
        for start in range(0, cache.count, batch_size):
            batch = cache.batch(order[start:start+batch_size], device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            probabilities = model(*(batch[name] for name in FEATURES))
            if metrics is not None:
                metrics.observe(order[start:start+batch_size], probabilities, batch)
            loss = multilayer_cross_entropy(probabilities, batch["labels"], batch["data_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite UDNet loss")
            if training:
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Missing/nonfinite UDNet gradient")
                optimizer.step()
            count = int(batch["data_mask"].sum())
            total += float(loss.detach())*count
            labels += count
    return total/labels


def _train(data_directory, run, backend, *, model_config=None, epochs=200, patience=20,
           learning_rate=.001, seed=20261027, batch_size=None, frame_batch_size=8,
           frame_limit=None, resume=False, stop_after_epochs=None):
    if (not isinstance(epochs, int) or epochs < 1 or not isinstance(patience, int) or patience < 1
            or learning_rate <= 0 or not np.isfinite(learning_rate)
            or (batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1))):
        raise ValueError("Invalid UDNet training controls")
    run, directory = Path(run), Path(data_directory)
    run.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run/"latest.pt", run/"best.pt"
    if not resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError("UDNet output exists; use --resume or another output")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if backend.device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(seed)
    model = UDNet(**(model_config or {})).to(device=backend.device, dtype=torch.float32)
    controls = dict(epochs=epochs, patience=patience, learning_rate=learning_rate, seed=seed,
                    requested_batch_size=batch_size, frame_limit=frame_limit, model=model.config,
                    optimizer="Adam", training_scope="one_channel_all_snr_D1_D2")
    with ExitStack() as stack:
        training = stack.enter_context(CompactDataset(directory/"train.h5"))
        validation = stack.enter_context(CompactDataset(directory/"validation.h5"))
        if (training.split != "train" or validation.split != "validation"
                or training.channel != validation.channel
                or training._h.attrs["model_identity"] != validation._h.attrs["model_identity"]
                or not np.array_equal(training.snr_db, validation.snr_db)
                or np.intersect1d(training._h["windows/record"][:], validation._h["windows/record"][:]).size):
            raise ValueError("Need disjoint train/validation records and matching channel/model/SNR grid")
        # Deliberately excludes test.h5: no test statistic participates in training/selection.
        guard = DataIdentity({name: directory/name for name in ("train.h5", "validation.h5", "model.h5")})
        identity = dict(data=guard.files, code=source_fingerprint(TRAINING_CODE), controls=controls,
                        environment=dict(numerical_environment(backend), torch=str(torch.__version__)))
        state = None
        if resume:
            if not last_path.exists():
                raise FileNotFoundError("No latest UDNet recovery checkpoint")
            state = torch.load(last_path, map_location="cpu", weights_only=True)
            if state.get("identity") != identity:
                raise ValueError("Resume rejected: data, model, code, controls or environment identity changed")
            if state.get("finished"):
                if not best_path.exists():
                    raise ValueError("Finished run has no best checkpoint")
                return best_path
        train_cache = build_features(training, run/"features", backend, guard,
                                     frame_batch_size=frame_batch_size, frame_limit=frame_limit)
        validation_cache = build_features(validation, run/"features", backend, guard,
                                          frame_batch_size=frame_batch_size, frame_limit=frame_limit)
        write_json(run/"feature_sources.json", dict(train=train_cache.manifest, validation=validation_cache.manifest))
        if state is not None:
            selected, resource = state["batch_size"], state["resource_probe"]
        elif batch_size is None:
            selected, resource = choose_batch_size(model, train_cache, backend.device, learning_rate)
        else:
            selected, resource = batch_size, dict(device=backend.device, selected=batch_size, explicit_override=True)
        write_json(run/"resource_probe.json", resource)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history, start_epoch, best_loss, bad_epochs, best_epoch = [], 0, float("inf"), 0, 0
        if state is not None:
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            history, start_epoch = state["history"], state["epoch"]
            best_loss, bad_epochs, best_epoch = state["best_loss"], state["bad_epochs"], state["best_epoch"]
            restore_random_state(state["random_state"], generator)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        metadata = dict(schema=SCHEMA, paper_doi=PAPER_DOI, preprocessing=PREPROCESSING,
                        inference_code=source_fingerprint(INFERENCE_CODE), identity=identity,
                        model_config=model.config, channel=training.channel,
                        model_identity=str(training._h.attrs["model_identity"]),
                        data_schema=str(training._h.attrs["schema"]), data_indices=training._indices.tolist(),
                        snr_db=training.snr_db.tolist(), training_scope="one_channel_all_snr_D1_D2",
                        commit=commit, batch_size=selected, resource_probe=resource,
                        training_windows=train_cache.count, validation_windows=validation_cache.count)
        completed_this_call = 0
        for epoch in range(start_epoch+1, epochs+1):
            started = time.perf_counter()
            train_loss = run_epoch(model, train_cache, backend.device, selected, optimizer=optimizer, generator=generator)
            validation_metrics = ValidationMetrics(validation.snr_db, validation_cache.count,
                                                   model.config["layers"], backend.device)
            validation_loss = run_epoch(model, validation_cache, backend.device, selected,
                                        metrics=validation_metrics)
            improved = validation_loss < best_loss
            if improved:
                best_loss, bad_epochs, best_epoch = validation_loss, 0, epoch
            else:
                bad_epochs += 1
            history.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss,
                                validation_metrics=validation_metrics.result(),
                                seconds=time.perf_counter()-started, best_epoch=best_epoch, bad_epochs=bad_epochs))
            finished = epoch == epochs or bad_epochs >= patience
            if epoch == 1 or epoch % 10 == 0 or finished:
                history[-1]["training_health"] = training_health(model, train_cache, backend.device)
                health = history[-1]["training_health"]
                LOG.info("%s numerical health: max saturated=%.4f zero-gradient layers=%s", training.channel,
                         max(row["saturated_fraction"] for row in health["layers"]), health["zero_gradient_layers"])
            history[-1]["seconds"] = time.perf_counter()-started
            guard.assert_unchanged()
            payload = dict(metadata, model=cpu_state(model), optimizer=optimizer.state_dict(),
                           random_state=random_state(generator), epoch=epoch, best_epoch=best_epoch,
                           best_loss=best_loss, bad_epochs=bad_epochs, history=history, finished=finished)
            if improved:
                atomic_save(best_path, payload)
            atomic_save(last_path, payload)
            write_json(run/"history.json", history)
            write_json(run/"training_summary.json", dict(channel=training.channel, complete=finished,
                epochs=epoch, best_epoch=best_epoch, best_validation_loss=best_loss, bad_epochs=bad_epochs,
                stop_reason="patience" if bad_epochs >= patience else "max_epochs" if finished else "running",
                batch_size=selected, training_windows=train_cache.count, validation_windows=validation_cache.count,
                identity=identity, resource_probe=resource, commit=commit))
            LOG.info("%s epoch %d/%d train=%.7f validation=%.7f best=%d patience=%d/%d %.1fs",
                     training.channel, epoch, epochs, train_loss, validation_loss, best_epoch, bad_epochs, patience,
                     history[-1]["seconds"])
            completed_this_call += 1
            if finished or (stop_after_epochs is not None and completed_this_call >= stop_after_epochs):
                break
        return best_path


def train(data_directory, run, backend, **kwargs):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".training.lock"):
        return _train(data_directory, run, backend, **kwargs)
