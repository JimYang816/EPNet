"""One OAMP-Net2 per channel; all fixed training SNRs and D1/D2 share weights."""
from __future__ import annotations

from contextlib import ExitStack
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch

from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.oamp_net2.checkpoint import SCHEMA, PAPER_DOI, INFERENCE_CODE
from src.algorithms.oamp_net2.model import OAMPNet2, FEATURES
from src.common.experiment import write_json
from src.common.torch_checkpoint import atomic_save, cpu_state, random_state, restore_random_state
from src.data.compact import CompactDataset
from src.data.oamp_net2 import PREPROCESSING
from src.data.oamp_net2_cache import build_features, CACHE_CODE
from src.evaluation.fingerprints import DataIdentity, numerical_environment, source_fingerprint
from src.evaluation.result_cache import FileLock
from src.training.oamp_net2_supervision import mse

LOG = logging.getLogger(__name__)
TRAINING_CODE = INFERENCE_CODE+CACHE_CODE+("src.training.oamp_net2", "src.common.torch_checkpoint",
    "src.common.experiment", "src.evaluation.fingerprints", "src.evaluation.result_cache")


def run_epoch(model, cache, device, batch_size, *, optimizer=None, generator=None, snr_db=None):
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(cache.count, generator=generator).numpy() if training else np.arange(cache.count)
    total = 0.
    errors = np.zeros(len(snr_db), dtype=np.int64) if snr_db is not None else None
    bits = np.zeros_like(errors) if errors is not None else None
    with torch.set_grad_enabled(training):
        for start in range(0, cache.count, batch_size):
            indices = order[start:start+batch_size]
            batch = cache.batch(indices, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            estimate = model(*(batch[name] for name in FEATURES))
            loss = mse(estimate, batch["symbols"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite OAMP-Net2 loss")
            if training:
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Missing/nonfinite OAMP-Net2 gradient")
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite OAMP-Net2 parameters")
            total += float(loss.detach())*len(indices)
            if errors is not None:
                different = decisions(estimate.detach().cpu().numpy()) != decisions(batch["symbols"].cpu().numpy())
                groups = indices//(cache.count//len(snr_db))
                np.add.at(errors, groups, different.sum(axis=(1, 2)))
                np.add.at(bits, groups, 2*estimate.shape[-1])
    metrics = None if errors is None else [dict(snr_db=float(snr), errors=int(e), total_bits=int(b), ber=float(e/b))
                                         for snr, e, b in zip(snr_db, errors, bits)]
    return total/cache.count, metrics


def _train(data_directory, run, backend, *, model_config=None, epochs=200, patience=20,
           learning_rate=.001, seed=20261030, batch_size=8, frame_batch_size=8,
           frame_limit=None, resume=False, stop_after_epochs=None, input_view="compensated"):
    integers = (epochs, patience, batch_size, frame_batch_size)
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in integers)
            or not np.isfinite(learning_rate) or learning_rate <= 0 or input_view != "compensated"
            or (frame_limit is not None and (backend.device != "cpu" or not isinstance(frame_limit, int)
                or isinstance(frame_limit, bool) or frame_limit < 1))
            or (stop_after_epochs is not None and (not isinstance(stop_after_epochs, int) or stop_after_epochs < 1))):
        raise ValueError("Invalid OAMP-Net2 training controls; use compensated CSI and CPU-only smoke limits")
    run, directory = Path(run), Path(data_directory)
    latest, best = run/"latest.pt", run/"best.pt"
    if not resume and (latest.exists() or best.exists()):
        raise FileExistsError("OAMP-Net2 output exists; use --resume or another output")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if backend.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    model = OAMPNet2(**(model_config or {})).to(backend.device)
    controls = dict(epochs=epochs, patience=patience, learning_rate=learning_rate, seed=seed,
        batch_size=batch_size, frame_limit=frame_limit, model=model.config, input_view=input_view,
        optimizer="Adam", loss="mean_example_squared_vector_norm_final_layer",
        training_scope="one_channel_all_snr_D1_D2")
    with ExitStack() as stack:
        training = stack.enter_context(CompactDataset(directory/"train.h5"))
        validation = stack.enter_context(CompactDataset(directory/"validation.h5"))
        if (training.split != "train" or validation.split != "validation"
                or training.channel != validation.channel
                or training._h.attrs["model_identity"] != validation._h.attrs["model_identity"]
                or not np.array_equal(training.snr_db, validation.snr_db)
                or not np.array_equal(training._indices, validation._indices)
                or np.intersect1d(training._h["windows/record"][:], validation._h["windows/record"][:]).size):
            raise ValueError("Need disjoint train/validation records and matching channel/model/SNR grid")
        # Never open or hash test.h5 during training or model selection.
        guard = DataIdentity({name: directory/name for name in ("train.h5", "validation.h5", "model.h5")})
        identity = dict(data=guard.files, code=source_fingerprint(TRAINING_CODE), controls=controls,
            environment=dict(numerical_environment(backend), torch=str(torch.__version__)))
        state = None
        if resume:
            state = torch.load(latest, map_location="cpu", weights_only=True)
            if state.get("identity") != identity:
                raise ValueError("Resume rejected: data, model, code, controls or environment identity changed")
            if state.get("finished"):
                if not best.is_file():
                    raise ValueError("Finished OAMP-Net2 run has no best checkpoint")
                return best
        train_cache = build_features(training, run/"features", backend, guard,
                                    frame_batch_size=frame_batch_size, frame_limit=frame_limit)
        stack.callback(train_cache.close)
        val_cache = build_features(validation, run/"features", backend, guard,
                                  frame_batch_size=frame_batch_size, frame_limit=frame_limit)
        stack.callback(val_cache.close)
        write_json(run/"feature_sources.json", dict(train=train_cache.manifest, validation=val_cache.manifest))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        history, start, best_loss, bad_epochs, best_epoch = [], 0, float("inf"), 0, 0
        if state is not None:
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            history, start = state["history"], state["epoch"]
            best_loss, bad_epochs, best_epoch = state["best_loss"], state["bad_epochs"], state["best_epoch"]
            restore_random_state(state["random_state"], generator)
        metadata = dict(schema=SCHEMA, paper_doi=PAPER_DOI, preprocessing=PREPROCESSING,
            input_view=input_view, inference_code=source_fingerprint(INFERENCE_CODE), identity=identity,
            model_config=model.config, channel=training.channel, model_identity=str(training._h.attrs["model_identity"]),
            data_schema=str(training._h.attrs["schema"]), data_indices=training._indices.tolist(),
            snr_db=training.snr_db.tolist(), training_scope=controls["training_scope"], batch_size=batch_size,
            smoke=bool(training._config.get("smoke")) or frame_limit is not None,
            training_blocks=train_cache.count, validation_blocks=val_cache.count,
            parameters=sum(p.numel() for p in model.parameters()))
        for epoch in range(start+1, epochs+1):
            started = time.perf_counter()
            train_loss, _ = run_epoch(model, train_cache, backend.device, batch_size, optimizer=optimizer, generator=generator)
            val_loss, metrics = run_epoch(model, val_cache, backend.device, batch_size, snr_db=validation.snr_db)
            improved = val_loss < best_loss
            if improved:
                best_loss, bad_epochs, best_epoch = val_loss, 0, epoch
            else:
                bad_epochs += 1
            history.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                validation_by_snr=metrics, seconds=time.perf_counter()-started))
            finished = epoch == epochs or bad_epochs >= patience
            guard.assert_unchanged()
            payload = dict(metadata, model=cpu_state(model), optimizer=optimizer.state_dict(),
                random_state=random_state(generator), epoch=epoch, best_epoch=best_epoch,
                best_loss=best_loss, bad_epochs=bad_epochs, history=history, finished=finished)
            if improved:
                atomic_save(best, payload)
            atomic_save(latest, payload)
            write_json(run/"history.json", history)
            write_json(run/"training_summary.json", dict(channel=training.channel, complete=finished,
                epochs=epoch, best_epoch=best_epoch, best_validation_loss=best_loss,
                stop_reason="patience" if bad_epochs >= patience else "max_epochs" if finished else "running",
                training_blocks=train_cache.count, validation_blocks=val_cache.count, identity=identity))
            LOG.info("%s OAMP-Net2 epoch %d/%d train=%.7g validation=%.7g best=%d patience=%d/%d",
                     training.channel, epoch, epochs, train_loss, val_loss, best_epoch, bad_epochs, patience)
            if finished or (stop_after_epochs is not None and epoch-start >= stop_after_epochs):
                break
        return best


def train(data_directory, run, backend, **kwargs):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".training.lock"):
        return _train(data_directory, run, backend, **kwargs)
