"""Separate channel models, each trained jointly on every SNR in its data."""
from __future__ import annotations

from contextlib import ExitStack
import logging
from pathlib import Path
import time

import numpy as np
import torch

from src.algorithms.epnet.checkpoint import SCHEMA, PAPER_DOI, atomic_save, cpu_state, load_bank
from src.algorithms.epnet.model import EPNet, layerwise_loss
from src.common.experiment import write_json
from src.data.compact import CompactDataset
from src.data.epnet import likelihood_batches
from src.evaluation.fingerprints import DataIdentity, source_fingerprint, numerical_environment
from src.evaluation.result_cache import FileLock
from src.training.meta_optimizer import LSTMOptimizer, train_meta_optimizer

LOG = logging.getLogger(__name__)
CODE = ("src.algorithms.epnet.model", "src.algorithms.epnet.checkpoint", "src.training.epnet",
        "src.training.meta_optimizer", "src.data.epnet", "src.evaluation.inputs",
        "src.algorithms.likelihood", "src.data.compact", "src.common.numerics",
        "data_generation.src.dataset.compact_math", "data_generation.src.transmitter.ofdm:qpsk",
        "data_generation.src.receiver.dpss:DPSSModel")


def _epoch(model, dataset, si, backend, batch_size, frame_limit, *, backward=False, progress=None):
    count = dataset.frame_count if frame_limit is None else min(frame_limit, dataset.frame_count)
    selected = list(range(len(dataset.snr_db))) if si is None else [si]
    total, errors, bits, processed = 0., 0, 0, 0
    per_snr = []
    with torch.set_grad_enabled(backward):
        for index in selected:
            snr_loss, snr_errors, snr_bits = 0., 0, 0
            for gram, matched, symbols in likelihood_batches(dataset, index, backend, batch_size=batch_size,
                                                            frame_limit=frame_limit):
                result = model(gram, matched)
                loss = layerwise_loss(result, symbols)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite EPNet training/validation loss")
                weight = len(symbols)/(2*count)
                if backward:
                    # Average over ALL examples/SNRs before ONE parameter update.
                    (loss*weight/len(selected)).backward()
                snr_loss += float(loss.detach())*weight
                snr_errors += int(((result.symbols.real < 0) != (symbols.real < 0)).sum())
                snr_errors += int(((result.symbols.imag < 0) != (symbols.imag < 0)).sum())
                snr_bits += 2*symbols.numel()
                processed += len(symbols)
                if progress is not None:
                    progress(processed, 2*count*len(selected), float(dataset.snr_db[index]))
            total += snr_loss/len(selected)
            errors += snr_errors
            bits += snr_bits
            per_snr.append(dict(snr_db=float(dataset.snr_db[index]), loss=snr_loss,
                                ber=snr_errors/snr_bits, errors=snr_errors, bits=snr_bits))
    return dict(loss=total, ber=errors/bits, errors=errors, bits=bits, per_snr=per_snr)


def _state_to(state, device):
    if state is None:
        return None
    return tuple(tuple(t.to(device) for t in pair) for pair in state)


def _train(data_directory, run, backend, *, model_config=None, epochs=100, batch_size=4,
          optimizer="lstm", learning_rate=.1, meta_config=None, seed=20261027,
          frame_limit=None, resume=False, adaptation_steps=20):
    """Exact dataset-average eq. (20) gradient, streamed in memory-bounded batches.

    Each gradient includes every SNR. One shared model per channel. Training
    NEVER opens test.h5. The LSTM is first trained on new Gaussian quadratic
    tasks, then frozen, following III-C. Adam is an explicitly labelled control.
    """
    if (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1
            or optimizer not in ("lstm", "adam") or not np.isfinite(learning_rate) or learning_rate <= 0
            or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1
            or not isinstance(adaptation_steps, int) or isinstance(adaptation_steps, bool) or adaptation_steps < 1
            or (frame_limit is not None and (not isinstance(frame_limit, int) or isinstance(frame_limit, bool) or frame_limit < 1))):
        raise ValueError("Invalid EPNet training controls")
    configuration = EPNet(**(model_config or {})).config
    if configuration["layers"] < 2:
        raise ValueError("Training needs at least two layers; the first cavity has no damping dependence")
    meta_controls = dict(epochs=100, tasks=20, steps=20, learning_rate=1e-3)
    meta_controls.update(meta_config or {})
    controls = dict(epochs=epochs, batch_size=batch_size, optimizer=optimizer, learning_rate=learning_rate,
                    meta=meta_controls, seed=seed, frame_limit=frame_limit, model=configuration,
                    adaptation_steps=adaptation_steps if optimizer == "lstm" else 1,
                    training_scope="joint_snr")
    directory, run = Path(data_directory), Path(run)
    run.mkdir(parents=True, exist_ok=True)
    last_path, final_path = run/"training_state.pt", run/"epnet.pt"
    if not resume and any(p.exists() for p in (last_path, final_path, run/"lstm_optimizer.pt")):
        raise FileExistsError("Training output exists; use resume or a new run directory")
    torch.manual_seed(seed)
    with ExitStack() as stack:
        train_ds = stack.enter_context(CompactDataset(directory/"train.h5"))
        validation_ds = stack.enter_context(CompactDataset(directory/"validation.h5"))
        if (train_ds.split != "train" or validation_ds.split != "validation"
                or train_ds.channel != validation_ds.channel
                or train_ds._h.attrs["model_identity"] != validation_ds._h.attrs["model_identity"]
                or not np.array_equal(train_ds.snr_db, validation_ds.snr_db)
                or np.intersect1d(train_ds._h["windows/record"][:], validation_ds._h["windows/record"][:]).size):
            raise ValueError("Training requires distinct train/validation records with one shared data model")
        guard = DataIdentity({name: directory/name for name in ("train.h5", "validation.h5", "model.h5")})
        identity = dict(data=guard.files, code=source_fingerprint(CODE), controls=controls,
                        environment=dict(numerical_environment(backend), torch=str(torch.__version__)))
        if resume and final_path.exists():
            bank = load_bank(final_path, backend.device)
            if bank.metadata.get("training_identity") != identity:
                raise ValueError("Completed training identity changed")
            bank.validate_dataset(train_ds)
            return final_path
        state = dict(identity=identity, current=None)
        if resume and last_path.exists():
            state = torch.load(last_path, map_location="cpu", weights_only=True)
            if state.get("identity") != identity:
                raise ValueError("Resume identity changed: data, code or training controls")
        meta = None
        if optimizer == "lstm":
            meta_path = run/"lstm_optimizer.pt"
            if resume and meta_path.exists():
                saved = torch.load(meta_path, map_location="cpu", weights_only=True)
                if saved.get("identity") != identity:
                    raise ValueError("Meta optimizer identity changed")
                meta = LSTMOptimizer().to(backend.device)
                meta.load_state_dict(saved["state"])
                meta.eval().requires_grad_(False)
            else:
                meta, history = train_meta_optimizer(layers=configuration["layers"], seed=seed,
                    device=backend.device, callback=lambda e, loss: LOG.info("Meta epoch=%d loss=%.6g", e, loss),
                    **meta_controls)
                guard.assert_unchanged()
                atomic_save(meta_path, dict(schema="epnet-lstm-optimizer-v1", identity=identity,
                                           state=cpu_state(meta), history=history))
        model = EPNet(**configuration).to(backend.device)
        updater = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer == "adam" else None
        lstm_state = None
        current = state["current"]

        def progress(phase, epoch, update=0):
            last_write = [0.]
            def report(done, total, snr):
                if time.monotonic()-last_write[0] >= 10 or done == total:
                    write_json(run/"progress.json", dict(channel=train_ds.channel, training_scope="joint_snr",
                        phase=phase, epoch=epoch, epochs=epochs, update=update,
                        updates_per_epoch=controls["adaptation_steps"], processed_blocks=done,
                        total_blocks=total, current_snr_db=snr, snr_db=train_ds.snr_db.tolist()))
                    last_write[0] = time.monotonic()
            return report

        if current is None:
            baseline = _epoch(model, validation_ds, None, backend, batch_size, frame_limit,
                              progress=progress("initial_validation", 0))
            current = dict(epoch=0, best_state=cpu_state(model), best_loss=baseline["loss"], best_epoch=0,
                           history=[dict(epoch=0, validation=baseline, damping=model.damping.detach().cpu().tolist())])
        else:
            model.load_state_dict(current["model"])
            if updater is not None:
                updater.load_state_dict(current["updater"])
            lstm_state = _state_to(current.get("lstm_state"), backend.device)
        for epoch in range(current["epoch"], epochs):
            for update in range(controls["adaptation_steps"]):
                model.zero_grad(set_to_none=True)
                training = _epoch(model, train_ds, None, backend, batch_size, frame_limit,
                                  backward=True, progress=progress("train", epoch+1, update+1))
                if model.beta.grad is None or not torch.isfinite(model.beta.grad).all():
                    raise FloatingPointError("Invalid EPNet damping gradient")
                if updater is not None:
                    updater.step()
                else:
                    with torch.no_grad():
                        delta, lstm_state = meta(model.beta.grad, lstm_state)
                        model.beta.add_(delta)
            validation = _epoch(model, validation_ds, None, backend, batch_size, frame_limit,
                                progress=progress("validation", epoch+1))
            if validation["loss"] < current["best_loss"]:
                current.update(best_loss=validation["loss"], best_epoch=epoch+1, best_state=cpu_state(model))
            current["history"].append(dict(epoch=epoch+1, train=training, validation=validation,
                damping=model.damping.detach().cpu().tolist()))
            current.update(epoch=epoch+1, model=cpu_state(model),
                updater=updater.state_dict() if updater is not None else None,
                lstm_state=_state_to(lstm_state, "cpu"))
            state["current"] = current
            guard.assert_unchanged()
            atomic_save(last_path, state)
            LOG.info("%s joint-SNR epoch=%d train=%.6g val=%.6g BER=%.6g damping=%s",
                     train_ds.channel, epoch+1, training["loss"], validation["loss"], validation["ber"],
                     model.damping.detach().cpu().tolist())
        payload = dict(schema=SCHEMA, complete=True, paper_doi=PAPER_DOI, channel=train_ds.channel,
            data_schema=str(train_ds._h.attrs["schema"]), model_identity=str(train_ds._h.attrs["model_identity"]),
            data_count=len(train_ds._indices), snr_db=train_ds.snr_db.tolist(), model_config=configuration,
            training_scope="joint_snr", models=[current["best_state"]], training_identity=identity,
            histories=[dict(snr_db=train_ds.snr_db.tolist(), best_epoch=current["best_epoch"],
                best_validation_loss=current["best_loss"], epochs=current["history"])],
            supervision="train symbols only; validation loss selects checkpoints; no test access",
            adaptation="existing compact-v2 CSI; full ICI; estimated pilot cancellation; thermal noise whitening",
            optimizer=optimizer, precision="float64/complex128",
            smoke=bool(train_ds._config.get("smoke")) or frame_limit is not None)
        guard.assert_unchanged()
        atomic_save(final_path, payload)
        write_json(run/"training.json", {k: v for k, v in payload.items() if k != "models"})
        last_path.unlink(missing_ok=True)
        return final_path


def train(data_directory, run, backend, **options):
    """Train with exclusive ownership of the run directory and epoch checkpoints."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run/".training.lock"):
        return _train(data_directory, run, backend, **options)
