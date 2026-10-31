"""Validation-only audits and bounded, separately saved OAMP-Net2 fine-tuning.

Diagnostic checkpoints deliberately have no production checkpoint schema. The
original checkpoints and test results are never overwritten or used for selection.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import logging
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from data_generation.src.transmitter.ofdm import decisions
from src.algorithms.likelihood import make_whitener
from src.algorithms.mmse_pic.detector import mmse_pic
from src.algorithms.oamp_net2.checkpoint import load_model, validate_dataset
from src.algorithms.oamp_net2.model import FEATURES, OAMPNet2
from src.common.experiment import file_sha256, write_json
from src.common.torch_checkpoint import atomic_save, cpu_state
from src.data.compact import CompactDataset
from src.data.oamp_net2 import spectral_features
from src.data.oamp_net2_cache import build_features
from src.evaluation.fingerprints import DataIdentity
from src.evaluation.inputs import prepare_batch
from src.training.oamp_net2_supervision import mse

LOG = logging.getLogger(__name__)


def dense_reference(a, y, model):
    """Independent receive-space solves and enumerated QPSK posterior means."""
    n, m = a.shape[-1], a.shape[-2]
    x = torch.zeros((*y.shape[:-1], n), dtype=a.dtype, device=a.device)
    receive_identity = torch.eye(m, dtype=a.dtype, device=a.device)
    transmit_identity = torch.eye(n, dtype=a.dtype, device=a.device)
    constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j],
                                dtype=a.dtype, device=a.device)/np.sqrt(2)
    outputs = []
    for layer in range(model.config["layers"]):
        residual = y-(a@x[..., None])[..., 0]
        v = ((residual.abs().square().sum(-1)-m)/a.abs().square().sum((-1, -2))).clamp_min(
            model.config["variance_floor"])
        w = (torch.linalg.solve(v[..., None, None]*(a@a.mH)+receive_identity,
                                a)*v[..., None, None]).mH
        w = w*(n/(w@a).diagonal(dim1=-2, dim2=-1).real.sum(-1))[..., None, None]
        r = x+model.gamma[layer]*(w@residual[..., None])[..., 0]
        c = transmit_identity-model.theta[layer]*(w@a)
        tau = ((c.abs().square().sum((-1, -2))*v
                +model.theta[layer].square()*w.abs().square().sum((-1, -2)))/n).clamp_min(
                    model.config["variance_floor"])
        probability = torch.softmax(-(r[..., None]-constellation).abs().square()/tau[..., None, None], -1)
        x = model.phi[layer]*((probability*constellation).sum(-1)-model.xi[layer]*r)
        outputs.append(x)
    return torch.stack(outputs, dim=-2)


def gradient_check(model, features, target, *, directions=4, step=1e-5):
    """Compare autograd with central differences along parameter directions."""
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    model.zero_grad(set_to_none=True)
    loss = mse(model(*(features[k] for k in FEATURES)), target)
    gradients = torch.autograd.grad(loss, params)
    generator = torch.Generator(device=params[0].device).manual_seed(1949)
    results = []
    try:
        for _ in range(directions):
            delta = [torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator) for p in params]
            norm = torch.sqrt(sum(v.square().sum() for v in delta))
            delta = [v/norm for v in delta]
            analytic = sum((g*d).sum() for g, d in zip(gradients, delta)).item()
            values = []
            with torch.no_grad():
                for sign in (1, -1):
                    for p, value, direction in zip(params, original, delta):
                        p.copy_(value+sign*step*direction)
                    values.append(mse(model(*(features[k] for k in FEATURES)), target).item())
            numeric = (values[0]-values[1])/(2*step)
            results.append(dict(analytic=analytic, numeric=numeric,
                                scaled_error=abs(analytic-numeric)/max(1., abs(analytic), abs(numeric))))
    finally:
        with torch.no_grad():
            for p, value in zip(params, original):
                p.copy_(value)
    return dict(directions=results, gradients={k: g.detach().cpu().tolist()
                for (k, _), g in zip(model.named_parameters(), gradients)},
                maximum_scaled_error=max(r["scaled_error"] for r in results))


def _tensor_features(batch):
    return [batch[k] for k in FEATURES]


@torch.no_grad()
def score(model, cache, snr_db, *, batch_size=64, layers=False, baselines=False):
    """Score every validation example; cache rows are ordered SNR/D/frame."""
    model.eval()
    per_snr = cache.count//len(snr_db)
    nlayer = model.config["layers"] if layers else 1
    errors = np.zeros((len(snr_db), nlayer), dtype=np.int64)
    squared = np.zeros_like(errors, dtype=np.float64)
    baseline_errors = {name: np.zeros(len(snr_db), np.int64) for name in ("MMSE-PIC", "Full-matrix LMMSE")}
    baseline_squared = {name: np.zeros(len(snr_db)) for name in baseline_errors}
    device = next(model.parameters()).device
    for start in range(0, cache.count, batch_size):
        indices = np.arange(start, min(start+batch_size, cache.count))
        b = cache.batch(indices, device)
        output = model(*_tensor_features(b), return_layers=layers)
        output = output if layers else output[:, None]
        truth = b["symbols"]
        different = decisions(output.cpu().numpy()) != decisions(truth.cpu().numpy())[:, None]
        difference = (output-truth[:, None]).abs().square().sum(-1).cpu().numpy()
        group = indices//per_snr
        np.add.at(errors, group, different.sum(axis=(-1, -2)))
        np.add.at(squared, group, difference)
        if baselines:
            q, eigenvalues, projected = b["eigenvectors"], b["eigenvalues"], b["projected_matched"]
            gram = (q*eigenvalues[:, None])@q.mH
            matched = (q@projected[..., None])[..., 0]
            identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=device)
            tensor_backend = SimpleNamespace(torch=torch, array=lambda a: torch.as_tensor(a, device=device))
            estimates = {"MMSE-PIC": mmse_pic(gram, matched, tensor_backend),
                "Full-matrix LMMSE": torch.linalg.solve(gram+identity, matched[..., None])[..., 0]}
            for name, estimate in estimates.items():
                different = decisions(estimate.cpu().numpy()) != decisions(truth.cpu().numpy())
                np.add.at(baseline_errors[name], group, different.sum(axis=(-1, -2)))
                np.add.at(baseline_squared[name], group,
                          (estimate-truth).abs().square().sum(-1).cpu().numpy())
    n = cache.arrays["symbols"].shape[-1]
    result = dict(validation_loss=float(squared[:, -1].sum()/cache.count),
                  by_snr=[dict(snr_db=float(s), vector_mse=(sq/per_snr).tolist(),
                               errors=e.tolist(), total_bits=per_snr*2*n,
                               ber=(e/(per_snr*2*n)).tolist())
                          for s, sq, e in zip(snr_db, squared, errors)])
    if baselines:
        result["baselines"] = {name: dict(validation_loss=float(baseline_squared[name].sum()/cache.count),
            by_snr=[dict(snr_db=float(s), vector_mse=float(sq/per_snr), errors=int(e),
                         total_bits=per_snr*2*n, ber=float(e/(per_snr*2*n)))
                    for s, sq, e in zip(snr_db, baseline_squared[name], baseline_errors[name])])
            for name in baseline_errors}
    return result


def _fine_tune(payload, training, validation, snr_db, run, *, epochs, rate, batch_size, device, single_snr=None):
    model = OAMPNet2(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
    for group in optimizer.param_groups:
        group["lr"] = rate
    generator = torch.Generator().manual_seed(20261030)
    base_indices = np.arange(training.count)
    target_si = None
    if single_snr is not None:
        target_si = list(snr_db).index(single_snr)
        count = training.count//len(snr_db)
        base_indices = base_indices[target_si*count:(target_si+1)*count]
    initial = score(model, validation, snr_db, batch_size=batch_size)
    objective = lambda result: result["validation_loss"] if target_si is None else result["by_snr"][target_si]["vector_mse"][-1]
    best_loss, best_state, best_epoch, history = objective(initial), cpu_state(model), 0, []
    run.mkdir(parents=True, exist_ok=False)
    controls = dict(epochs=epochs, learning_rate=rate, batch_size=batch_size, single_snr=single_snr,
                    training_blocks=len(base_indices), validation_blocks=validation.count,
                    optimizer="Adam_parent_moments_retained", seed=20261030)
    write_json(run/"controls.json", controls)
    for epoch in range(1, epochs+1):
        started = time.perf_counter()
        model.train()
        order = base_indices[torch.randperm(len(base_indices), generator=generator).numpy()]
        accumulated = 0.
        for start in range(0, len(order), batch_size):
            indices = order[start:start+batch_size]
            b = training.batch(indices, next(model.parameters()).device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(model(*_tensor_features(b)), b["symbols"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Invalid diagnostic gradient")
            optimizer.step()
            accumulated += loss.item()*len(indices)
        row = dict(additional_epoch=epoch, training_loss=accumulated/len(order), seconds=time.perf_counter()-started)
        # The original global validation objective selects joint-SNR candidates;
        # single-SNR experiments select using that SNR's VALIDATION MSE only.
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            result = score(model, validation, snr_db, batch_size=batch_size)
            row.update(validation_loss=result["validation_loss"],
                       by_snr=result["by_snr"], selection_objective=objective(result))
            if objective(result) < best_loss:
                best_loss, best_state, best_epoch = objective(result), cpu_state(model), epoch
            LOG.info("%s extra epoch %d/%d validation=%.8g selection=%.8g best=%d",
                     run.name, epoch, epochs, result["validation_loss"], objective(result), best_epoch)
        history.append(row)
        write_json(run/"history.json", history)
    model.load_state_dict(best_state)
    result = score(model, validation, snr_db, batch_size=batch_size, layers=True)
    atomic_save(run/"best.pt", dict(diagnostic_only=True, model=best_state,
        model_config=model.config, parent_epoch=payload["epoch"], additional_epoch=best_epoch, controls=controls))
    write_json(run/"result.json", dict(controls=controls, best_additional_epoch=best_epoch,
               selection_objective=best_loss, initial=initial, final=result,
               parameters={k: v.tolist() for k, v in best_state.items()}))
    return dict(controls=controls, best_additional_epoch=best_epoch, final=result)


def diagnose(directory, checkpoint, run, backend, *, joint_epochs=50, snr_epochs=100, batch_size=64):
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (joint_epochs, snr_epochs, batch_size)):
        raise ValueError("Positive epoch counts and batch size required")
    directory, checkpoint, run = Path(directory), Path(checkpoint), Path(run)
    if (run/"diagnostics.json").exists():
        raise FileExistsError("Choose a new diagnostic output directory")
    run.mkdir(parents=True, exist_ok=True)
    model, payload = load_model(checkpoint, backend.device)
    model.requires_grad_(True)
    with ExitStack() as stack:
        training = stack.enter_context(CompactDataset(directory/"train.h5"))
        validation = stack.enter_context(CompactDataset(directory/"validation.h5"))
        validate_dataset(payload, training)
        validate_dataset(payload, validation)
        if np.intersect1d(training._h["windows/record"][:], validation._h["windows/record"][:]).size:
            raise ValueError("Training/validation physical records overlap")
        guard = DataIdentity({name: directory/name for name in ("train.h5", "validation.h5", "model.h5")})
        if payload["identity"]["data"] != guard.files:
            raise ValueError("Diagnostic data changed since parent training")
        report = dict(diagnostic_only=True, channel=training.channel, input_view="compensated",
            parent_checkpoint=str(checkpoint), parent_sha256=file_sha256(checkpoint), parent_epoch=payload["epoch"],
            data_identity=guard.files, test_data_opened=False, trials={})
        white = make_whitener(validation._shared["unit_rf_noise_covariance"][:], backend)
        audit_frames = list(range(min(2, validation.frame_count)))
        sample = prepare_batch(validation, len(validation.snr_db)-1, audit_frames, 0, backend)
        f = {k: torch.as_tensor(v, device=backend.device) for k, v in spectral_features(sample, white, backend).items()}
        target = torch.as_tensor(np.stack([validation.labels(i)["symbols"][0] for i in audit_frames]), device=backend.device)
        report["gradient_check"] = gradient_check(model, f, target)
        if report["gradient_check"]["maximum_scaled_error"] > 1e-3:
            raise AssertionError("Actual-data gradient check failed")
        with torch.no_grad():
            white_tensor = torch.as_tensor(white, device=backend.device, dtype=torch.complex128)
            h = white_tensor@torch.as_tensor(sample.channel[..., sample.data_indices], device=backend.device,
                                            dtype=torch.complex128)
            y = (white_tensor@torch.as_tensor(sample.cancelled_observations, device=backend.device,
                                              dtype=torch.complex128)[..., None])[..., 0]
            scale = torch.sqrt(torch.as_tensor(sample.variances, device=backend.device))
            reference = dense_reference(h/scale[:, None, None], y/scale[:, None], model)
            actual = model(*(f[k] for k in FEATURES), return_layers=True)
            report["dense_reference"] = dict(max_absolute=(actual-reference).abs().max().item(),
                relative_l2=((actual-reference).norm()/reference.norm()).item())
            if report["dense_reference"]["relative_l2"] > 1e-7:
                raise AssertionError("Actual-data dense reference mismatch")
        write_json(run/"diagnostics.json", report)
        val_cache = build_features(validation, run/"features", backend, guard, frame_batch_size=8)
        stack.callback(val_cache.close)
        report["original_validation"] = score(model, val_cache, validation.snr_db, layers=True,
            baselines=True, batch_size=batch_size)
        plain = OAMPNet2(**payload["model_config"]).to(backend.device)
        report["untrained_oamp_validation"] = score(plain, val_cache, validation.snr_db, layers=True,
                                                    batch_size=batch_size)
        write_json(run/"diagnostics.json", report)
        LOG.info("%s full validation audit complete; preparing full training features", training.channel)
        train_cache = build_features(training, run/"features", backend, guard, frame_batch_size=8)
        stack.callback(train_cache.close)
        last_snr = float(training.snr_db[-1])
        for name, rate, epochs, snr in [("joint_lr1e-3", 1e-3, joint_epochs, None),
                                        ("joint_lr1e-4", 1e-4, joint_epochs, None),
                                        (f"snr{last_snr:g}_lr1e-4", 1e-4, snr_epochs, last_snr)]:
            report["trials"][name] = _fine_tune(payload, train_cache, val_cache, training.snr_db,
                run/name, epochs=epochs, rate=rate, batch_size=batch_size, single_snr=snr, device=backend.device)
            guard.assert_unchanged()
            write_json(run/"diagnostics.json", report)
        report["complete"] = True
        write_json(run/"diagnostics.json", report)
    return report
