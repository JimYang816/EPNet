"""Verify that both channels use—and actually learn—the full unfolding detector."""
from __future__ import annotations

from pathlib import Path

import torch

from src.algorithms.icr_mmse.checkpoint import load_model
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.experiment import file_sha256


ARCHITECTURE_KEYS = ("layers", "bandwidth", "precision", "noise_mode", "detector_mode",
                     "window_size", "center_size", "hidden_size", "use_uncertainty", "use_gate",
                     "correction_passes", "equalizer_repeats", "step_mode", "initial_soft_damping")


def detector_audit(payload):
    controls = payload["identity"]["controls"]
    trained_torch = payload["identity"].get("environment", {}).get("torch")
    if trained_torch is not None and trained_torch != str(torch.__version__):
        raise ValueError("Parameter initialization audit requires the training Torch version")
    # Initialize in the same Torch environment as training. Restoring RNG keeps
    # this inspection from changing any caller's subsequent sampling.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(payload["seed"])
        initial = ICRMMSEnet(**payload["model_config"]).state_dict()
    rows = []
    for layer in range(payload["model_config"]["layers"]):
        prefix = f"equalizers.{layer}."
        changed = [name[len(prefix):] for name, value in payload["model"].items()
                   if name.startswith(prefix) and not torch.equal(value.cpu(), initial[name])]
        physics = [name for name in changed if not name.startswith(("hidden.", "classifier."))]
        head = payload["model"][prefix+"classifier.weight"]
        rows.append(dict(layer=layer+1, changed_fields=changed, changed_physical_fields=physics,
                         hidden_weights_changed="hidden.weight" in changed,
                         classifier_weights_changed="classifier.weight" in changed,
                         classifier_nonzero=bool(torch.count_nonzero(head))))
    policy = controls.get("detector_parameters", "all")
    passed = (policy == "all" and not controls.get("stage_c_final_equalizer_only", False)
              and all(row["changed_physical_fields"] and row["hidden_weights_changed"]
                      and row["classifier_weights_changed"] and row["classifier_nonzero"] for row in rows))
    return dict(channel=payload["channel"], torch=str(torch.__version__), detector_parameters=policy,
                neural_learning_rate_scale=controls.get("neural_learning_rate_scale", 1.),
                layers=rows, passed=bool(passed))


def audit_study(checkpoints):
    reports, signatures, payloads = {}, {}, {}
    for name, path in checkpoints.items():
        _, payload = load_model(path)
        if payload.get("smoke") or not payload.get("complete"):
            raise ValueError(f"{name}: requires a completed formal checkpoint")
        row = detector_audit(payload)
        if not row["passed"]:
            raise ValueError(f"{name}: every layer must learn physical parameters, hidden weights and classifier")
        signatures[name] = {key: payload["model_config"][key] for key in ARCHITECTURE_KEYS}
        reports[name] = dict(checkpoint=str(Path(path)), sha256=file_sha256(path), **row)
        payloads[name] = payload
    if any(value != next(iter(signatures.values())) for value in signatures.values()):
        raise ValueError("The two channels must use the same detector architecture")
    shared = {}
    for channel in ("ncs1", "nof1"):
        fixed, joint = (payloads[prefix+channel] for prefix in ("fixed_", "joint_"))
        a, b = (payload["identity"]["data"].get("warmup_checkpoint") for payload in (fixed, joint))
        if a is None or a != b:
            raise ValueError(f"{channel}: fixed/joint detectors must share the same all-parameter warmup")
        shared[channel] = a
    return dict(complete=True, architecture=next(iter(signatures.values())),
                shared_warmup=shared, cases=reports,
                note="Channel-specific trained values, loading initialization and optimizer rates are allowed; "
                     "architecture and trainable detector components are identical.")
