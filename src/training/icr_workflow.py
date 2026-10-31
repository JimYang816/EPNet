"""Resolve the shared physical/neural design and train its two CSI branches."""
from copy import deepcopy
import json
from pathlib import Path

DEFAULT_CONFIG = Path("configs/training/icr_mmse.json")
STAGES = ("warmup", "fixed", "joint")


def resolve_config(path=DEFAULT_CONFIG, *, channel, stage="joint", output=None):
    if stage not in STAGES:
        raise ValueError(f"Unknown training branch: {stage}")
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if channel not in config["datasets"]:
        raise ValueError(f"Channel {channel} is absent from the training config")
    config = deepcopy(config)
    overrides = config.pop("channel_overrides", {}).get(channel, {})
    config["model"].update(overrides.get("model", {}))
    controls = config["training"]
    controls.update(overrides.get("training", {}))
    if config.get("input_view") != "compensated":
        raise ValueError("Current ICR training requires accepted compensated CSI")
    if config["model"].get("detector_mode") != "budgeted" or controls.get("detector_parameters") != "all":
        raise ValueError("Current ICR requires residual unfolding with all physical and neural parameters")
    root = Path(output or config["output"])
    epochs = controls["stage_epochs"]
    if stage in ("warmup", "fixed"):
        controls.update(overrides.get("warmup_training", {}))
        controls["stage_epochs"] = [epochs[0], 0, 0 if stage == "warmup" else epochs[2]]
        config["model"]["correction_mode"] = "fixed"
    else:
        config["model"]["correction_mode"] = "iterative"
    if stage == "warmup":
        controls.pop("warmup_checkpoint", None)
    else:
        controls["warmup_checkpoint"] = str(root/"warmup"/channel.lower()/"warmup.pt")
    config["datasets"] = {channel: config["datasets"][channel]}
    config["output"] = str(root/stage)
    return config


def run_channel(path, channel, backend, *, output=None, resume=False, frame_limit=None):
    from src.training.icr_mmse import train

    result = {}
    for stage in STAGES:
        config = resolve_config(path, channel=channel, stage=stage, output=output)
        controls = config["training"]
        if frame_limit is not None:
            controls["frame_limit"] = frame_limit
        result[stage] = train(config["datasets"][channel],
            Path(config["output"])/channel.lower(), backend,
            model_config=config["model"], resume=resume, **controls)
    return result
