"""Compensated-CSI, channel-bound OAMP-Net2 checkpoints."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.algorithms.oamp_net2.model import OAMPNet2
from src.data.oamp_net2 import PREPROCESSING, PREPROCESS_CODE
from src.evaluation.fingerprints import source_fingerprint

SCHEMA = "watermark-oamp-net2-joint-snr-v1"
PAPER_DOI = "10.1109/TSP.2020.2976585"
INFERENCE_CODE = PREPROCESS_CODE+("src.algorithms.oamp_net2.model",
    "src.algorithms.oamp_net2.inference", "src.algorithms.oamp_net2.checkpoint")


def load_model(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA or payload.get("paper_doi") != PAPER_DOI
            or payload.get("input_view") != "compensated"
            or payload.get("preprocessing") != PREPROCESSING
            or payload.get("inference_code") != source_fingerprint(INFERENCE_CODE)
            or payload.get("training_scope") != "one_channel_all_snr_D1_D2"):
        raise ValueError("OAMP-Net2 checkpoint model/preprocessing/compensated CSI identity mismatch")
    model = OAMPNet2(**payload["model_config"]).to(device=device)
    model.load_state_dict(payload["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite OAMP-Net2 checkpoint")
    model.eval().requires_grad_(False)
    return model, payload


def validate_dataset(payload, dataset):
    if dataset.input_view != "compensated" or payload.get("input_view") != dataset.input_view:
        raise ValueError("OAMP-Net2 requires its trained compensated CSI view")
    if (payload["channel"] != dataset.channel
            or payload["model_identity"] != dataset._h.attrs["model_identity"]
            or payload["data_schema"] != dataset._h.attrs["schema"]
            or payload["data_indices"] != dataset._indices.tolist()
            or payload["snr_db"] != dataset.snr_db.tolist()):
        raise ValueError("OAMP-Net2 checkpoint belongs to a different data model/channel/SNR grid")
    if payload.get("smoke") and not dataset._config.get("smoke"):
        raise ValueError("Smoke OAMP-Net2 checkpoints cannot enter formal evaluation")
    if dataset.split == "test" and "test.h5" in payload["identity"]["data"]:
        raise ValueError("Test data participated in OAMP-Net2 training")
    if not np.array_equal(dataset._indices, np.unique(dataset._indices)):
        raise ValueError("Invalid data-carrier ordering")
