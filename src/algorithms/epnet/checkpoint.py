"""One versioned EPNet per channel, shared by every trained SNR."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.algorithms.epnet.model import EPNet

SCHEMA = "watermark-epnet-joint-snr-v2"
PAPER_DOI = "10.1109/TWC.2020.3030882"


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class EPNetBank:
    def __init__(self, payload, device="cpu"):
        if (payload.get("schema") != SCHEMA or not payload.get("complete")
                or payload.get("paper_doi") != PAPER_DOI
                or payload.get("training_scope") != "joint_snr"
                or len(payload.get("models", [])) != 1):
            raise ValueError("Incomplete or incompatible EPNet checkpoint")
        self.metadata = {k: v for k, v in payload.items() if k != "models"}
        self.snr_db = np.asarray(payload["snr_db"], dtype=float)
        if (self.snr_db.ndim != 1 or not len(self.snr_db) or not np.isfinite(self.snr_db).all()
                or len(np.unique(self.snr_db)) != len(self.snr_db)):
            raise ValueError("Invalid checkpoint SNR grid")
        self.models = []
        for state in payload["models"]:
            model = EPNet(**payload["model_config"]).to(device=device)
            model.load_state_dict(state, strict=True)
            if not torch.isfinite(model.beta).all():
                raise ValueError("Nonfinite EPNet parameters")
            model.eval().requires_grad_(False)
            self.models.append(model)

    def model_for_snr(self, snr_db):
        selected = np.flatnonzero(np.isclose(self.snr_db, snr_db, rtol=0, atol=1e-8))
        if len(selected) != 1:
            raise ValueError(f"No trained EPNet for SNR {snr_db}")
        return self.models[0]

    def validate_dataset(self, dataset):
        if (self.metadata["channel"] != dataset.channel
                or self.metadata["model_identity"] != dataset._h.attrs["model_identity"]
                or self.metadata["data_schema"] != dataset._h.attrs["schema"]
                or self.metadata["data_count"] != len(dataset._indices)):
            raise ValueError("EPNet checkpoint belongs to a different channel/data model")
        for snr in dataset.snr_db:
            self.model_for_snr(float(snr))


def load_bank(path, device="cpu"):
    return EPNetBank(torch.load(Path(path), map_location="cpu", weights_only=True), device)
