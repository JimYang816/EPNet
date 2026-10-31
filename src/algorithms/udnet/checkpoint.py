"""UDNet checkpoint identity and safe state serialization."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from src.algorithms.udnet.model import UDNet
from src.data.udnet import PREPROCESSING
from src.data.udnet_cache import PREPROCESS_CODE
from src.evaluation.fingerprints import source_fingerprint

SCHEMA = "watermark-udnet-joint-snr-v2"
PAPER_DOI = "10.1109/TVT.2022.3230143"
INFERENCE_CODE = PREPROCESS_CODE+("src.algorithms.udnet.model", "src.algorithms.udnet.inference",
                                  "src.algorithms.udnet.checkpoint", "src.algorithms.interface")


def atomic_save(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".partial")
    torch.save(state, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def random_state(generator):
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
                python=random.getstate(), shuffle=generator.get_state())


def restore_random_state(state, generator):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA resume requires CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    random.setstate(state["python"])
    generator.set_state(state["shuffle"])


def load_model(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (payload.get("schema") != SCHEMA or payload.get("paper_doi") != PAPER_DOI
            or payload.get("preprocessing") != PREPROCESSING
            or payload.get("inference_code") != source_fingerprint(INFERENCE_CODE)
            or payload.get("training_scope") != "one_channel_all_snr_D1_D2"):
        raise ValueError("UDNet checkpoint model/preprocessing/code identity mismatch")
    model = UDNet(**payload["model_config"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(payload["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite UDNet checkpoint")
    model.eval().requires_grad_(False)
    return model, payload


def validate_dataset(payload, dataset):
    if payload.get("input_view", "compensated") != dataset.input_view:
        raise ValueError("UDNet checkpoint CSI input view mismatch; retrain on the selected view")
    if (payload["channel"] != dataset.channel or payload["model_identity"] != dataset._h.attrs["model_identity"]
            or payload["data_schema"] != dataset._h.attrs["schema"]
            or payload["data_indices"] != dataset._indices.tolist()
            or payload["snr_db"] != dataset.snr_db.tolist()):
        raise ValueError("UDNet checkpoint belongs to a different data model/channel/SNR grid")
