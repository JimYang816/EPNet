"""Atomic learning checkpoints and restartable random generators."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch


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
