import platform

import pytest
import torch

from src.common.numerics import default_device


@pytest.mark.parametrize("system,cuda,expected", [
    ("Windows", True, "cpu"), ("Windows", False, "cpu"),
    ("Linux", True, "cuda"), ("Linux", False, "cpu"),
])
def test_default_device(monkeypatch, system, cuda, expected):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert default_device() == expected
