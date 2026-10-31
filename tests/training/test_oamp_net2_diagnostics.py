"""Numerical audit independence and validation-only data boundaries."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.algorithms.oamp_net2.model import FEATURES, OAMPNet2
from src.training.oamp_net2_diagnostics import dense_reference, gradient_check, score


def features():
    generator = torch.Generator().manual_seed(941)
    a = torch.randn((4, 6, 3), dtype=torch.complex128, generator=generator)
    y = torch.randn((4, 6), dtype=torch.complex128, generator=generator)
    gram = a.mH@a
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    return a, y, dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors,
        projected_matched=(eigenvectors.mH@a.mH@y[..., None])[..., 0],
        observation_energy=y.abs().square().sum(-1), receive_count=torch.full((4,), 6.))


def test_actual_forward_and_directional_gradients_match_independent_reference():
    a, y, f = features()
    model = OAMPNet2(layers=3)
    with torch.no_grad():
        model.gamma.copy_(torch.tensor([.7, 1.3, .9]))
        model.theta.copy_(torch.tensor([.8, 1.1, .7]))
        model.xi.copy_(torch.tensor([.1, -.1, .2]))
    expected = dense_reference(a, y, model)
    actual = model(*(f[k] for k in FEATURES), return_layers=True)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    target = torch.ones((4, 3), dtype=torch.complex128)*(1+1j)/np.sqrt(2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = gradient_check(model, f, target)
    assert result["maximum_scaled_error"] < 1e-7
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_full_validation_scoring_counts_snr_and_layers_without_dataset_access():
    _, _, f = features()
    target = torch.ones((4, 3), dtype=torch.complex128)*(1+1j)/np.sqrt(2)
    arrays = {k: v.numpy() for k, v in dict(f, symbols=target).items()}
    cache = SimpleNamespace(count=4, arrays=arrays,
        batch=lambda indices, device: {k: torch.as_tensor(v[indices], device=device) for k, v in arrays.items()})
    result = score(OAMPNet2(layers=3), cache, [0, 30], batch_size=3, layers=True, baselines=True)
    assert len(result["by_snr"]) == 2
    assert all(r["total_bits"] == 12 and len(r["ber"]) == 3 for r in result["by_snr"])
    for baseline in result["baselines"].values():
        assert all(r["total_bits"] == 12 for r in baseline["by_snr"])


def test_diagnostic_uses_only_train_validation_and_its_weights_cannot_enter_formal_evaluation(tmp_path, monkeypatch):
    import h5py
    from threadpoolctl import threadpool_limits
    from data_generation.tests.test_compact import fixture
    from data_generation.src.dataset.compact_export import generate
    from src.algorithms.oamp_net2.checkpoint import load_model
    from src.common.numerics import DenseBackend
    from src.training.oamp_net2 import train
    from src.training.oamp_net2_diagnostics import diagnose
    config, prior, chain, archive = fixture()
    config["snr_db"] = [0, 30]
    data, generation = tmp_path/"data", tmp_path/"generation"
    generation.mkdir()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with threadpool_limits(2):
            generate(data, generation, config, archive, chain, prior, DenseBackend(), {"commit": "synthetic"})
            checkpoint = train(data, tmp_path/"model", DenseBackend(), epochs=1, batch_size=2,
                               model_config={"layers": 3})
            original_bytes = checkpoint.read_bytes()
            file_class = h5py.File
            def protected(path, *args, **kwargs):
                if Path(path).name == "test.h5":
                    raise AssertionError("Diagnostic opened test data")
                return file_class(path, *args, **kwargs)
            monkeypatch.setattr(h5py, "File", protected)
            result = diagnose(data, checkpoint, tmp_path/"diagnostic", DenseBackend(),
                              joint_epochs=1, snr_epochs=1, batch_size=2)
            assert result["complete"] and not result["test_data_opened"]
            assert len(result["trials"]) == 3
            assert checkpoint.read_bytes() == original_bytes
            with pytest.raises(ValueError, match="identity mismatch"):
                load_model(tmp_path/"diagnostic/joint_lr1e-4/best.pt")
    finally:
        torch.set_num_threads(old_threads)
