"""Frozen deployment preserves learned predictions, including CSI feedback."""
import pytest
import torch

from src.algorithms.icr_mmse.inference import prepare_frozen_inference
from src.algorithms.icr_mmse.model import ICRMMSEnet
from tests.algorithms.test_icr_mmse import inputs


@pytest.mark.parametrize("noise_mode", ["diagonal", "full"])
@pytest.mark.parametrize("learned_head", [False, True])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_preparation_preserves_predictions_on_distinct_inputs(noise_mode, learned_head, precision):
    torch.set_num_threads(2)
    model = ICRMMSEnet(layers=3, bandwidth=1, detector_mode="budgeted",
                       initial_soft_damping=.8, noise_mode=noise_mode, precision=precision)
    with torch.no_grad():
        for p in model.corrector.parameters():
            p.add_(.01*torch.randn_like(p))
        for layer in model.equalizers:
            layer.csi_loading_log.add_(.2)
            layer.temperature_log.add_(.1)
            if learned_head:
                for p in layer.classifier.parameters():
                    p.add_(.01*torch.randn_like(p))
    model.eval().requires_grad_(False)
    batches = [inputs(size, 2, getattr(torch, precision)) for size in (19, 19, 23)]
    batches[1]["observations"] *= .7
    with torch.inference_mode():
        expected = [model(**batch) for batch in batches]
        prepare_frozen_inference(model)
        prepare_frozen_inference(model)  # preparation is idempotent
        for batch, ref in zip(batches, expected):
            actual = model(**batch)
            for name in ("probabilities", "log_probabilities", "symbols", "variances", "bands"):
                torch.testing.assert_close(getattr(actual, name), getattr(ref, name), rtol=0, atol=0)
            for left, right in zip(actual.diagnostics, ref.diagnostics):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    model.train()
    with pytest.raises(RuntimeError, match="cannot be trained"):
        model(**batches[0])


def test_preparation_rejects_trainable_parameters():
    model = ICRMMSEnet(detector_mode="budgeted").eval()
    with pytest.raises(ValueError, match="frozen weights"):
        prepare_frozen_inference(model)
