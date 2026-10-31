"""Check preserved CSI trajectories, validation isolation and exact continuation."""
from dataclasses import replace
import json

import h5py
import pytest
import torch

from src.algorithms.icr_mmse.checkpoint import cpu_state, load_model
from src.algorithms.icr_mmse.model import ICRMMSEnet, layerwise_loss
from src.common.experiment import write_json
from src.common.numerics import DenseBackend
from src.training.icr_final_detector import run_pair
from src.training.icr_mmse import configure_stage, train
from tests.algorithms.test_icr_mmse import inputs
from tests.training.test_icr_mmse_training import experiment


@pytest.mark.parametrize("use_gate", (True, False))
def test_final_detector_update_preserves_every_csi_and_earlier_decision(use_gate):
    torch.set_num_threads(2)
    item = inputs(size=12)
    model = ICRMMSEnet(layers=3, bandwidth=2, window_size=8, center_size=4,
                      use_gate=use_gate, noise_mode="diagonal")
    with torch.no_grad():
        model.corrector.head.weight.normal_(std=.01)
    configure_stage(model, "C", final_equalizer_only=True)
    initial_state = cpu_state(model)
    before = model(**item)
    last = replace(before, probabilities=before.probabilities[:, -1:], channels=before.channels[-1:])
    labels = torch.zeros(2, int(item["data_mask"].sum()), dtype=torch.long)
    loss = layerwise_loss(last, labels, item["initial_channel"], item["data_mask"], lambda_h=.01)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss.backward()
    for name, value in model.named_parameters():
        if not name.startswith("equalizers.2."):
            assert value.grad is None
    optimizer.step()
    with torch.no_grad():
        after = model(**item)
    torch.testing.assert_close(before.bands, after.bands, rtol=0, atol=0)
    torch.testing.assert_close(before.symbols[:, :-1], after.symbols[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(before.variances[:, :-1], after.variances[:, :-1], rtol=0, atol=0)
    changed = [name for name, value in model.state_dict().items() if not torch.equal(value, initial_state[name])]
    assert changed and all(name.startswith("equalizers.2.") for name in changed)
    assert not torch.equal(before.probabilities[:, -1], after.probabilities[:, -1])


def test_paired_diagnostic_skips_ab_test_and_preserves_nmse(experiment, monkeypatch):
    directory, root = experiment
    original_file = h5py.File

    def no_test(path, *args, **kwargs):
        assert not str(path).replace("\\", "/").endswith("/test.h5")
        return original_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", no_test)
    options = dict(model_config=dict(layers=2, bandwidth=2, window_size=8, center_size=4,
                                    noise_mode="diagonal"), stage_epochs=(1, 1, 1),
        batch_size=1, seed=17, operator_backend="dense_device", prefetch_batches=0,
        reference_cache_mib=2, sample_order="snr_interleaved", per_snr_acceptance={})
    with h5py.File(directory/"train.h5") as handle:
        channel = str(handle.attrs["channel"])
    source = root/"source"
    paired = source/channel.lower()/"seed_17"
    train(directory, paired/"gate", DenseBackend(), stop_after_epochs=2, **options)
    train(directory, paired/"no_gate", DenseBackend(), stop_after_epochs=1,
        warmup_checkpoint=paired/"gate/warmup.pt",
        **dict(options, model_config=dict(options["model_config"], use_gate=False)))
    (source/"_control").mkdir()
    write_json(source/"_control/config.json", dict(input_view="compensated", seeds=[17],
        channels=[channel], datasets={channel: str(directory)}))
    output = root/"diagnostic"
    result = run_pair(source, output, channel, DenseBackend(), allow_smoke=True)
    assert result["complete"] and not result["test_access"] and result["split"] == "validation"
    assert all(len(item["history"]) == 2 and item["every_epoch_validation_nmse_unchanged"]
               for item in result["variants"].values())
    assert (output/"validation_ber_snr.png").exists()
    assert (output/"validation_nmse_snr.png").exists()
    with pytest.raises(ValueError, match="one or two epochs"):
        run_pair(source, root/"unbounded", channel, DenseBackend(), epochs=3)

    # Repeat just gate C with interruption. Identity must bind the frozen scope.
    controls = json.loads((output/"gate/training.json").read_text())["controls"]
    controls["model_config"] = controls.pop("model")
    controls.pop("input_view")
    controls.pop("sampler")
    resumed = root/"resumed"
    train(directory, resumed, DenseBackend(), stop_after_epochs=1, **controls)
    with pytest.raises(ValueError, match="identity changed"):
        train(directory, resumed, DenseBackend(), resume=True,
              **dict(controls, stage_c_final_equalizer_only=False))
    checkpoint = train(directory, resumed, DenseBackend(), resume=True, **controls)
    _, actual = load_model(checkpoint)
    _, expected = load_model(output/"gate/icr_mmse.pt")
    assert actual["history"] == expected["history"]
    for name, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][name], rtol=0, atol=0)
