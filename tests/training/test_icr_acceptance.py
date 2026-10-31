"""Reject hidden per-SNR regressions and reproduce interrupted interleaved runs."""
import h5py
import pytest
import torch

from src.common.numerics import DenseBackend
from src.training.icr_acceptance import make_limits, assess
from src.training.icr_mmse import train, _selection_key
from tests.training.test_icr_mmse_training import experiment


def metrics(ber=(.2, .02), nmse=(.2, .02)):
    return dict(ber=sum(ber)/2, initial_nmse=1., corrected_nmse=sum(nmse)/2,
        per_snr=[dict(snr_db=snr, bits=10000, blocks=20, ber=b, initial_nmse=1., corrected_nmse=n)
                 for snr, b, n in zip((0, 30), ber, nmse)])


def test_per_snr_rejects_regression_hidden_by_better_global_means():
    reference = metrics()
    limits = make_limits(reference, {})
    ber_regression = metrics(ber=(.1, .021), nmse=(.1, .01))
    nmse_regression = metrics(ber=(.1, .01), nmse=(.1, .021))
    for bad in (ber_regression, nmse_regression):
        assert _selection_key("C", bad) < _selection_key("C", reference)
        assert not assess(bad, limits)["eligible"]
        assert assess(bad, limits)["violations"][0]["snr_db"] == 30
        for stage in ("B", "C"):
            assert _selection_key(stage, reference, per_snr_limits=limits) < _selection_key(stage, bad, per_snr_limits=limits)
    assert assess(reference, limits)["eligible"]
    assert assess(metrics(ber=(.19, .019), nmse=(.19, .019)), limits)["eligible"]
    with pytest.raises(ValueError, match="coverage changed"):
        assess(dict(reference, per_snr=reference["per_snr"][:1]), limits)
    with pytest.raises(ValueError, match="finite"):
        assess(metrics(nmse=(float("nan"), .01)), limits)


@pytest.mark.parametrize("stop", (1, 2, 4))
def test_interleaved_resume_matches_uninterrupted_training_and_bounds(experiment, monkeypatch, stop):
    directory, root = experiment
    actual_file = h5py.File

    def no_test(path, *args, **kwargs):
        assert not str(path).replace("\\", "/").endswith("/test.h5")
        return actual_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", no_test)
    options = dict(model_config=dict(layers=2, bandwidth=2, window_size=8, center_size=4, noise_mode="diagonal"),
        stage_epochs=(1, 2, 2), batch_size=1, seed=17, operator_backend="dense_device", prefetch_batches=2,
        reference_cache_mib=2, sample_order="snr_interleaved", per_snr_acceptance={},
        stage_c_learning_rate=1e-4, stage_c_correction_learning_rate=1e-5)
    full = train(directory, root/"full", DenseBackend(), **options)
    train(directory, root/"resumed", DenseBackend(), stop_after_epochs=stop, **options)
    resumed = train(directory, root/"resumed", DenseBackend(), resume=True, **options)
    a, b = [torch.load(p, weights_only=True) for p in (full, resumed)]
    assert a["history"] == b["history"]
    assert a["per_snr_selection"] == b["per_snr_selection"]
    for key, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
    for selection in a["per_snr_selection"].values():
        assert assess(selection["selected_validation"], selection["limits"])["eligible"]
    for changed in (dict(sample_order="sequential"), dict(per_snr_acceptance=dict(nmse_relative_tolerance=.01))):
        with pytest.raises(ValueError, match="identity changed"):
            train(directory, root/"resumed", DenseBackend(), resume=True, **dict(options, **changed))
