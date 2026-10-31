import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset, data_system
from src.evaluation.comparison import evaluate, test_batch as prepare_test_batch
from src.algorithms.MMSE.single_tap import mmse


def test_cpu_end_to_end_mmse_full_test_counts_and_resume(tmp_path):
    with threadpool_limits(2):
        c, prior, chain, archive = fixture()
        c["learning_dataset"]["repeats"]["train"] = 4
        data, generation, run = tmp_path/"data", tmp_path/"generation", tmp_path/"evaluation"
        generation.mkdir()
        generate(data, generation, c, archive, chain, prior, DenseBackend(),
                 dict(commit="test", archive_sha256="test"))
        cache = tmp_path/"cache"
        report = evaluate(data/"test.h5", run, DenseBackend(), batch_size=2, cache_dir=cache)
        assert report["complete"] and report["frame_count"] == 2
        assert report["bits_per_snr"] == 64
        assert set(report["curves"]) == {"Single-tap MMSE", "Full-matrix LMMSE"}
        assert "robust_covariance" not in report
        assert (run/"ber_snr.png").is_file()
        assert (run/"nmse_snr.png").is_file()
        for curve in report["curves"].values():
            with h5py.File(curve["provenance"]["cache_file"], "r") as h:
                assert np.all(h["complete"][:]) and h.attrs["complete"]
                assert h["bit_errors"].shape == (7, 2, 2)
        resumed = evaluate(data/"test.h5", run, DenseBackend(), batch_size=1, cache_dir=cache)
        assert all(c["provenance"]["reused"] for c in resumed["curves"].values())
        for name in report["curves"]:
            assert resumed["curves"][name]["errors"] == report["curves"][name]["errors"]
        with pytest.raises(ValueError, match="test split"):
            evaluate(data/"validation.h5", tmp_path/"invalid", DenseBackend())
        with CompactDataset(data/"test.h5") as ds:
            for d in range(2):
                channel, y, variances, single = prepare_test_batch(ds, 1, [0, 1], d, DenseBackend())
                for j, fi in enumerate([0, 1]):
                    inputs = ds.model_input(1, fi)
                    residual, full = data_system(inputs)
                    np.testing.assert_allclose(channel[j], full[d], atol=2e-15)
                    np.testing.assert_allclose(y[j], residual[d], atol=2e-15)
                    expected = mmse(inputs["csi"][d], inputs["observations"][d+1], inputs["masks"]["data"],
                                    variances[j]*np.diag(inputs["noise"]["unit_rx_covariance"]).real)
                    np.testing.assert_allclose(single[j], expected, atol=2e-15)
