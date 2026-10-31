"""Faithful dense construction, receiver-only reads and cached supervision."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits

from data_generation.src.receiver.chain import taps_to_matrix
from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.icr_dense_training import DenseCoefficients, dense_input_batches
from src.data.icr_mmse import input_batches, supervision
from src.data.icr_training import ReferenceCache, training_batches


@pytest.mark.parametrize("size", (9, 16))
@pytest.mark.parametrize("precision", ("complex64", "complex128"))
def test_device_dense_construction_matches_original_double_fft(size, precision):
    torch.set_num_threads(2)
    rng = np.random.default_rng(512)
    basis = rng.normal(size=(size, 3)) + 1j*rng.normal(size=(size, 3))
    coefficients = rng.normal(size=(2, 3, size-2)) + 1j*rng.normal(size=(2, 3, size-2))
    expected = np.stack([taps_to_matrix(np.einsum("nq,ql->nl", basis, value, optimize=True))
                         for value in coefficients]).astype(precision)
    prepared = DenseCoefficients(torch.as_tensor(coefficients), torch.as_tensor(basis), getattr(torch, precision))
    actual = prepared.dense("cpu")
    torch.testing.assert_close(actual, torch.as_tensor(expected),
                               rtol=2e-6 if precision == "complex64" else 3e-14,
                               atol=1e-6 if precision == "complex64" else 3e-14)


def test_receiver_and_prefetched_dense_pipeline_preserve_legacy_fields(tmp_path, monkeypatch):
    import h5py
    torch.set_num_threads(2)
    with threadpool_limits(2):
        config, prior, chain, archive = fixture()
        config["snr_db"] = [0, 10]
        generation = tmp_path/"generation"
        generation.mkdir()
        generate(tmp_path/"data", generation, config, archive, chain, prior, DenseBackend(), {"commit": "dense-device"})
        cache = ReferenceCache(2**20)
        with CompactDataset(tmp_path/"data"/"validation.h5", input_view="raw") as dataset:
            original = h5py.Group.__getitem__
            def guard(group, name):
                if any(value in str(name) for value in ("reference", "bits", "compensation", "actual_snr")):
                    raise AssertionError("Truth accessed by receiver input reader")
                return original(group, name)
            with monkeypatch.context() as scoped:
                scoped.setattr(h5py.Group, "__getitem__", guard)
                receiver, keys = next(dense_input_batches(dataset, batch_size=2, frame_limit=1))
            expected, expected_keys = next(input_batches(dataset, batch_size=2, frame_limit=1))
            assert keys == expected_keys
            torch.testing.assert_close(receiver["initial_channel"].dense("cpu"), expected["initial_channel"])
            for depth in (0, 2):
                with training_batches(dataset, batch_size=2, frame_limit=1, operator_backend="dense_device",
                        compact_reference=False, cache=cache, prefetch_batches=depth) as batches:
                    inputs, targets, keys = next(batches)
                original_target = supervision(dataset, keys)
                for key in inputs:
                    torch.testing.assert_close(inputs[key], expected[key])
                for key in targets:
                    torch.testing.assert_close(targets[key], original_target[key], rtol=0, atol=0)
            assert cache.hits > 0
            assert cache.bytes <= cache.max_bytes
