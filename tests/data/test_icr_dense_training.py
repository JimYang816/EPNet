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


@pytest.mark.parametrize("precision", ("complex64", "complex128"))
def test_compensated_reference_cache_reuses_decoded_windows_exactly(tmp_path, monkeypatch, precision):
    import h5py
    from src.data.icr_training import _cpu_batches
    torch.set_num_threads(2)
    with threadpool_limits(2):
        config, prior, chain, archive = fixture()
        config["snr_db"] = [0, 10, 30]
        generation = tmp_path/"generation"
        generation.mkdir()
        generate(tmp_path/"data", generation, config, archive, chain, prior, DenseBackend(), {"commit": "cache"})
        with CompactDataset(tmp_path/"data"/"validation.h5") as dataset:
            options = dict(batch_size=2, precision=precision)
            original = list(dense_input_batches(dataset, **options))
            reads = []
            getitem = h5py.Dataset.__getitem__
            def read(storage, key):
                if storage.name == "/windows/reference_csi":
                    reads.append(key)
                return getitem(storage, key)
            monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
            cache = ReferenceCache(2**20)
            for _ in range(2):
                for (inputs, keys), (expected, expected_keys) in zip(
                        dense_input_batches(dataset, reference_cache=cache, **options), original, strict=True):
                    assert keys == expected_keys
                    torch.testing.assert_close(inputs["initial_channel"].dense("cpu"),
                        expected["initial_channel"].dense("cpu"), rtol=0, atol=0)
            windows = set(dataset._h["samples/window_index"][:].tolist())
            assert len(reads) == len(windows)  # one paired D1/D2 decode per window, across all SNRs/epochs
            assert cache.bytes == len(windows)*2*dataset._config["nfft"]**2*getattr(torch, precision).itemsize
            assert cache.bytes <= cache.max_bytes
            with monkeypatch.context() as scoped:
                scoped.setattr(torch.Tensor, "pin_memory", lambda value: value)
                for pinned in (False, True):
                    batches = _cpu_batches(dataset, include_reference=True, compact_reference=False,
                        bandwidth=2, cache=cache, pin_memory=pinned, batch_size=2, frame_limit=None,
                        precision=precision, operator_backend="dense_device", snr_indices=None, frame_indices=None)
                    inputs, targets, _ = next(batches)
                    assert targets["reference_channel"] is inputs["initial_channel"].reference
                    batches.close()


@pytest.mark.parametrize("blocks", (0, 1, 2, 3))
def test_reference_cache_eviction_keeps_exact_storage_budget(tmp_path, blocks):
    import h5py
    from types import SimpleNamespace
    values = np.arange(3*2*4*4, dtype=np.float32).reshape(3, 2, 4, 4).astype(np.complex64)
    path = tmp_path/"reference.h5"
    with h5py.File(path, "w") as h:
        h.attrs["model_identity"] = "eviction"
        h.create_dataset("windows/reference_csi", data=values, chunks=(1, 2, 4, 4), compression="lzf")
    cache = ReferenceCache(blocks*values[0, 0].nbytes)
    with h5py.File(path, "r") as h:
        dataset = SimpleNamespace(path=path, _h=h)
        for window in (0, 1, 2, 0):
            for block in (0, 1):
                actual = cache.get_full(dataset, window, block, "complex64")
                np.testing.assert_array_equal(actual[0].numpy(), values[window, block])
                assert cache.bytes <= cache.max_bytes
                assert cache.bytes == sum(value.untyped_storage().nbytes() for value in cache.entries.values())
