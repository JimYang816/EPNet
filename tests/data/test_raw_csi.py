"""Raw-view receiver inputs are independent of audit and supervision fields."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from data_generation.src.dataset.compact_math import reconstruct
from src.algorithms.MMSE.single_tap import equalize_batch
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.inputs import prepare_batch


@pytest.fixture
def raw_data(tmp_path):
    with threadpool_limits(2):
        c, prior, chain, archive = fixture()
        c["snr_db"] = [0, 10]
        generation = tmp_path/"generation"
        generation.mkdir()
        generate(tmp_path/"data", generation, c, archive, chain, prior, DenseBackend(), {"commit": "raw-test"})
        yield tmp_path/"data"


def test_raw_reader_and_batches_do_not_access_forbidden_fields(raw_data, monkeypatch):
    original = h5py.Group.__getitem__
    forbidden = ("reference", "compensation", "actual_snr", "bits", "classes", "labels", "model_residual",
                 "effective_rf_variance", "training_calibration/")
    def guard(group, name):
        if any(key in str(name) for key in forbidden):
            raise AssertionError("Forbidden input read: "+str(name))
        return original(group, name)
    monkeypatch.setattr(h5py.Group, "__getitem__", guard)
    monkeypatch.setattr(CompactDataset, "_channels", lambda *a: pytest.fail("Mixed CSI path accessed"))
    with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
        result = ds.model_input(0, 0)
        assert set(result["noise"]) == {"rf_variance", "unit_rx_covariance"}
        assert "uncertainty" not in result
        batch = prepare_batch(ds, 0, [0], 0, DenseBackend())
        np.testing.assert_allclose(batch.channel[0], result["csi"][0], atol=2e-15)


def test_changing_test_truth_alpha_and_diagnostics_preserves_inputs_and_predictions(raw_data):
    def capture():
        with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
            inp = ds.raw_model_input(1, 0)
            batch = prepare_batch(ds, 1, [0], 0, DenseBackend())
            pred = equalize_batch(batch, DenseBackend()).symbols
            return inp, batch, pred
    before, batch_before, prediction_before = capture()
    with h5py.File(raw_data/"test.h5", "r+") as h:
        h["windows/reference_csi"][:] = 123+87j
        h["windows/reference_projection"][:] = -321-75j
        h["samples/bits"][:] ^= np.uint8(1)
        h["compensation/alpha"][:] = .00001
        h["samples/actual_snr_db"][:] = -99
    after, batch_after, prediction_after = capture()
    for name in ("csi", "observations", "known", "data_indices", "carrier_positions"):
        np.testing.assert_array_equal(before[name], after[name])
    np.testing.assert_array_equal(batch_before.channel, batch_after.channel)
    np.testing.assert_array_equal(prediction_before, prediction_after)


def test_raw_is_unmixed_reconstruction_and_historical_view_is_explicit(raw_data):
    with CompactDataset(raw_data/"test.h5", input_view="raw") as raw, CompactDataset(raw_data/"test.h5", input_view="compensated") as old:
        expected = reconstruct(raw._model, raw._h["samples/raw_coefficients"][0, 0].astype(complex))
        np.testing.assert_array_equal(raw.model_input(0, 0)["csi"], expected)
        assert old.model_input(0, 0)["input_view"] == "compensated"
        assert not np.allclose(expected, old.model_input(0, 0)["csi"])


def test_icr_batch_input_and_separate_supervision(raw_data, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.data.icr_mmse import input_batches, supervision
    with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
        original = h5py.Group.__getitem__
        def guard(group, name):
            if any(key in str(name) for key in ("reference", "compensation", "bits", "actual_snr")):
                raise AssertionError("Forbidden ICR input read: "+str(name))
            return original(group, name)
        with monkeypatch.context() as scoped:
            scoped.setattr(h5py.Group, "__getitem__", guard)
            inputs, keys = next(input_batches(ds, batch_size=2, snr_indices=[1]))
        target = supervision(ds, keys)
        assert inputs["observations"].shape == (2, 14)
        assert inputs["initial_channel"].shape == (2, 14, 14)
        assert target["reference_channel"].shape == (2, 14, 14)
        assert target["classes"].shape == (2, 8)
        for name in ("observations", "initial_channel", "known", "unit_covariance"):
            assert inputs[name].dtype == torch.complex64
        assert inputs["rf_variance"].dtype == torch.float32
        assert target["reference_channel"].dtype == target["symbols"].dtype == torch.complex64
        assert target["classes"].dtype == target["records"].dtype == torch.long
        assert keys == [(1, 0, 0), (1, 1, 0)]
        with monkeypatch.context() as scoped:
            def no_reference(group, name):
                if "reference" in str(name):
                    raise AssertionError("Warmup accessed CSI reference")
                return original(group, name)
            scoped.setattr(h5py.Group, "__getitem__", no_reference)
            warmup = supervision(ds, keys, include_reference=False)
            assert "reference_channel" not in warmup
        torch.testing.assert_close(inputs["initial_channel"][0],
            torch.as_tensor(ds.raw_model_input(1, 0)["csi"][0], dtype=torch.complex64))


def test_icr_double_precision_is_explicit_and_invalid_precision_is_rejected(raw_data):
    torch = pytest.importorskip("torch")
    from src.data.icr_mmse import input_batches, supervision
    with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
        inputs, keys = next(input_batches(ds, batch_size=1, precision="complex128"))
        target = supervision(ds, keys, precision="complex128")
        assert inputs["initial_channel"].dtype == inputs["unit_covariance"].dtype == torch.complex128
        assert inputs["rf_variance"].dtype == torch.float64
        assert target["reference_channel"].dtype == target["symbols"].dtype == torch.complex128
        with pytest.raises(ValueError, match="ICR precision"):
            next(input_batches(ds, precision="complex32"))
        with pytest.raises(ValueError, match="ICR precision"):
            supervision(ds, keys, precision="complex32")


@pytest.mark.parametrize("precision", ("complex64", "complex128"))
def test_icr_transfers_requested_precision_directly_from_raw_cpu_arrays(raw_data, monkeypatch, precision):
    torch = pytest.importorskip("torch")
    from src.data.icr_mmse import input_batches, supervision
    complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    numpy_complex = np.dtype(precision)
    numpy_real = np.dtype("float32" if precision == "complex64" else "float64")
    original = torch.as_tensor
    transferred = []
    def checked_as_tensor(value, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if isinstance(value, np.ndarray) and dtype in (complex_dtype, real_dtype):
            expected = numpy_complex if dtype == complex_dtype else numpy_real
            assert value.dtype == expected
            transferred.append(value.dtype)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch, "as_tensor", checked_as_tensor)
    with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
        monkeypatch.setattr(ds, "model_input", lambda *args: pytest.fail("ICR used the baseline preparation path"))
        batches = list(input_batches(ds, batch_size=1, frame_limit=1, snr_indices=[1], precision=precision))
        assert [keys for _, keys in batches] == [[(1, 0, 0)], [(1, 0, 1)]]
        raw = ds.raw_model_input(1, 0)
        for d, (inputs, keys) in enumerate(batches):
            np.testing.assert_array_equal(inputs["initial_channel"][0].numpy(), raw["csi"][d].astype(numpy_complex))
            np.testing.assert_array_equal(inputs["observations"][0].numpy(), raw["observations"][d+1].astype(numpy_complex))
            supervision(ds, keys, precision=precision)
    assert numpy_complex in transferred and numpy_real in transferred


def test_fixed_icr_prediction_survives_truth_changes(raw_data, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.algorithms.icr_mmse import ICRMMSEnet
    from src.data.icr_mmse import input_batches
    torch.set_num_threads(2)
    torch.manual_seed(314)
    model = ICRMMSEnet(layers=2, bandwidth=1, window_size=8, center_size=4).eval()
    def prediction():
        original = h5py.Group.__getitem__
        def guard(group, name):
            if any(key in str(name) for key in ("reference", "compensation", "bits", "actual_snr", "model_residual")):
                raise AssertionError("Forbidden fixed-model inference read: "+str(name))
            return original(group, name)
        with monkeypatch.context() as scoped, torch.no_grad():
            scoped.setattr(h5py.Group, "__getitem__", guard)
            with CompactDataset(raw_data/"test.h5", input_view="raw") as ds:
                inputs, _ = next(input_batches(ds, batch_size=1, snr_indices=[1]))
                return inputs, model(**inputs).probabilities
    before_inputs, before = prediction()
    with h5py.File(raw_data/"test.h5", "r+") as h:
        h["windows/reference_csi"][:] = -500+99j
        h["samples/bits"][:] ^= np.uint8(1)
        h["compensation/alpha"][:] = .9
        h["samples/actual_snr_db"][:] = -123
    after_inputs, after = prediction()
    for name in before_inputs:
        torch.testing.assert_close(before_inputs[name], after_inputs[name], rtol=0, atol=0)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
