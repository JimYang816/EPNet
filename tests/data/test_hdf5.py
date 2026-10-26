from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from data_generation.src.pdd import load_pdd_config
from data_generation.src.dataset.pdd_export import initialize
from data_generation.src.transmitter.ofdm import make_frame
from src.data import NCS1Dataset


def test_hdf5_dual_d_shared_csi_masks_and_incomplete_guard(tmp_path):
    c = load_pdd_config()
    c.update(nfft=14, cp=6, fft_advance=1, guard_samples=4, null_per_edge=0,
             pilot_count=2, data_count=8, test_records=[1], odd_test_starts_seconds=[2], snr_db=[0, 30])
    chain = SimpleNamespace(noise_covariance=np.eye(14), noise_gain=1.0)
    prior = SimpleNamespace(source_records=(51,), model=SimpleNamespace(count=2, length=4, bandwidth_hz=1, tail_ratio=.001, basis=np.eye(80, 2)),
                            empirical_weight=0, scale=1, fit_domain="full_frame_including_cp",
                            delay_block_size=2, regularizer=1e-6)
    path = tmp_path/"sample.h5"
    h = initialize(path, c, chain, prior, {"commit": "fixture"})
    assert h["observations"].shape == (2, 1, 4, 14)
    assert h["reference_csi"].shape == (1, 2, 14, 14)
    assert h["estimated_csi"].shape == (2, 1, 2, 14, 14)
    assert h["reference_csi"].dtype == np.complex64
    frame = make_frame(c, 1, 2)
    h["reference_csi"][0] = np.eye(14, dtype=np.complex128)[None]
    h["estimated_csi"][:, 0] = np.eye(14, dtype=np.complex128)[None, None]
    h["observations"][:, 0] = frame.x[None]
    h["data_bits"][0] = frame.bits
    h["rf_noise_variance"][:, 0] = [.1, .0001]
    h.close()
    with pytest.raises(ValueError, match="incomplete"):
        NCS1Dataset(path)
    with h5py.File(path, "r+") as h:
        h.attrs["complete"] = True
        h["sample_complete"][0] = True
    with NCS1Dataset(path) as dataset:
        sample = dataset.sample(1, 0)
        np.testing.assert_array_equal(sample["bits"], frame.bits)
        np.testing.assert_allclose(sample["y"], frame.x, atol=3e-8)
        assert sample["noise_variance"] == .0001
        np.testing.assert_array_equal(sample["known"][1:3, dataset.masks["data"]], 0)
    for attribute, value, message in (("csi_compensation_json", "{}", "Reference-assisted"),
                                      ("schema", "ncs1-pdp-full-csi-v1", "schema"),
                                      ("schema", "ncs1-pdd-full-csi-v1", "schema"),
                                      ("schema", "ncs1-pdd-full-csi-v2", "schema"),
                                      ("frame_structure", "PDD", "frame layout"),
                                      ("estimation_mode", "frozen_prior_prefix_no_feedback_v1", "estimation mode")):
        with h5py.File(path, "r+") as h:
            original = h.attrs.get(attribute)
            h.attrs[attribute] = value
        with pytest.raises(ValueError, match=message):
            NCS1Dataset(path)
        with h5py.File(path, "r+") as h:
            if original is None:
                del h.attrs[attribute]
            else:
                h.attrs[attribute] = original
