"""The user's fixed compensated CSI is the default across both dense paths."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.contract import DEFAULT_INPUT_VIEW, target_nmse_db, validate_compensated_contract
from src.data.icr_mmse import input_batches
from src.data.icr_training import ReferenceCache, training_batches


@pytest.mark.parametrize("channel", ("NOF1", "NCS1"))
def test_default_csi_nmse_matches_fixed_targets_on_both_dense_paths(tmp_path, channel):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        config, prior, chain, archive = fixture()
        config.update(channel=channel, snr_db=[0, 15, 30])
        generation = tmp_path/"generation"
        generation.mkdir()
        data = tmp_path/"data"
        generate(data, generation, config, archive, chain, prior, DenseBackend(), {"commit": "fixed-csi"})
        for split in ("train", "validation", "test"):
            with CompactDataset(data/(split+".h5")) as dataset:
                assert dataset.input_view == DEFAULT_INPUT_VIEW == "compensated"
                # Exercise formal target validation, too, on the miniature fixture.
                dataset._config["smoke"] = False
                validate_compensated_contract(dataset)
                values = [[] for _ in dataset.snr_db]
                for inputs, keys in input_batches(dataset, precision="complex128", batch_size=3):
                    for i, (si, frame, block) in enumerate(keys):
                        reference = dataset.reference_csi(frame)[block]
                        h = inputs["initial_channel"][i].numpy()
                        values[si].append(np.sum(abs(h-reference)**2)/np.sum(abs(reference)**2))
                actual = 10*np.log10([np.mean(x) for x in values])
                np.testing.assert_allclose(actual, target_nmse_db(channel, dataset.snr_db), atol=2e-12)
                cache = ReferenceCache(2**20)
                for depth in (0, 2):
                    expected = input_batches(dataset, batch_size=3, frame_limit=2)
                    with training_batches(dataset, batch_size=3, frame_limit=2,
                            operator_backend="dense_device", compact_reference=False,
                            cache=cache, prefetch_batches=depth) as batches:
                        for (inputs, target, keys), (reference, expected_keys) in zip(batches, expected, strict=True):
                            assert keys == expected_keys
                            for name in inputs:
                                torch.testing.assert_close(inputs[name], reference[name], rtol=2e-6, atol=1e-6)
                dataset._h.close()
                import h5py
                with h5py.File(data/(split+".h5"), "r+") as h:
                    h["compensation/alpha"][0] *= 1.1
                dataset._h = h5py.File(data/(split+".h5"), "r")
                with pytest.raises(ValueError, match="user-fixed"):
                    validate_compensated_contract(dataset)


def test_default_configs_are_pinned_to_compensated_single_seed():
    root = Path(__file__).resolve().parents[2]
    icr = json.loads((root/"configs/training/icr_mmse.json").read_text())
    udnet = json.loads((root/"configs/training/udnet.json").read_text())
    comparison = json.loads((root/"configs/evaluation/ber_comparison.json").read_text())
    assert icr["input_view"] == comparison["input_view"] == "compensated"
    assert udnet["training"]["input_view"] == udnet["evaluation"]["input_view"] == "compensated"
    assert icr["selection"]["seeds"] == [20261028]
