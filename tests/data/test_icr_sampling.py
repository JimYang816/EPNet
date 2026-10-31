"""Sampling changes order, never sample identities, compensation or supervision."""
from collections import Counter

import torch
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.data.icr_sampling import batch_schedule, epoch_sampling_seed
from src.data.icr_training import training_batches


def test_interleaved_schedule_is_complete_balanced_and_reproducible():
    frames, snrs = list(range(5)), [0, 1, 2]
    seed = epoch_sampling_seed(17, "B", 2)
    order = list(batch_schedule(frames, snrs, 2, sampling_seed=seed))
    assert order == list(batch_schedule(frames, snrs, 2, sampling_seed=seed))
    assert order != list(batch_schedule(frames, snrs, 2, sampling_seed=epoch_sampling_seed(17, "B", 3)))
    assert len(order) == 18 and sum(len(indices) for _, _, indices in order) == 30
    keys = [(si, frame, block) for si, block, indices in order for frame in indices]
    assert Counter(keys) == Counter((si, frame, block) for si in snrs for frame in frames for block in range(2))
    assert all(a[0] != b[0] for a, b in zip(order, order[1:]))
    for i in range(0, len(order), len(snrs)):
        block = order[i:i+len(snrs)]
        assert {item[0] for item in block} == set(snrs)
        assert all(item[1:] == block[0][1:] for item in block)


def test_interleaved_hdf5_batches_preserve_exact_inputs_labels_and_tail(tmp_path):
    torch.set_num_threads(2)
    with threadpool_limits(2):
        config, prior, chain, archive = fixture()
        config["snr_db"] = [0, 10, 30]
        config["learning_dataset"]["starts_seconds"] = [2, 4, 6]
        generation = tmp_path/"generation"
        generation.mkdir()
        generate(tmp_path/"data", generation, config, archive, chain, prior, DenseBackend(), {"commit": "sampling"})
        with CompactDataset(tmp_path/"data/train.h5") as dataset:
            expected = {}
            with training_batches(dataset, batch_size=2, prefetch_batches=0) as batches:
                for inputs, targets, keys in batches:
                    for i, key in enumerate(keys):
                        expected[key] = {name: value[i].clone() for name, value in {**inputs, **targets}.items()
                                         if name in ("initial_channel", "observations", "rf_variance", "known",
                                                     "reference_channel", "classes", "symbols", "snr_indices")}
            observed = []
            for depth in (0, 2):
                order = []
                with training_batches(dataset, batch_size=2, prefetch_batches=depth, sampling_seed=71) as batches:
                    for inputs, targets, keys in batches:
                        order.extend(keys)
                        for i, key in enumerate(keys):
                            for name, value in expected[key].items():
                                torch.testing.assert_close({**inputs, **targets}[name][i], value, rtol=0, atol=0)
                assert Counter(order) == Counter(expected.keys())
                observed.append(order)
            assert observed[0] == observed[1]
            assert observed[0] != list(expected)
