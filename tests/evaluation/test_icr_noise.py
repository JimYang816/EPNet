"""CPU integration checks for the paired noise-model comparison."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.algorithms.icr_mmse.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, atomic_save, cpu_state
from src.algorithms.icr_mmse.model import ICRMMSEnet
from src.common.experiment import file_sha256
from src.common.numerics import DenseBackend
from src.data.compact import CompactDataset
from src.evaluation.fingerprints import source_fingerprint
from src.evaluation import icr_noise


@pytest.fixture
def small_comparison(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(1):
            c, prior, chain, archive = fixture()
            c["snr_db"] = [0, 10]
            c["learning_dataset"]["records"]["test"] = [3, 4, 5]
            directory, generation = tmp_path / "data", tmp_path / "generation"
            generation.mkdir()
            generate(directory, generation, c, archive, chain, prior, DenseBackend(),
                     dict(commit="paired noise fixture"))
            torch.manual_seed(74)
            model = ICRMMSEnet(layers=2, bandwidth=2, window_size=6, center_size=3,
                              hidden_size=4, precision="complex128", noise_mode="full")
            # Exercise feedback-dependent CSI updates rather than zero-initialized repair.
            with torch.no_grad():
                model.corrector.head.weight[:10].normal_(std=.002)
            checkpoint = tmp_path / "icr.pt"
            state = cpu_state(model)
            with CompactDataset(directory / "train.h5", input_view="compensated") as dataset:
                atomic_save(checkpoint, dict(schema=SCHEMA, input_view="compensated", model_config=model.config,
                    feedback_layout=FEEDBACK_LAYOUT,
                    inference_code=source_fingerprint(INFERENCE_CODE), model=state,
                    training_scope="one_channel_all_snr_D1_D2", channel=dataset.channel,
                    data_schema=str(dataset._h.attrs["schema"]),
                    model_identity=str(dataset._h.attrs["model_identity"]),
                    data_indices=dataset._indices.tolist(), snr_db=dataset.snr_db.tolist()))
            yield directory, checkpoint, state
    finally:
        torch.set_num_threads(previous_threads)


def test_paired_noise_comparison_preserves_weights_samples_and_scoring_order(small_comparison, tmp_path, monkeypatch):
    directory, checkpoint, state = small_comparison
    original_forward = ICRMMSEnet.forward
    original_supervision = icr_noise.supervision
    pending = {"full": [], "diagonal": []}
    scored_batches = []
    checked_weights = set()
    paths = [directory / (split + ".h5") for split in ("train", "validation", "test", "model")]
    paths.append(checkpoint)
    hashes_before = {path: file_sha256(path) for path in paths}

    def tracked_forward(model, **inputs):
        mode = model.config["noise_mode"]
        if mode not in checked_weights:
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            checked_weights.add(mode)
        pending[mode].append((inputs["observations"].data_ptr(), inputs["initial_channel"].data_ptr()))
        return original_forward(model, **inputs)

    def tracked_supervision(dataset, keys, **options):
        # Each mode must finish every repeated prediction before labels/reference CSI open.
        assert len(pending["full"]) == len(pending["diagonal"]) >= 2
        assert len(set(pending["full"] + pending["diagonal"])) == 1
        scored_batches.append(tuple(keys))
        pending["full"].clear()
        pending["diagonal"].clear()
        return original_supervision(dataset, keys, **options)

    monkeypatch.setattr(ICRMMSEnet, "forward", tracked_forward)
    monkeypatch.setattr(icr_noise, "supervision", tracked_supervision)
    run = tmp_path / "comparison"
    report = icr_noise.compare_noise(directory / "test.h5", checkpoint, run, DenseBackend(),
        batch_size=2, frame_limit=1, timing_repeats=2, bootstrap_samples=8)

    assert checked_weights == {"full", "diagonal"}
    assert len(scored_batches) == report["batches"] == 4
    assert report["blocks"] == 4
    assert report["complete"] and report["smoke"]
    assert report["trained_noise_mode"] == "full"
    assert report["timing_repeats"] == 2
    assert report["shared_input_preparation_seconds"] > 0
    assert report["maximum_probability_difference"] < 1e-10
    full, diagonal = report["modes"]["full"], report["modes"]["diagonal"]
    assert full["aggregate"]["total_bits"] == diagonal["aggregate"]["total_bits"] == 64
    assert full["aggregate"]["bit_errors"] == diagonal["aggregate"]["bit_errors"]
    np.testing.assert_allclose(full["aggregate"]["mean_nmse_linear"],
                               diagonal["aggregate"]["mean_nmse_linear"], rtol=1e-12)
    for full_row, diagonal_row in zip(full["per_snr"], diagonal["per_snr"]):
        assert full_row["total_bits"] == diagonal_row["total_bits"] == 32
        np.testing.assert_array_equal(full_row["bit_errors"], diagonal_row["bit_errors"])
        np.testing.assert_allclose(full_row["mean_nmse_linear"], diagonal_row["mean_nmse_linear"], rtol=1e-12)
    for result in (full, diagonal):
        performance = result["performance"]
        repeats = performance["inference_seconds_by_repeat"]
        assert len(repeats) == 2 and all(value > 0 for value in repeats)
        assert performance["median_inference_seconds"] == pytest.approx(np.median(repeats))
        assert performance["preprocessing_and_inference_seconds"] == pytest.approx(
            report["shared_input_preparation_seconds"] + performance["median_inference_seconds"])
        assert performance["end_to_end_seconds_per_block"] == pytest.approx(
            performance["preprocessing_and_inference_seconds"] / report["blocks"])
        assert performance["cold_first_batch_seconds"] > 0
    assert report["differences"]["ber"] == 0
    assert abs(report["differences"]["mean_nmse_db"]) < 1e-10
    assert report["differences"]["inference_speedup"] > 0
    assert report["differences"]["end_to_end_speedup"] > 0
    for name in ("comparison.json", "ber_snr.png", "nmse_snr.png", "runtime.png"):
        assert (run / name).stat().st_size > 0
    assert {path: file_sha256(path) for path in paths} == hashes_before


def test_noise_comparison_rejects_training_or_validation_data(small_comparison, tmp_path):
    directory, checkpoint, _ = small_comparison
    for split in ("train", "validation"):
        with pytest.raises(ValueError, match="independent test split"):
            icr_noise.compare_noise(directory / (split + ".h5"), checkpoint, tmp_path / split,
                DenseBackend(), frame_limit=1, timing_repeats=1, bootstrap_samples=8)


def test_evenly_spaced_frames_keep_raw_inputs_and_supervision_keys_aligned(small_comparison, tmp_path, monkeypatch):
    directory, checkpoint, _ = small_comparison
    original_batches, original_supervision = icr_noise._sample_batches, icr_noise.supervision
    seen_keys = []

    def checked_batches(dataset, indices, **options):
        assert dataset.frame_count == 6
        assert indices == [0, 2, 5]
        for inputs, keys in original_batches(dataset, indices, **options):
            for row, (si, fi, block) in enumerate(keys):
                raw = dataset.model_input(si, fi)
                torch.testing.assert_close(inputs["observations"][row],
                    torch.as_tensor(raw["observations"][block + 1]), atol=0, rtol=0)
                torch.testing.assert_close(inputs["initial_channel"][row],
                    torch.as_tensor(raw["csi"][block]), atol=0, rtol=0)
                assert inputs["rf_variance"][row].item() == raw["noise"]["rf_variance"]
            yield inputs, keys

    def checked_supervision(dataset, keys, **options):
        targets = original_supervision(dataset, keys, **options)
        assert targets["frames"].tolist() == [fi for _, fi, _ in keys]
        for row, (_, fi, block) in enumerate(keys):
            labels = dataset.labels(fi)
            torch.testing.assert_close(targets["symbols"][row], torch.as_tensor(labels["symbols"][block]),
                                       atol=0, rtol=0)
            torch.testing.assert_close(targets["reference_channel"][row],
                torch.as_tensor(dataset.reference_csi(fi)[block]), atol=0, rtol=0)
            window = int(dataset._h["samples/window_index"][fi])
            assert targets["records"][row].item() == int(dataset._h["windows/record"][window])
        seen_keys.extend(keys)
        return targets

    monkeypatch.setattr(icr_noise, "_sample_batches", checked_batches)
    monkeypatch.setattr(icr_noise, "supervision", checked_supervision)
    report = icr_noise.compare_noise(directory / "test.h5", checkpoint, tmp_path / "spaced", DenseBackend(),
        batch_size=2, frame_limit=3, timing_repeats=1, bootstrap_samples=8)

    assert report["sample_selection"] == "evenly_spaced"
    assert report["selected_frame_indices"] == [0, 2, 5]
    assert len(seen_keys) == report["blocks"] == 12
    assert set(seen_keys) == {(si, fi, block) for si in range(2) for fi in (0, 2, 5) for block in range(2)}
    assert report["batches"] == 8
    for mode in report["modes"].values():
        assert mode["aggregate"]["total_bits"] == 192
        for row in mode["per_snr"]:
            assert row["blocks"] == 6 and row["total_bits"] == 96
            assert row["records"] == 3
            assert [record["record"] for record in row["by_record"]] == [3, 4, 5]
