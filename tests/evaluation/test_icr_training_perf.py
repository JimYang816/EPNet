"""Small CPU integration plus memory-constrained batch selection."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits

from data_generation.tests.test_compact import fixture
from data_generation.src.dataset.compact_export import generate
from src.common.numerics import DenseBackend
from src.evaluation.icr_training_perf import benchmark_training, choose_batch


@pytest.mark.parametrize("backend", ("dpss", "dense_device"))
def test_small_cpu_benchmark_validates_math_and_measures_real_updates(tmp_path, backend):
    torch.set_num_threads(2)
    with threadpool_limits(2):
        config, prior, chain, archive = fixture()
        config["snr_db"] = [0, 10]
        generation = tmp_path/"generation"
        generation.mkdir()
        generate(tmp_path/"data", generation, config, archive, chain, prior, DenseBackend(), {"commit": "perf-test"})
        run = tmp_path/"performance"
        report = benchmark_training(tmp_path/"data", run,
            model_config=dict(layers=2, bandwidth=2, window_size=8, center_size=4, noise_mode="diagonal"),
            device="cpu", batch_sizes=(2,), stages=("C",), frame_limit=2, validation_frames=2,
            repeats=1, threads=2, reference_cache_mib=2, ablations=True, operator_backend=backend)
    assert report["complete"] and report["equivalence"]["passed"]
    assert report["equivalence"]["bit_disagreements"] == 0
    assert {row["mode"] for row in report["trials"]} == ({
        "dense_reference", "dpss_full_reference", "compact", "prefetch"} if backend == "dpss" else {
        "dense_reference", "dense_cached", "dense_device", "prefetch"})
    trial = report["trials"][-1]
    assert trial["blocks_per_repeat"] == 2*len(trial["selected_frame_indices"])
    # The tiny fixture has a tail batch; it must be measured, never dropped.
    assert trial["blocks_per_repeat"] > 0
    assert trial["optimizer_updates"] and not trial["checkpoint_saved"]
    assert trial["blocks_per_second"] > 0
    assert trial["reference_cache_hits"] > 0
    assert report["selection"]["batch_size"] == 2
    assert not report["test_access"]
    for name in ("comparison.json", "equivalence.json", "ber_snr.png", "nmse_snr.png", "batch_speed_memory.png"):
        assert (run/name).stat().st_size > 0


def test_batch_selection_prefers_small_near_fastest_with_memory_headroom():
    trials = []
    for size, seconds, memory in ((2, 1., 1), (4, .7, 2), (8, .68, 4), (16, .4, 15)):
        for stage in "ABC":
            trials.append(dict(mode="prefetch", stage=stage, status="ok", batch_size=size,
                median_seconds=seconds, cuda_total_bytes=16*2**30, peak_cuda_reserved_bytes=memory*2**30,
                monitor=dict(sampled_device_peak_mib=memory*1024)))
    selected = choose_batch(trials, "ABC")
    assert selected["batch_size"] == 4
    assert 16 not in [row["batch_size"] for row in selected["candidates"]]
    with pytest.raises(RuntimeError, match="No tested batch"):
        choose_batch([dict(trials[0], status="oom")], "ABC")
