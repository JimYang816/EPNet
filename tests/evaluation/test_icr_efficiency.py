"""Isolated peak-memory and end-to-end timing on compensated receiver inputs."""
import json

import h5py
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from threadpoolctl import threadpool_limits


def test_all_four_detectors_use_fresh_processes_and_no_test_truth(tmp_path):
    from data_generation.tests.test_compact import fixture
    from data_generation.src.dataset.compact_export import generate
    from src.algorithms.icr_mmse import checkpoint as icr_checkpoint
    from src.algorithms.icr_mmse.model import ICRMMSEnet
    from src.algorithms.udnet import checkpoint as ud_checkpoint
    from src.algorithms.udnet.model import UDNet
    from src.common.experiment import write_json
    from src.common.numerics import DenseBackend
    from src.data.compact import CompactDataset
    from src.data.udnet import PREPROCESSING
    from src.evaluation.comparison import evaluate
    from src.evaluation.fingerprints import source_fingerprint
    from src.evaluation.icr_efficiency import benchmark

    torch.set_num_threads(1)
    with threadpool_limits(1):
        c, prior, chain, archive = fixture()
        c["snr_db"] = [15]
        c["learning_dataset"]["repeats"]["train"] = 4
        directory, generation = tmp_path/"data", tmp_path/"generation"
        generation.mkdir()
        generate(directory, generation, c, archive, chain, prior, DenseBackend(), {"commit": "synthetic"})
        report = evaluate(directory/"test.h5", tmp_path/"baseline", DenseBackend(),
                          batch_size=1, cache_dir=tmp_path/"cache")
        comparison = tmp_path/"baseline.json"
        write_json(comparison, report)
        icr, ud = ICRMMSEnet(layers=2, bandwidth=2), UDNet()
        with CompactDataset(directory/"train.h5") as dataset:
            common = dict(input_view="compensated", channel=dataset.channel,
                model_identity=str(dataset._h.attrs["model_identity"]), data_schema=str(dataset._h.attrs["schema"]),
                data_indices=dataset._indices.tolist(), snr_db=dataset.snr_db.tolist(),
                training_scope="one_channel_all_snr_D1_D2")
            icr_path, ud_path = tmp_path/"icr.pt", tmp_path/"ud.pt"
            torch.save(dict(common, schema=icr_checkpoint.SCHEMA, model_config=icr.config,
                feedback_layout=icr_checkpoint.FEEDBACK_LAYOUT,
                model=icr_checkpoint.cpu_state(icr), inference_code=source_fingerprint(icr_checkpoint.INFERENCE_CODE)), icr_path)
            torch.save(dict(common, schema=ud_checkpoint.SCHEMA, paper_doi=ud_checkpoint.PAPER_DOI,
                preprocessing=PREPROCESSING, model_config=ud.config, model=ud_checkpoint.cpu_state(ud),
                inference_code=source_fingerprint(ud_checkpoint.INFERENCE_CODE)), ud_path)
        # Test labels may change without being read by the timing workers.
        # Worker read guards independently reject any attempted field access.
        with h5py.File(directory/"test.h5", "r+") as test:
            test["samples/bits"][:] = 1-test["samples/bits"][:]
        (directory/"train.h5").rename(directory/"unavailable-training.h5")
        result = benchmark(directory/"test.h5", comparison, icr_path, ud_path,
                           output=tmp_path/"efficiency.json", batch_size=1, frame_limit=1, threads=1)
        rows = result["algorithms"]
        assert len(rows) == len({row["process_id"] for row in rows}) == 4
        assert {row["algorithm"] for row in rows} == {"single_tap_mmse", "full_matrix_lmmse", "udnet", "icr_mmse"}
        assert len({row["blocks"] for row in rows}) == 1
        assert all(row["blocks"] == 2 for row in rows)
        assert all(row["setup_seconds"] > 0 and row["reconstruction_and_all_iterations_seconds"] > 0
                   and row["peak_process_rss_bytes"] > 0 for row in rows)
        assert all(row["parameters"] == 0 for row in rows[:2])
        assert all(row["parameters"] > 0 for row in rows[2:])
        assert rows[-1]["numerical_environment"]["precision"] == "complex64/float32"
        assert rows[-2]["numerical_environment"]["precision"] == "complex128/float64_to_float32"
        assert not result["test_labels_read"]
        assert result["reference_assisted_csi_construction"]
        assert result["initial_provenance_validation_seconds"] > 0
        assert "excluded from setup/amortized runtime" in result["provenance_timing"]
        assert "train.h5" not in result["data_identity"]
        for row in rows:
            assert row["setup_and_prediction_seconds"] == pytest.approx(
                row["setup_seconds"] + row["reconstruction_and_all_iterations_seconds"])
        assert json.loads((tmp_path/"efficiency.json").read_text(encoding="utf-8"))["input_view"] == "compensated"
        wrong = dict(report, input_view="raw")
        write_json(comparison, wrong)
        with pytest.raises(ValueError, match="compensated CSI comparison report"):
            benchmark(directory/"test.h5", comparison, icr_path, ud_path, algorithms=("full_matrix_lmmse",),
                      batch_size=1, frame_limit=1, threads=1)


@pytest.mark.parametrize("identifier", ("ep", "robust_ep"))
def test_retired_detectors_are_rejected_before_loading_inputs(identifier):
    from src.evaluation.icr_efficiency import benchmark, _worker
    with pytest.raises(ValueError, match="Invalid efficiency benchmark controls"):
        benchmark("missing.h5", "missing.json", "missing-icr.pt", "missing-ud.pt", algorithms=(identifier,))
    with pytest.raises(ValueError, match="Unknown efficiency algorithm"):
        _worker(dict(algorithm=identifier, device="cpu"))
