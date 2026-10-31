"""Frozen budget comparisons, label isolation and physical-record statistics."""
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from src.evaluation.icr_budgeted import paired, reproduction_audit


def test_historical_replay_records_small_roundoff_and_detects_large_drift():
    small = reproduction_audit([25000], [25002], [1228800])
    assert small["within_tolerance"] and not small["exact"]
    assert small["bit_error_difference"] == [2]
    assert not reproduction_audit([25000], [25013], [1228800])["within_tolerance"]
    assert reproduction_audit([25000], [25000], [1228800])["exact"]


def test_loaded_lmmse_control_matches_independent_normal_equations():
    import math
    from src.algorithms.icr_mmse.model import ICRMMSEnet
    from src.evaluation.icr_budgeted import LoadedLMMSEControl
    from tests.algorithms.test_icr_mmse import inputs

    item = inputs(11, 2, torch.complex128)
    item["unit_covariance"] = torch.eye(11, dtype=torch.complex128)
    model = ICRMMSEnet(layers=1, bandwidth=1, detector_mode="lmmse", correction_mode="fixed",
                       noise_mode="diagonal", precision="complex128")
    model.equalizers[0] = LoadedLMMSEControl(torch.tensor(math.log(math.expm1(.05)), dtype=torch.float64))
    output = model(**item)
    h = item["initial_channel"]
    hd = h[..., item["data_mask"]]
    noise = item["rf_variance"] + .05*hd.abs().square().sum(-2).mean(-1)
    y = item["observations"] - (h@item["known"][..., None]).squeeze(-1)
    system = hd.mH@hd + noise[:, None, None]*torch.eye(hd.shape[-1])
    expected = torch.linalg.solve(system, hd.mH@y[..., None]).squeeze(-1)
    torch.testing.assert_close(output.diagnostics[0]["linear_estimate"], expected, rtol=1e-10, atol=1e-10)


def test_paired_statistics_keep_each_physical_record_whole_across_snr():
    a, b = [], []
    for snr in (0, 10):
        a.append(dict(snr_db=snr, by_record=[dict(record=7, bits=100, bit_errors=[3]),
                                            dict(record=9, bits=100, bit_errors=[5])]))
        b.append(dict(snr_db=snr, by_record=[dict(record=7, bits=100, bit_errors=[6]),
                                            dict(record=9, bits=100, bit_errors=[7])]))
    result = paired(a, b)
    assert result["overall"]["records"] == 2
    assert result["overall"]["estimate"] == -.025
    b[0]["by_record"][0]["bits"] += 1
    with pytest.raises(ValueError, match="bit counts"):
        paired(a, b)


def test_comparison_predicts_before_labels_and_preserves_shared_fixed_csi(tmp_path, monkeypatch):
    from data_generation.tests.test_compact import fixture
    from data_generation.src.dataset.compact_export import generate
    from src.algorithms.icr_mmse.checkpoint import SCHEMA, FEEDBACK_LAYOUT, INFERENCE_CODE, atomic_save, cpu_state
    from src.algorithms.icr_mmse.model import ICRMMSEnet
    from src.common.numerics import DenseBackend
    from src.data.compact import CompactDataset
    from src.evaluation.fingerprints import source_fingerprint, DataIdentity
    from src.evaluation import icr_budgeted as scoring

    torch.set_num_threads(1)
    with threadpool_limits(1):
        c, prior, chain, archive = fixture()
        c["snr_db"] = [0, 10]
        directory, generation = tmp_path/"data", tmp_path/"generation"
        generation.mkdir()
        generate(directory, generation, c, archive, chain, prior, DenseBackend(), {"commit": "synthetic"})
        model = ICRMMSEnet(layers=2, bandwidth=2, window_size=6, center_size=3, hidden_size=4,
                          detector_mode="budgeted", correction_mode="fixed", noise_mode="diagonal",
                          initial_soft_damping=.8)
        checkpoint = tmp_path/"model.pt"
        with CompactDataset(directory/"train.h5") as dataset:
            atomic_save(checkpoint, dict(schema=SCHEMA, input_view="compensated", model_config=model.config,
                identity=dict(data=DataIdentity({name: directory/name
                    for name in ("train.h5", "validation.h5", "model.h5")}).files),
                feedback_layout=FEEDBACK_LAYOUT, inference_code=source_fingerprint(INFERENCE_CODE),
                model=cpu_state(model), seed=20261028, training_scope="one_channel_all_snr_D1_D2",
                channel=dataset.channel, data_schema=str(dataset._h.attrs["schema"]),
                model_identity=str(dataset._h.attrs["model_identity"]),
                data_indices=dataset._indices.tolist(), snr_db=dataset.snr_db.tolist()))
        calls = [0]
        forward, supervise = ICRMMSEnet.forward, scoring.supervision
        legacy_forward = scoring.LegacyICRMMSEnet.forward

        def tracked_forward(*args, **kwargs):
            calls[0] += 1
            return forward(*args, **kwargs)

        def tracked_supervision(*args, **kwargs):
            assert calls[0] >= 5
            calls[0] = 0
            return supervise(*args, **kwargs)

        def tracked_legacy_forward(*args, **kwargs):
            calls[0] += 1
            return legacy_forward(*args, **kwargs)

        monkeypatch.setattr(ICRMMSEnet, "forward", tracked_forward)
        monkeypatch.setattr(scoring.LegacyICRMMSEnet, "forward", tracked_legacy_forward)
        monkeypatch.setattr(scoring, "supervision", tracked_supervision)
        result = scoring.compare(directory/"validation.h5", checkpoint, tmp_path/"evaluation",
                                 DenseBackend(), batch_size=2, cache_mib=16)
        assert result["complete"] and not result["test_access"]
        rows = result["models"]["lmmse"]["per_snr"]
        for item in result["models"].values():
            for ref, row in zip(rows, item["per_snr"]):
                np.testing.assert_allclose(row["mean_nmse_linear"], ref["mean_nmse_linear"][0])
        assert (tmp_path/"evaluation/ber_snr.png").exists()
        assert (tmp_path/"evaluation/nmse_snr.png").exists()
        with pytest.raises(FileExistsError):
            scoring.compare(directory/"validation.h5", checkpoint, tmp_path/"evaluation", DenseBackend())
