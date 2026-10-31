"""Compact protocol, independent covariance solve, resume and leakage acceptance."""
from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.src.pdd import load_pdd_config
from data_generation.src.dataset.compact_protocol import plan, validate_plan, target_nmse_db
from data_generation.src.dataset.compact_math import reconstruct, coefficient_nmse, matrix_to_taps, projection_power
from data_generation.src.dataset.compact_export import prepare_model, generate_split, generate, calibrate_training
from data_generation.src.dataset.compact_acceptance import accept
from data_generation.src.receiver.dpss import DPSSModel
from data_generation.src.receiver.estimation import FrozenPrior
from data_generation.src.transmitter.ofdm import make_frame, carrier_masks
from src.common.numerics import DenseBackend
from src.data import CompactDataset, data_system
from src.data.uncertainty import CovarianceProxy, blocks_from_prior


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(2):
        yield


def fixture():
    c = load_pdd_config()
    c.update(smoke=True, nfft=14, cp=6, guard_samples=4, fft_advance=1, null_per_edge=0,
             data_count=8, pilot_count=2, delay_length=4, dpss_expected_count=2,
             dpss_half_bandwidth_hz=1, dpss_tail_ratio=.001, delay_block_size=2,
             prior_records=[51], validation_records=[56], starts_seconds=[2], learning_dataset=plan(True))
    rng = np.random.default_rng(92)
    basis, _ = np.linalg.qr(rng.normal(size=(80, 2)))
    model = DPSSModel(basis, 1, .001, 4, c)
    coefficients = rng.normal(size=(5, 2, 4))+1j*rng.normal(size=(5, 2, 4))
    prior = FrozenPrior.fit_coefficients(model, coefficients, 0, 1, [51], delay_block_size=2)
    chain = SimpleNamespace(noise_covariance=np.eye(14)*.7, noise_gain=.7, u=np.zeros((235, 1)),
                            operators=lambda *args: (None, None))

    def frame(sounding, start, tx, *, operators=None):
        taps = np.stack([np.pad(model.reconstruct(coefficients[0]+sounding.record*.01, s), ((0, 0), (0, 10))) for s in range(4)])
        # Keep a real out-of-support component: full CSI must not be projected/truncated.
        taps[:, :, 9] = .01j
        from data_generation.src.receiver.chain import taps_to_matrix
        matrices = np.stack([taps_to_matrix(t) for t in taps])
        y = np.einsum("sij,sj->si", matrices, tx.x)
        y += (sounding.record*.002)*(1+.3j)  # Physical model mismatch, separate from CSI error.
        return y.copy(), y, taps, matrices[[1, 2]], dict(prediction_errors=[0]*4)

    def noise(clean, record, start, split, *, noise_repeat=None):
        rng = np.random.default_rng([c["seed"], split, record, round(start*1000), 1, noise_repeat])
        standard = rng.normal(size=(4, 14))+1j*rng.normal(size=(4, 14))
        power = np.mean(abs(clean[[1, 2]])**2)
        return standard, power/.7/10**(np.array(c["snr_db"])/10), power

    chain.frame, chain.noise = frame, noise
    archive = SimpleNamespace(load=lambda record: SimpleNamespace(record=record, fs_delay=16000, duration=32))
    return c, prior, chain, archive


def test_frozen_formal_counts_targets_and_overlap():
    c = load_pdd_config()
    counts = validate_plan(c)
    assert [counts[s]["d_blocks_all_snr"] for s in ("train", "validation", "test")] == [7350, 1050, 8400]
    assert counts["test"]["bits_per_snr"] == 1228800
    assert [counts[s]["physical_windows"] for s in ("train", "validation", "test")] == [525, 75, 150]
    assert set(sum(c["learning_dataset"]["records"].values(), [])) == set(range(1, 51))
    for channel, endpoints in (("NOF1", [-12.7, -18.3]), ("NCS1", [-6.2, -10.67])):
        values = target_nmse_db(channel, c["snr_db"])
        np.testing.assert_allclose(values[[0, -1]], endpoints)
        assert np.all(np.diff(values) < 0) and np.all(np.diff(np.diff(values)) > 0)
        if channel == "NCS1":
            assert [f"{v:.2f}" for v in values] == ["-6.20", "-8.05", "-9.17", "-9.85", "-10.27", "-10.52", "-10.67"]
    c["learning_dataset"]["records"]["train"].append(51)
    with pytest.raises(ValueError, match="disjoint"):
        validate_plan(c)


def test_independent_payload_and_noise_repeat_seeds():
    from data_generation.src.receiver.chain import ReceiverChain
    c = load_pdd_config()
    a = make_frame(c, 11, 2, 4, payload_repeat=0)
    b = make_frame(c, 11, 2, 4, payload_repeat=1)
    assert not np.array_equal(a.bits, b.bits)
    np.testing.assert_array_equal(a.known, b.known)
    mock = SimpleNamespace(c=c, noise_gain=1, receive=lambda x: x)
    clean = np.ones(6232)
    z0, v0, _ = ReceiverChain.noise(mock, clean, 11, 2, 4, noise_repeat=0)
    z1, _, _ = ReceiverChain.noise(mock, clean, 11, 2, 4, noise_repeat=1)
    assert not np.array_equal(z0, z1)
    z2, _, _ = ReceiverChain.noise(mock, clean, 11, 2, 4, noise_repeat=0)
    np.testing.assert_array_equal(z0, z2)
    np.testing.assert_array_equal(a.bits, make_frame(c, 11, 2, 4, payload_repeat=0).bits)


def test_fourier_reconstruction_and_coherent_projection_residual():
    c, prior, chain, archive = fixture()
    frame = make_frame(c, 1, 2, 2, payload_repeat=0)
    _, _, taps, reference, _ = chain.frame(archive.load(1), 2, frame)
    np.testing.assert_allclose(matrix_to_taps(reference), taps[[1, 2]], atol=1e-14)
    raw = prior.coefficients[:3]
    nmse = coefficient_nmse(prior.model, raw, reference)
    expected = [np.sum(abs(reconstruct(prior.model, coeff)-reference)**2, axis=(1, 2))/np.sum(abs(reference)**2, axis=(1, 2)) for coeff in raw]
    np.testing.assert_allclose(nmse, expected, rtol=1e-13, atol=1e-14)
    projected = reconstruct(prior.model, raw[0])
    residual = reference-projected
    data = np.flatnonzero(carrier_masks(c)["data"])
    # Enumerate unit-energy unknown tones; public pilots contribute as one coherent vector.
    direct = np.stack([sum(abs(residual[d, :, k])**2 for k in data)+abs(residual[d]@frame.known[d+1])**2 for d in range(2)])
    np.testing.assert_allclose(projection_power(reference, projected, frame.known, data), direct)


def test_covariance_proxy_against_explicit_dense_solve():
    from data_generation.tests.test_pdd import small_fixture, dense_contract
    from data_generation.src.receiver.pdd import JointEstimator
    prior, known, masks, noise, y = small_fixture()
    prepared = JointEstimator(prior, known, masks, noise, DenseBackend())
    covariance, cross, rows = dense_contract(prior, known, masks)
    from scipy.linalg import block_diag
    selected_noise = block_diag(noise, noise, noise, noise)[np.ix_(rows, rows)]
    q, length = prior.model.count, prior.model.length
    p = np.zeros((q, length, q, length), complex)
    for s, temporal, delay in prior.separable_blocks():
        p[:, s, :, s] = np.einsum("qr,lk->qlrk", temporal, delay)
    p = p.reshape(q*length, q*length)
    variance = .3
    dense = p-cross@np.linalg.solve(covariance+(variance+prepared.ridge)*selected_noise, cross.conj().T)
    proxy = CovarianceProxy(blocks_from_prior(prior), prepared.gain, prepared.values, prepared.ridge,
                            variance, (q, length), scale=.7)
    actual = proxy.matmul(np.eye(q*length))
    np.testing.assert_allclose(actual, .7*dense, atol=5e-13)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-13)
    assert np.linalg.eigvalsh(actual).min() >= -1e-12
    np.testing.assert_allclose(proxy.trace(), np.trace(actual).real, atol=2e-12)
    values, modes = proxy.modes(2)
    np.testing.assert_allclose(actual@modes, modes*values, atol=1e-7)


def test_complete_compact_generation_acceptance_and_resume(tmp_path):
    c, prior, chain, archive = fixture()
    provenance = dict(commit="test", archive_sha256="test")
    run = tmp_path/"run"
    run.mkdir()
    resumed, fresh = tmp_path/"resumed", tmp_path/"fresh"
    resumed.mkdir()
    prepared, identity = prepare_model(resumed/"model.h5", c, chain, prior, DenseBackend(), provenance)
    assert not generate_split(resumed/"train.h5", run, c, "train", archive, chain, prepared, identity, stop_after=1)
    with pytest.raises(ValueError, match="incomplete"):
        CompactDataset(resumed/"train.h5")
    generate(resumed, run, c, archive, chain, prior, DenseBackend(), provenance)
    generate(fresh, run, c, archive, chain, prior, DenseBackend(), provenance)
    for name in ("model.h5", "train.h5", "validation.h5", "test.h5"):
        with h5py.File(resumed/name, "r") as a, h5py.File(fresh/name, "r") as b:
            names = []
            a.visititems(lambda key, obj: names.append(key) if isinstance(obj, h5py.Dataset) else None)
            for key in names:
                np.testing.assert_array_equal(a[key][:], b[key][:], err_msg=key)
    report = accept(resumed, run, c, plots=False)
    assert report["complete"] and all(report["checks"].values())
    assert max(s["maximum_target_error_db"] for s in report["splits"].values()) < 1e-8
    with CompactDataset(resumed/"test.h5", input_view="compensated") as ds:
        inputs, labels = ds[0]
        assert set(inputs) == {"input_view", "observations", "csi", "noise", "known", "masks", "data_indices", "pilot_indices", "uncertainty"}
        assert set(labels) == {"bits", "symbols", "classes", "constellation", "constellation_bits"}
        assert ds.audit(0, 0)["identity"]["payload_repeat"] == 0
        proxy = inputs["uncertainty"]["coefficient_covariance"]
        assert proxy.trace() > 0
        uncertainty = inputs["uncertainty"]
        assert uncertainty["sampled_dpss_basis"].shape == (2, 14, 2)
        eigenvalues, modes = proxy.modes(1)
        from data_generation.src.receiver.chain import taps_to_matrix
        taps = np.einsum("dnq,ql->dnl", uncertainty["sampled_dpss_basis"], modes[:, 0].reshape(uncertainty["coefficient_shape"]))
        np.testing.assert_allclose(np.stack([taps_to_matrix(t) for t in taps]), reconstruct(ds._model, modes[:, 0].reshape(2, 4)))
        assert data_system(inputs)[1].shape == (2, 14, 8)
        for si in range(1, len(c["snr_db"])):
            np.testing.assert_array_equal(ds.labels(0)["bits"], labels["bits"])
            # Filtered standard noise is the same vector at each SNR.
            observed = ds.model_input(si, 0)
            clean = ds._h["samples/clean_observations"][0]
            np.testing.assert_allclose((observed["observations"]-clean)/np.sqrt(observed["noise"]["rf_variance"]),
                                       (inputs["observations"]-clean)/np.sqrt(inputs["noise"]["rf_variance"]), atol=1e-13)
    with pytest.raises(ValueError, match="training data"):
        calibrate_training(resumed/"model.h5", resumed/"test.h5", c)
    with h5py.File(resumed/"model.h5", "r+") as h:
        h["training_calibration"].attrs["source_split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        accept(resumed, run, c, plots=False)


def test_physical_invariance_corruption_is_rejected(tmp_path):
    c, prior, chain, archive = fixture()
    run = tmp_path/"run"
    run.mkdir()
    generate(tmp_path/"data", run, c, archive, chain, prior, DenseBackend(), {"commit": "test"})
    with h5py.File(tmp_path/"data/train.h5", "r+") as h:
        h["samples/clean_observations"][0, 1, 0] += .1
    with pytest.raises(ValueError, match="changed during compensation"):
        accept(tmp_path/"data", run, c, plots=False)


def test_retarget_reuses_raw_arrays_and_updates_csi_and_uncertainty(tmp_path, monkeypatch):
    from data_generation.src.dataset.compact_protocol import TARGET_PARAMETERS
    from data_generation.src.dataset.compact_retarget import retarget
    from src.common.experiment import file_sha256

    c, prior, chain, archive = fixture()
    source, destination = tmp_path/"original", tmp_path/"updated"
    original_run, updated_run = tmp_path/"original_run", tmp_path/"updated_run"
    original_run.mkdir()
    with monkeypatch.context() as old_targets:
        old_targets.setitem(TARGET_PARAMETERS, "NCS1", (-6.8, 4.6))
        generate(source, original_run, c, archive, chain, prior, DenseBackend(), {"commit": "original"})
        assert accept(source, original_run, c, plots=False)["complete"]
    hashes = {p.name: file_sha256(p) for p in source.glob("*.h5")}
    with CompactDataset(source/"test.h5", input_view="compensated") as dataset:
        before = dataset.model_input(0, 0)
        reference = dataset.audit(0, 0)["reference_csi"]
        before_alpha = float(dataset._h["compensation/alpha"][0])
        before_covariance = before["uncertainty"]["coefficient_covariance"].matmul(np.eye(8))

    report = retarget(source, destination, updated_run, {"commit": "retarget"}, plots=False)
    assert report["complete"] and all(report["checks"].values())
    assert report["nmse_retarget"]["generation_provenance"]["commit"] == "original"
    assert not report["nmse_retarget"]["raw_data_regenerated"]
    for split in ("train", "validation", "test"):
        np.testing.assert_allclose(report["splits"][split]["mean_nmse_db"], target_nmse_db("NCS1", c["snr_db"]), atol=1e-10)
    for name, expected_hash in hashes.items():
        assert file_sha256(source/name) == expected_hash
        with h5py.File(source/name, "r") as original, h5py.File(destination/name, "r") as updated:
            names = []
            original.visititems(lambda key, obj: names.append(key) if isinstance(obj, h5py.Dataset) else None)
            for key in names:
                if not key.startswith("compensation/"):
                    np.testing.assert_array_equal(original[key][:], updated[key][:], err_msg=f"{name}: {key}")
    with CompactDataset(destination/"test.h5", input_view="compensated") as dataset:
        after = dataset.model_input(0, 0)
        ratio = float(dataset._h["compensation/alpha"][0])/before_alpha
        assert ratio > 1
        np.testing.assert_allclose(after["csi"]-reference, ratio*(before["csi"]-reference), atol=1e-13)
        np.testing.assert_array_equal(after["observations"], before["observations"])
        np.testing.assert_allclose(after["uncertainty"]["coefficient_covariance"].matmul(np.eye(8)), ratio**2*before_covariance)
        np.testing.assert_allclose(after["uncertainty"]["csi_residual_power"], ratio**2*before["uncertainty"]["csi_residual_power"])
        np.testing.assert_array_equal(after["uncertainty"]["model_residual_power"], before["uncertainty"]["model_residual_power"])
    # Targets saved in the file must agree with reconstructed CSI, not just its plot.
    with h5py.File(destination/"test.h5", "r+") as h:
        h["compensation/target_nmse_db"][0] += .1
    with pytest.raises(ValueError, match="Persisted NMSE target mismatch"):
        accept(destination, updated_run, c, plots=False)


def test_official_downloader_range_checks_and_resume(tmp_path, monkeypatch):
    from io import BytesIO
    import zipfile
    import data_generation.scripts.fetch_watermark as downloader
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        for channel in ("NCS1", "NOF1"):
            for number in range(1, 61):
                z.writestr(f"Watermark/input/channels/{channel}/mat/{channel}_{number:03d}.mat", b"crc fixture")
    payload = archive.getvalue()

    class Response(BytesIO):
        def __init__(self, request):
            if request.get_method() == "HEAD":
                super().__init__(b"")
                self.status = 200
                self.headers = {"Content-Length": str(len(payload))}
            else:
                begin, end = map(int, request.get_header("Range").split("=")[1].split("-"))
                super().__init__(payload[begin:end+1])
                self.status = 206
                self.headers = {"Content-Range": f"bytes {begin}-{end}/{len(payload)}"}

    monkeypatch.setattr(downloader.urllib.request, "urlopen", lambda request, **kwargs: Response(request))
    target = tmp_path/"WatermarkV1.zip"
    target.with_suffix(".zip.partial").write_bytes(payload[:4096])
    result = downloader.fetch(target, workers=3, chunk_bytes=1024)
    assert target.read_bytes() == payload
    assert {k: len(v) for k, v in result["records"].items()} == {"NCS1": 60, "NOF1": 60}
    assert not target.with_suffix(".zip.parts").exists()
    assert downloader.fetch(target)["archive_sha256"] == result["archive_sha256"]


def test_spawn_workers_match_serial_real_receiver(tmp_path):
    from io import BytesIO
    import zipfile
    from scipy.io import savemat
    from data_generation.src.channel.watermark import WatermarkArchive
    from data_generation.src.receiver.chain import ReceiverChain
    c, prior, _, _ = fixture()
    c["learning_dataset"]["starts_seconds"] = [.01, .03]
    archive_path = tmp_path/"synthetic.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        for record in (1, 2, 3):
            taps = np.zeros((16, 64), complex)
            taps[:, 0] = 1+record*.01
            taps[:, 40] = .2j
            data = BytesIO()
            savemat(data, dict(h=taps, fs_t=250, fs_tau=16000, fc=14000, V0=0))
            z.writestr(f"Watermark/input/channels/NCS1/mat/NCS1_{record:03d}.mat", data.getvalue())
    archive = WatermarkArchive(archive_path)
    chain = ReceiverChain(c)
    run = tmp_path/"run"
    run.mkdir()
    for workers in (1, 2):
        generate(tmp_path/f"workers_{workers}", run, c, archive, chain, prior, DenseBackend(),
                 {"commit": "parallel fixture"}, workers=workers, threads=2)
    for name in ("model.h5", "train.h5", "validation.h5", "test.h5"):
        with h5py.File(tmp_path/"workers_1"/name, "r") as a, h5py.File(tmp_path/"workers_2"/name, "r") as b:
            keys = []
            a.visititems(lambda key, obj: keys.append(key) if isinstance(obj, h5py.Dataset) else None)
            for key in keys:
                np.testing.assert_array_equal(a[key][:], b[key][:], err_msg=key)
    assert accept(tmp_path/"workers_2", run, c, plots=False)["complete"]
