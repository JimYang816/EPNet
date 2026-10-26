from __future__ import annotations

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from data_generation.src.config import load_config
from data_generation.src.transmitter.ofdm import carrier_masks, make_frame, qpsk, decisions, ifft, fft, overhead, receive_symbols
from data_generation.src.channel.watermark import Sounding, direct_convolution
from data_generation.src.receiver.chain import (ReceiverChain, resample, resample_operator, taps_to_matrix,
                                                relative_energy, full_frame_kernels)


@pytest.fixture(autouse=True)
def limit_threads():
    with threadpool_limits(4):
        yield


def test_layout_and_qpsk():
    c = load_config()
    bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.uint8)
    np.testing.assert_array_equal(decisions(qpsk(bits)), bits)
    np.testing.assert_allclose(abs(qpsk(bits)), 1)
    m = carrier_masks(c)
    assert [m[k].sum() for k in ("active", "pilot", "data", "null")] == [896, 128, 512, 384]
    np.testing.assert_array_equal(np.flatnonzero(m["pilot"]), 67+np.arange(128)*7)
    f = make_frame(c, 1, 2)
    assert f.baseband.shape == (6232,)
    u = f.baseband[300:-300].reshape(4, 1408)
    np.testing.assert_array_equal(u[:, :384], u[:, -384:])
    np.testing.assert_allclose(fft(u[:, 384:]), f.x, atol=2e-15)
    np.testing.assert_array_equal(make_frame(c, 1, 2).x, f.x)
    np.testing.assert_array_equal(make_frame(c, 2, 8).known, f.known)
    assert f.bits.size == 2048
    assert overhead(c)["nondata_grid_fraction"] == pytest.approx(3/4)
    assert overhead(c)["useful_passband_samples"] == 16384
    assert overhead(c)["cp_passband_samples"] == 6144


@pytest.mark.parametrize("up,down", [(16, 1), (1, 6), (6, 1), (1, 16), (8, 3)])
def test_independent_polyphase_operator(up, down):
    c = load_config()
    rng = np.random.default_rng(9)
    x = rng.normal(size=73)+1j*rng.normal(size=73)
    np.testing.assert_allclose(resample_operator(len(x), up, down, c)@x, resample(x, up, down, c), atol=2e-15)


def test_official_interpolation_loop_and_boundaries():
    h = np.array([[1, 2j, 0, 0], [2, 1j, 0, 0], [3, 0, 0, 0], [4, -1j, 0, 0]], complex)
    s = Sounding(h, 2, 8, 14000, 0, "fixture")
    x = np.arange(11)+1j*np.arange(11)[::-1]
    np.testing.assert_allclose(s.operator(len(x), 0)@x, direct_convolution(x, s, 0))
    with pytest.raises(ValueError, match="boundary"):
        s.interpolate([s.duration])


def test_full_static_and_time_varying_matrices():
    n = 64
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)+1j*rng.normal(size=n)
    g = np.zeros((n, 20), complex)
    g[:, 0], g[:, 17] = 1, .2j
    h = taps_to_matrix(g)
    np.testing.assert_allclose(h@x, (1+.2j*np.exp(-2j*np.pi*np.arange(-n//2, n//2)*17/n))*x, atol=2e-14)
    g[:, 0] *= np.exp(2j*np.pi*.41*np.arange(n)/n)
    h = taps_to_matrix(g)
    np.testing.assert_allclose(h@x, fft(ifft(x)*g[:, 0]+.2j*np.roll(ifft(x), 17)), atol=2e-14)
    assert abs(h[10, 30]) > 1e-4


def test_physical_passband_and_noise_statistics():
    c = load_config()
    chain = ReceiverChain(c)
    # Synthetic multipath varying within each OFDM symbol, not a frozen symbol.
    time = np.arange(128)/31.372549019607842
    h = np.zeros((128, 510), complex)
    h[:, 0] = np.exp(2j*np.pi*.7*time)
    h[:, 120] = .2j*np.exp(-2j*np.pi*.4*time)
    s = Sounding(h, 31.372549019607842, 16000, 14000, 0, "fixture")
    f = make_frame(c, 1, 2)
    clean, y, taps, ref, diagnostics = chain.frame(s, 2, f)
    a, _ = chain.operators(s, 2-c["guard_samples"]/c["fs_baseband"])
    for length in (256, 384):
        full = full_frame_kernels(a, c, length)
        sampled = np.stack([full[j*1408+384-32+np.arange(1024)] for j in range(4)])
        np.testing.assert_allclose(sampled, taps[..., :length], rtol=1e-12, atol=1e-14)
    # The largest support crosses the beginning of the finite input operator.
    np.testing.assert_array_equal(full[0, 333:], 0)
    assert max(diagnostics["prediction_errors"]) < 1e-6
    for d, symbol in enumerate((1, 2)):
        assert relative_energy(y[symbol]-ref[d]@f.x[symbol], y[symbol]) < 1e-6
    assert np.sum(abs(taps[..., 256:])**2)/np.sum(abs(taps)**2) < 1e-4
    noise, variances, power = chain.noise(clean, 1, 2)
    np.testing.assert_allclose(power/(chain.noise_gain*variances), 10**(np.array(c["snr_db"])/10))
    rng = np.random.default_rng(11)
    # Independent repeats verify both power and selected off-diagonal entries.
    z = np.stack([chain.receive(rng.normal(size=len(clean)))[200:4400] for _ in range(32)])
    measured = np.mean(abs(z)**2)
    assert abs(measured/chain.noise_gain-1) < .02
    frequency = np.fft.ifftshift(chain.noise_covariance, axes=(0, 1))
    ncov = np.fft.ifft(np.fft.fft(frequency, axis=1, norm="ortho"), axis=0, norm="ortho")
    for lag in (1, 2, 5):
        value = np.mean(z[:, lag:]*z[:, :-lag].conj())
        assert abs(value-ncov[lag, 0])/chain.noise_gain < .012


def test_diagnostic_policy_keeps_isi_but_rejects_incorrect_replay():
    c = load_config()
    c["model_mismatch_policy"] = "reject"
    c.update(nfft=14, cp=6, fft_advance=1, guard_samples=6, null_per_edge=0, data_count=8, pilot_count=2)
    h = np.zeros((32, 64), complex)
    h[:, 0], h[:, 40] = 1, .2j  # Late path exceeds this fixture's CP.
    sounding = Sounding(h, 250, 16000, 14000, 0, "late-path fixture")
    chain = ReceiverChain(c)
    frame = make_frame(c, 1, .01)
    with pytest.raises(RuntimeError, match="Physical link acceptance"):
        chain.frame(sounding, .01, frame)
    c["model_mismatch_policy"] = "diagnose"
    clean, y, taps, reference, diagnostic = chain.frame(sounding, .01, frame)
    assert diagnostic["model_mismatch_retained"] and not diagnostic["circular_model_passed"]
    assert diagnostic["waveform_operator_error"] < 1e-20
    np.testing.assert_array_equal(receive_symbols(chain.receive(clean), c), y)
    assert relative_energy(y[1]-reference[0]@frame.x[1], y[1]) > 1e-6
    original_replay = chain.replay
    chain.replay = lambda *args: original_replay(*args)+1
    with pytest.raises(RuntimeError, match="Physical link acceptance"):
        chain.frame(sounding, .01, frame)
