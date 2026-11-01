import numpy as np

from scripts.evaluate.djk_channel_comparison import (
    FS, multipath_metrics, replay_tvir, shape_correlation,
)


def test_interpolated_tvir_replay_matches_direct_time_domain_sum():
    rng = np.random.default_rng(7)
    h = rng.normal(size=(8, 13))+1j*rng.normal(size=(8, 13))
    x = rng.normal(size=80)+1j*rng.normal(size=80)
    count, start, fs_time = 92, .0013, 600.
    actual = replay_tvir(h, fs_time, x, start, count)
    expected = np.zeros(count, complex)
    for n in range(count):
        position = (start+n/FS)*fs_time
        k = int(position); alpha = position-k
        taps = (1-alpha)*h[k]+alpha*h[k+1]
        for delay in range(len(taps)):
            if 0 <= n-delay < len(x):
                expected[n] += taps[delay]*x[n-delay]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_shape_metric_removes_only_common_complex_gain():
    reference = np.array([1., .5j, .2])
    gains = np.array([1, -2j, .3*np.exp(.7j)])
    h = (gains[:, None]*reference)[None]
    np.testing.assert_allclose(shape_correlation(h, 1, np.array([0., 2., 3.])), 1.)
    changed = h.copy(); changed[0,1,1] *= -1
    assert np.max(shape_correlation(changed, 1, np.array([0., 2., 3.]))) < .9


def test_two_path_delay_and_energy_have_known_physical_values():
    h = np.sqrt(np.array([.8, .2]))[None, None]
    metrics, _ = multipath_metrics(h, np.array([0., 10.]), cutoff_db=-30)
    np.testing.assert_allclose(metrics["windowed_rms_delay_ms"], 4.)
    np.testing.assert_allclose(metrics["energy_outside_main_1ms_fraction"], .2)


def test_common_timing_shift_is_not_mistaken_for_channel_shape_change():
    delay = np.arange(-5., 25.001, .0625)
    reference = np.exp(-delay**2/.03)+.4j*np.exp(-(delay-4)**2/.05)
    shifted = np.roll(reference, 8)*np.exp(.6j)*2
    h = np.stack([reference, shifted])[None]
    assert shape_correlation(h, 1, delay)[0, 0] < .1
    np.testing.assert_allclose(shape_correlation(h, 1, delay, delay_search_ms=2), 1., atol=1e-10)
