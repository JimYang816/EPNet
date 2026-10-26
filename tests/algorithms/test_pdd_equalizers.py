import numpy as np

from src.algorithms.MMSE.single_tap import mmse
from src.evaluation.pdd import ep_inputs


def test_single_tap_analytic_raw_observation_and_zero_denominator():
    h = np.array([[1+2j, 8, 1j], [2, 0, 5], [4, 7, .2j]], complex)
    y = np.array([3-1j, 100, -5j])
    mask = np.array([True, False, True])
    variance = np.array([.4, 0, .2])
    expected = np.array([(1-2j)*(3-1j)/(5+.4), (-.2j)*(-5j)/(.04+.2)])
    np.testing.assert_allclose(mmse(h, y, mask, variance), expected)
    np.testing.assert_array_equal(mmse(np.zeros((3, 3)), y, mask, 0), 0)
    # Pilot ICI is deliberately present in raw y; only block EP subtracts it.
    known = np.array([0, 2j, 0])
    columns, residual = ep_inputs(h, y, known, mask)
    assert columns.shape == (3, 2)
    np.testing.assert_array_equal(columns, h[:, mask])
    np.testing.assert_array_equal(residual, y-h@known)
    assert abs(columns[1, 0]) > 0 and abs(columns[1, 1]) > 0
