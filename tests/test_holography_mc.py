"""Numerical propagation checks: python3 -m unittest discover -s tests -p test_holography_mc.py."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holography_mc import (average_grid, grid_layout, calibrate_phase,
                           correct_aperture_phase, remove_phase_tilt,
                           run_monte_carlo, noise_sigmas)


class MonteCarloTests(unittest.TestCase):
    def setUp(self):
        axis = np.array([-1., 0., 1.])
        az = np.array([-1., 0., 0., 1., -1., 1.])
        el = np.array([-1., -1., -1., -1., 1., 1.])
        self.layout = grid_layout(az, el, axis, axis)
        self.x, self.y = np.meshgrid(axis, axis)
        self.mask = np.ones((3, 3), dtype=bool)
        self.times = np.arange(6.)
        self.on = np.array([0, 5])
        self.field = 2 * np.exp(1j * np.linspace(.1, .6, 6))

    def simulate(self, sigma, seed=42, linear=False):
        return run_monte_carlo(
            self.field, np.full(6, sigma), self.times, self.on, np.arange(6),
            self.layout, self.x, self.y, self.mask, [self.mask], .0355,
            trials=5, seed=seed, linear_fit=linear)

    def test_gridding_matches_original(self):
        values = np.array([1+2j, 3+1j, 4-1j, 2-3j, 1j, 5j])
        cell, count, _ = self.layout
        sums = np.zeros((3, 3), complex)
        for index, value in zip(cell, values):
            sums.ravel()[index] += value
        occupied = count > 0
        raw = np.zeros((3, 3), complex)
        raw[occupied] = sums[occupied] / count[occupied]
        y, x = np.where(occupied)
        yy, xx = np.mgrid[:3, :3]
        expected = griddata((y, x), raw[occupied], (yy, xx), method='nearest')
        filled, actual = average_grid(values, self.layout)
        np.testing.assert_array_equal(raw, actual)
        np.testing.assert_array_equal(filled, expected)

    def test_on_noise_propagates_to_other_points(self):
        for linear in [False, True]:
            corrected, reference, _ = calibrate_phase(self.field, self.times, self.on, linear)
            np.testing.assert_allclose(corrected, 2)
            noisy = self.field.copy()
            noisy[0] *= np.exp(1j * .2)
            changed, new_reference, _ = calibrate_phase(noisy, self.times, self.on, linear)
            self.assertFalse(np.isclose(new_reference[2], reference[2]))
            self.assertFalse(np.isclose(changed[2], corrected[2]))

    def test_zero_noise_and_rms_definition(self):
        for linear in [False, True]:
            result = self.simulate(0, linear=linear)
            phase, _, _ = remove_phase_tilt(
                correct_aperture_phase(result['nominal_aperture']), self.x, self.y, self.mask)
            for surface in result['surface_trials_mm']:
                np.testing.assert_allclose(surface, phase * .0355 * 1000 / (4*np.pi))
            self.assertLess(np.max(result['surface_std_mm']), 1e-14)
            np.testing.assert_allclose(result['rms_trials_mm'][:, 0], np.sqrt(np.mean(
                result['surface_trials_mm'][:, self.mask] ** 2, axis=1)))

    def test_seed_and_sample_standard_deviation(self):
        a, b = self.simulate(.01), self.simulate(.01)
        np.testing.assert_array_equal(a['surface_trials_mm'], b['surface_trials_mm'])
        self.assertFalse(np.array_equal(a['surface_trials_mm'], self.simulate(.01, seed=1)['surface_trials_mm']))
        np.testing.assert_allclose(a['surface_std_mm'], a['surface_trials_mm'].std(axis=0, ddof=1))

    def test_noise_sigmas_and_missing_snr(self):
        sigma, imputed = noise_sigmas([1, 2, .1, 1], [10, 10, 0, 20], [.5, .5, .5, 10])
        np.testing.assert_allclose(sigma, [.1, .2, .15, .05])
        np.testing.assert_array_equal(imputed, [False, False, True, False])
        with self.assertRaises(ValueError):
            noise_sigmas([1, 1], [10, 0], [.5, 10])

    def test_phase_error_high_snr(self):
        rng = np.random.default_rng(4)
        z = 1 + .01 * (rng.normal(size=200000) + 1j*rng.normal(size=200000))
        self.assertLess(abs(np.std(np.angle(z))/.01 - 1), .01)


if __name__ == '__main__':
    unittest.main()
