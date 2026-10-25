"""Numerical checks of the complex Gaussian likelihood and Fourier convention."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import linalg

spec = importlib.util.spec_from_file_location('bayesian_holography', Path(__file__).resolve().parents[1]/'bayesian_holography.py')
bh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bh)


class BayesianTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(30)
        f = rng.normal(size=(7, 3))+1j*rng.normal(size=(7, 3))
        d = rng.normal(size=7)+1j*rng.normal(size=7)
        sigma = np.linspace(.3, .8, 7)
        eig = np.array([0., .5, 2.])
        args = SimpleNamespace(length_prior_m=2., scale_prior=1., length_log_sd=.8, scale_log_sd=2.)
        return bh.GaussianAperture(f, d, sigma, eig, args), f, d, sigma, eig

    def test_collapsed_likelihood_matches_data_covariance(self):
        model, f, d, sigma, eig = self.model()
        def direct(theta):
            ell, tau = np.exp(theta)
            q = (1+ell**2*eig)**2/tau**2
            s = np.diag(sigma**2)+(f/q)@f.conj().T
            _, logdet = np.linalg.slogdet(s)
            return -logdet-.5*np.vdot(d, linalg.solve(s, d)).real-.5*np.sum(((theta-model.prior_mean)/model.prior_sd)**2)
        a, b = np.log([.8, 1.2]), np.log([2.3, .7])
        self.assertAlmostEqual(model.conditional(a)[0]-model.conditional(b)[0], direct(a)-direct(b), places=10)

    def test_complex_conditional_draw_covariance(self):
        model, *_ = self.model()
        _, mean, chol = model.conditional(np.log([1., 1.]))
        rng = np.random.default_rng(1)
        z = rng.normal(size=(3, 60000))+1j*rng.normal(size=(3, 60000))
        samples = linalg.solve_triangular(chol.conj().T, z)
        covariance = samples@samples.conj().T/samples.shape[1]
        expected = 2*linalg.cho_solve((chol, True), np.eye(3))
        np.testing.assert_allclose(covariance, expected, atol=.002, rtol=.025)
        np.testing.assert_allclose(samples.mean(axis=1), 0, atol=.005)
        # Proper complex Gaussian has zero pseudocovariance.
        np.testing.assert_allclose(samples@samples.T/samples.shape[1], 0, atol=.003)

    def test_fourier_sign_and_pixel_integral(self):
        x, y = np.array([3., -2.]), np.array([-4., 1.])
        az, el, wave, dx = np.array([1., -2.]), np.array([.3, 1.]), .0355, .7
        result = bh.fourier_operator(az, el, x, y, wave, dx)
        expected = np.exp(-2j*np.pi*bh.ARCMIN/wave*(az[:, None]*x+el[:, None]*y))/2
        expected *= (np.sinc(dx*az*bh.ARCMIN/wave)*np.sinc(dx*el*bh.ARCMIN/wave))[:, None]
        np.testing.assert_allclose(result, expected, atol=1e-14)
        np.testing.assert_allclose(bh.fourier_operator([0], [0], x, y, wave, dx)@np.ones(2), 1)

    def test_motion_average_matches_time_integration(self):
        length = np.array([.5]); rates = np.array([[3., -.3]])
        x, y = np.array([7., -3.]), np.array([4., 2.])
        f = bh.fourier_operator([2.], [-1.], x, y, .0355, 1., length=length, rates=rates)
        t = np.linspace(-.25, .25, 10001)
        instantaneous = bh.fourier_operator(2+3*t, -1-.3*t, x, y, .0355, 1.)
        from scipy.integrate import trapezoid
        numerical = trapezoid(instantaneous, t, axis=0)/.5
        np.testing.assert_allclose(f[0], numerical, atol=2e-6)

    def test_diagnostics_detect_displaced_chains(self):
        rng = np.random.default_rng(0)
        independent = rng.normal(size=(4, 2000))
        good = bh.diagnostics(independent)
        self.assertLess(good['rank_split_rhat'], 1.01)
        self.assertGreater(good['bulk_ess'], 4000)
        independent[0] += 3
        self.assertGreater(bh.diagnostics(independent)['rank_split_rhat'], 1.1)

    def test_zero_snr_uses_mean_sigma_at_matching_length(self):
        table = pd.DataFrame(dict(Epoch=['a','b','c','d','e','f'],
                                  Length=[.5,.5,.5,.5,10,10],
                                  Amp=[.2,.6,.000065,0,10,1],
                                  SNR=[2,2,0,0,10,0]))
        sigma, imputed = bh.measurement_sigmas(table)
        # mean(.1,.3)=.2; the long ON's sigma=1 must not contribute.
        np.testing.assert_allclose(sigma, [.1,.3,.2,.2,1,1])
        np.testing.assert_array_equal(imputed, [False,False,True,True,False,True])
        table.loc[0, 'SNR'] = -1
        for override in (None, .5):
            with self.assertRaisesRegex(ValueError, 'negative SNR'):
                bh.measurement_sigmas(table, override)

    def test_zero_snr_without_matching_donors_fails(self):
        table = pd.DataFrame(dict(Epoch=['a','b'], Length=[.5,10], Amp=[.1,1], SNR=[0,2]))
        with self.assertRaisesRegex(ValueError, 'no positive-SNR points'):
            bh.measurement_sigmas(table)
        sigma, imputed = bh.measurement_sigmas(table, .25)
        np.testing.assert_allclose(sigma, .25)
        self.assertFalse(imputed.any())

    def test_on_calibration_and_34m_sign(self):
        args = bh.parser().parse_args(['--in', 'placeholder', '--antenna', '34'])
        with tempfile.TemporaryDirectory() as d:
            args.input = Path(d)/'beam.txt'
            epochs = pd.date_range('2026-07-03', periods=6, freq='20s')
            table = pd.DataFrame(dict(Epoch=epochs.strftime('%Y/%j %H:%M:%S.%f'), Length=[10,.5,.5,.5,.5,10],
                                      Amp=np.full(6, 2.), Phase=np.full(6, -30.), SNR=np.full(6, 20.),
                                      Az_Offset=[0,1,2,3,4,0], El_Offset=[0,1,2,3,4,0]))
            table.to_csv(args.input, index=False)
            rows, used, field, sigma, _ = bh.read_data(args)
            np.testing.assert_allclose(field, 1, atol=1e-14)
            np.testing.assert_allclose(sigma, .05)
            self.assertEqual(len(used), 4)
            table.loc[2, 'SNR'] = 0
            table.to_csv(args.input, index=False)
            rows, used, field, sigma, _ = bh.read_data(args)
            np.testing.assert_allclose(sigma, .05)
            self.assertTrue(rows.quadrature_sigma_imputed.iloc[2])
            self.assertAlmostEqual(rows.raw_quadrature_sigma.iloc[2], .1)
            np.testing.assert_allclose(rows.reference_phase_rad, np.pi/6)
            table.loc[2, 'Epoch'] = table.loc[1, 'Epoch']
            table.to_csv(args.input, index=False)
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                bh.read_data(args)


if __name__ == '__main__':
    unittest.main()
