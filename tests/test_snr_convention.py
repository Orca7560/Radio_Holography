import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from holography_mc import noise_sigmas
class SNRConvention(unittest.TestCase):
 def test_recovers_quadrature_sigma_from_mean_noise_amplitude(self):
  rng=np.random.default_rng(21);sigma=.04
  n=sigma*(rng.normal(size=300000)+1j*rng.normal(size=300000))
  mean_amp=np.abs(n).mean();est,_=noise_sigmas([1],[1/mean_amp],[.5],"mean-amplitude")
  self.assertLess(abs(est[0]/sigma-1),.005)
 def test_legacy_definition_and_invalid_choice(self):
  a,_=noise_sigmas([1],[10],[.5],"quadrature");self.assertAlmostEqual(a[0],.1)
  with self.assertRaises(ValueError):noise_sigmas([1],[10],[.5],"power")
