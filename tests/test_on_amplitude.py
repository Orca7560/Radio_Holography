import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holography_mc import calibrate_amplitude, calibrate_phase
class OnAmplitudeTests(unittest.TestCase):
    def test_recovers_known_variable_gain(self):
        t=np.arange(7.); on=np.array([0,3,6]); gain=2+t/3; phase=.1*t
        beam=np.array([1,.4+.2j,.2j,1,-.3j,.2,1])
        field=beam*gain*np.exp(1j*phase)
        phase_cal,_,_=calibrate_phase(field,t,on)
        result,ref=calibrate_amplitude(phase_cal,t,on)
        np.testing.assert_allclose(result,beam,atol=1e-14)
        np.testing.assert_allclose(ref,gain)
    def test_on_amplitude_noise_propagates(self):
        t=np.arange(5.); field=np.ones(5,complex);on=np.array([0,4])
        field[0]=1.2
        result,ref=calibrate_amplitude(field,t,on)
        self.assertAlmostEqual(ref[2],1.1)
        self.assertAlmostEqual(result[2].real,1/1.1)
    def test_invalid_reference_fails(self):
        with self.assertRaises(ValueError):
            calibrate_amplitude(np.array([0,1]),np.arange(2.),np.array([0,1]))
