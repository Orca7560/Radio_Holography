"""Run with python3 -m unittest test_integrate_beams.py."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from integrate_beams import integrate


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(inputs=[], out=str(self.root / 'integrated.txt'),
            antenna=34, grid_step=(1.5, 1.5), grid_origin=(0, 0), on_length=10,
            on_amp_min=None, exclude_center_crossings=True, on_phase_linear_fit=False,
            overwrite=False)

    def observation(self, name, gain, phase, length, beam_phase, offset=0):
        # ON gain/phase drift linearly; calibrating must recover the same beam amplitude.
        rows = []
        for second, x, duration, amplitude, intrinsic_phase in [
            (0, 0, 10, 1, 0), (20, 1.5 + offset, length, .2, beam_phase),
            (40, 0, 10, 1, 0)]:
            local_gain = gain * (1 + second / 40)
            rows.append(dict(Epoch=f'2026/001 00:00:{second:02d}.000000',
                Length=duration, Amp=local_gain * amplitude,
                Phase=phase + second / 2 + intrinsic_phase, SNR=100,
                Az_Offset=x, El_Offset=0))
        path = self.root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return str(path)

    def run_integration(self):
        with contextlib.redirect_stdout(io.StringIO()):
            integrate(self.args)
        return pd.read_csv(self.args.out)

    def test_coherent_exposure_weighting_after_individual_on_calibration(self):
        self.args.inputs = [self.observation('a.txt', 2, 30, 1, 179),
                            self.observation('b.txt', 9, -40, 3, -179, .1)]
        result = self.run_integration()
        self.assertEqual(len(result), 1)  # no stationary ON in raster
        row = result.iloc[0]
        expected = .2 * (np.exp(1j*np.deg2rad(179)) + 3*np.exp(1j*np.deg2rad(-179))) / 4
        actual = row.Amp * np.exp(1j*np.deg2rad(row.Phase))
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertEqual(row.Length, 4)
        self.assertEqual(row.Az_Offset, 1.5)
        self.assertEqual(row.N_files, 2)
        self.assertEqual(row.N_samples, 2)
        self.assertEqual(row.Beam_State, 'on_calibrated')
        self.assertEqual(row.Antenna_m, 34)
        # Two independent calibrated sigma values, with 1:3 exposure weights.
        self.assertAlmostEqual(row.SNR, abs(expected) / (.2/100 * np.sqrt(10)/4))

    def test_duplicate_contents_rejected(self):
        a = self.observation('a.txt', 2, 0, 1, 10)
        b = self.root / 'copy.txt'
        b.write_bytes(Path(a).read_bytes())
        self.args.inputs = [a, str(b)]
        with self.assertRaisesRegex(ValueError, '重複'):
            self.run_integration()
        self.assertFalse(Path(self.args.out).exists())

    def test_missing_on_rejected(self):
        a = self.observation('a.txt', 2, 0, 1, 10)
        data = pd.read_csv(a).iloc[1:2]
        data.to_csv(a, index=False)
        self.args.inputs = [a]
        with self.assertRaisesRegex(ValueError, 'at least two ON'):
            self.run_integration()

    def test_no_overwrite(self):
        self.args.inputs = [self.observation('a.txt', 2, 0, 1, 10)]
        Path(self.args.out).write_text('keep')
        with self.assertRaisesRegex(ValueError, '存在'):
            self.run_integration()
        self.assertEqual(Path(self.args.out).read_text(), 'keep')

    def test_unavailable_snr_does_not_discard_field(self):
        for snr in (0, -1, np.nan, np.inf):
            with self.subTest(snr=snr):
                a = self.observation('a.txt', 2, 30, 1, 179)
                data = pd.read_csv(a)
                data['SNR'] = snr
                data.to_csv(a, index=False)
                self.args.inputs = [a]
                self.args.overwrite = True
                row = self.run_integration().iloc[0]
                self.assertAlmostEqual(row.Amp, .2)
                self.assertAlmostEqual(row.Phase, 179)
                self.assertEqual(row.N_samples, 1)
                self.assertEqual(row.SNR, 0)
                self.assertFalse(row.Noise_known)

    def test_missing_noise_uses_same_length_peer_before_on_normalization(self):
        a = self.observation('a.txt', 2, 0, 1, 0)
        data = pd.read_csv(a)
        peer = data.iloc[1].copy()
        peer['Epoch'] = '2026/001 00:00:30.000000'
        peer['Amp'] = .7  # .2 times the ON reference (3.5)
        peer['Phase'] = 15
        peer['SNR'] = 0
        data = pd.concat([data, peer.to_frame().T], ignore_index=True)
        data.to_csv(a, index=False)
        self.args.inputs = [a]
        row = self.run_integration().iloc[0]
        self.assertAlmostEqual(row.Amp, .2)
        self.assertEqual(row.N_samples, 2)
        self.assertTrue(row.Noise_known)
        expected_snr = .2 / np.sqrt(((.6 / 100 / 3) ** 2 + (.6 / 100 / 3.5) ** 2) / 4)
        self.assertAlmostEqual(row.SNR, expected_snr)

    def test_bad_field_reports_column_record_and_value_once(self):
        a = self.observation('a.txt', 2, 0, 1, 0)
        data = pd.read_csv(a)
        data.loc[1, 'Length'] = 0
        data.loc[1, 'Amp'] = -1
        data.to_csv(a, index=False)
        self.args.inputs = [a]
        with self.assertRaises(ValueError) as caught:
            self.run_integration()
        message = str(caught.exception)
        self.assertEqual(message.count(a), 1)
        self.assertIn('Length: 1件 (3行目=0', message)
        self.assertIn('Amp: 1件 (3行目=-1', message)


if __name__ == '__main__':
    unittest.main()
