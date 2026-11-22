"""Regression coverage for the padded raster and nominal coordinate mapping."""
import unittest
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import numpy as np
from corr_fringe import offset_tasks_for_scan
from group_up_txt_prd import prd_offset_at


class CenteredRaster(unittest.TestCase):
    def test_full_grid_and_lag(self):
        scan = ET.fromstring('<process><skip>39</skip><length>82</length></process>')
        start = datetime(2026, 7, 3, 8, 21, 40)
        end = start + timedelta(seconds=41)
        for axis in ('az', 'el'):
            for direction in (-1, 1):
                a, b = -60 * direction, 60 * direction
                records = [(start, a, 12), (end, b, 12)] if axis == 'az' else [(start, 12, a), (end, 12, b)]
                for half in (False, True):
                    for lag in (-1000, 0, 1000):
                        tasks = offset_tasks_for_scan(scan, .5, scan_half=half, lag_ms=lag)
                        self.assertEqual(len(tasks), 81)
                        coordinates = []
                        for task in tasks:
                            self.assertEqual(task['length'], task['nominal_length'])
                            self.assertGreaterEqual(task['skip'], 0)
                            self.assertLessEqual(task['skip'] + task['length'], 43000)
                            self.assertEqual(task['skip'] - task['nominal_skip'], lag)
                            epoch = start + timedelta(seconds=task['nominal_skip']/1000 - 1)
                            point = prd_offset_at(records, [start, end], epoch, task['length']/1000,
                                                  0, scan_axis=axis, centered_raster=True)
                            coordinates.append(point[0 if axis == 'az' else 1])
                        np.testing.assert_allclose(coordinates, np.linspace(a, b, 81))

    def test_legacy_window_still_accepted(self):
        scan = ET.fromstring('<process><skip>40</skip><length>81</length></process>')
        self.assertEqual(len(offset_tasks_for_scan(scan, .5)), 82)


if __name__ == '__main__':
    unittest.main()
