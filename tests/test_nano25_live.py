import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from nano25_live import (Nano25Reader, baseline_from_samples, comparison_panel, force_scale,
                         match_force, parse_force_map)


def sample(t, values=(1., 2., 3.), status=0, sequence=1):
    return SimpleNamespace(monotonic=t, values=(*values, 0., 0., 0.),
                           status=status, ft_sequence=sequence)


class NanoComparisonTests(unittest.TestCase):
    def test_live_zero_collects_post_request_samples_and_keeps_old_baseline_on_timeout(self):
        reader = Nano25Reader.__new__(Nano25Reader)
        reader.scale = 1.
        reader.baseline_n = np.zeros(3)
        rows = [sample(10.01+i*.009, sequence=i) for i in range(10)]
        reader.acquisition = Mock()
        reader.acquisition.snapshot.return_value = (rows, '', 0)
        with patch('nano25_live.time.monotonic', side_effect=[10., 10.1, 10.1]):
            reader.zero()
        np.testing.assert_array_equal(reader.baseline_n, [1, 2, 3])
        reader.acquisition.snapshot.return_value = (rows, 'UDP disconnected', 1)
        with patch('nano25_live.time.monotonic', side_effect=[20., 20.1, 20.2]), \
                patch('nano25_live.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'UDP disconnected'):
                reader.zero(timeout=.15)
        np.testing.assert_array_equal(reader.baseline_n, [1, 2, 3])

    def test_lbf_baseline_is_removed_before_signed_axis_mapping_in_newtons(self):
        scale = force_scale('lbf')
        samples = [sample(11+i*.01, sequence=i) for i in range(10)]
        baseline = baseline_from_samples(samples, since=10., now=11.1, scale=scale)
        result = match_force([sample(12., (2., 4., 6.))], 12.005, baseline,
                             scale, parse_force_map('-y,x,z'))
        np.testing.assert_allclose(result['force_n'], [-8.896443230521, 4.4482216152605, 13.3446648457815])
        self.assertAlmostEqual(result['delta_seconds'], -.005)

    def test_mapping_rejects_repeated_or_missing_axes_and_unknown_force_units(self):
        for value in ('x,x,z', 'x,y', 'x,y,w', 'x,y,2z', 'x,y,z,w'):
            with self.assertRaises(ValueError):
                parse_force_map(value)
        for unit in ('counts', '', 'N/mm'):
            with self.assertRaises(ValueError):
                force_scale(unit)
        self.assertEqual(force_scale('N'), 1.)
        self.assertEqual(force_scale('kN'), 1000.)

    def test_baseline_uses_fresh_unique_post_click_samples(self):
        rows = [sample(10+i*.01, sequence=i) for i in range(10)]
        rows += [sample(10.09, (99, 99, 99), sequence=9)]
        np.testing.assert_array_equal(baseline_from_samples(rows, since=9., now=10.1, scale=1.), [1, 2, 3])
        for since, now in ((10.05, 10.1), (9., 12.)):
            with self.assertRaises(ValueError):
                baseline_from_samples(rows, since=since, now=now, scale=1.)

    def test_unmatched_faulted_or_nonfinite_ft_is_never_displayed_as_ground_truth(self):
        for rows, error in (([], ''), ([sample(9.)], ''),
                            ([sample(10., status=1)], ''),
                            ([sample(10., (np.nan, 1, 2))], ''),
                            ([sample(10.)], 'UDP disconnected')):
            result = match_force(rows, 10., np.zeros(3), 1., error=error)
            self.assertIsNone(result['force_n'])
            self.assertTrue(result['status'])

    def test_matching_uses_camera_receipt_time_not_latest_inference_time(self):
        rows = [sample(10., (2, 3, 4)), sample(10.2, (100, 100, 100))]
        result = match_force(rows, 10.003, np.zeros(3), 1.)
        np.testing.assert_array_equal(result['force_n'], [2, 3, 4])

    def test_display_handles_unavailable_gt_and_labels_unverified_axes(self):
        panel, lines = comparison_panel(np.zeros((240, 320, 3), np.uint8), [1, 2, 3],
                                       {'force_n': None, 'status': 'No F/T sample', 'delta_seconds': None})
        self.assertEqual(panel.ndim, 3)
        self.assertGreater(panel.shape[0], 240)
        self.assertTrue(any('unverified' in line.lower() for line in lines))
        self.assertTrue(any('No F/T sample' in line for line in lines))
        self.assertFalse(any('error' in line.lower() for line in lines))

    def test_display_shows_both_vectors_and_residual_only_with_explicit_mapping(self):
        panel, lines = comparison_panel(np.zeros((240, 320, 3), np.uint8), [1, 2, 3],
                                       {'force_n': np.array([.5, 1.5, 2.5]), 'status': 'OK', 'delta_seconds': .002},
                                       mapping=parse_force_map('x,y,z'))
        self.assertEqual(panel.dtype, np.uint8)
        self.assertTrue(any('Pred-GT' in line for line in lines))
        self.assertTrue(any('Fx' in line and '+1.000' in line and '+0.500' in line for line in lines))

    def test_force_table_fits_on_screen_for_portrait_and_landscape_inputs(self):
        for shape in ((320, 240, 3), (240, 320, 3), (1080, 1920, 3)):
            panel, _ = comparison_panel(np.zeros(shape, np.uint8), [1, 2, 3],
                                        {'force_n': None, 'status': 'No F/T sample', 'delta_seconds': None})
            self.assertLessEqual(panel.shape[0], 700)
            self.assertLessEqual(panel.shape[1], 1000)


if __name__ == '__main__':
    unittest.main()
