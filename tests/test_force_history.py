import unittest

import numpy as np

from force_history import ForceHistory, draw_history, force_limits, trace_segments
from nano25_live import comparison_panel


class ForceHistoryTests(unittest.TestCase):
    def test_vertical_limits_cover_both_series_on_one_scale_and_include_zero(self):
        np.testing.assert_allclose(force_limits([1, 2], [-100, 100]), [-116, 116])
        low, high = force_limits([3, 4], [5, 6])
        self.assertLessEqual(low, 0.)
        self.assertGreater(high, 6.)
        np.testing.assert_allclose(force_limits([np.nan], []), [-.116, .116])

    def test_history_uses_camera_time_and_expires_old_samples(self):
        history = ForceHistory(window_seconds=2.)
        for t in (10., 10.2, 11., 12.5):
            history.append(t, [t, -t, 0], [t+.1, -t+.1, .1])
        times, prediction, gt = history.arrays()
        np.testing.assert_allclose(times, [-1.5, 0.])
        np.testing.assert_allclose(prediction[:, 0], [11., 12.5])
        np.testing.assert_allclose(gt[:, 0], [11.1, 12.6])

    def test_missing_gt_breaks_line_but_does_not_remove_prediction(self):
        history = ForceHistory()
        history.append(10., [1, 2, 3], [2, 3, 4])
        history.append(10.1, [2, 3, 4], None)
        history.append(10.2, [3, 4, 5], [4, 5, 6])
        times, pred, gt = history.arrays()
        self.assertTrue(np.isnan(gt[1]).all())
        self.assertEqual([len(s) for s in trace_segments(times, gt[:, 0])], [1, 1])
        self.assertEqual([len(s) for s in trace_segments(times, pred[:, 0])], [3])

    def test_long_capture_gap_breaks_lines_without_inventing_intermediate_values(self):
        segments = trace_segments(np.array([-4., -3.9, 0.]), np.array([1., 2., 3.]))
        self.assertEqual([len(s) for s in segments], [2, 1])

    def test_inputs_are_copied_and_memory_is_bounded_even_at_high_rate(self):
        history = ForceHistory(max_samples=3)
        force = np.array([1., 2., 3.])
        for i in range(5):
            history.append(10.+i*.001, force, force)
        force[:] = 100
        _, pred, gt = history.arrays()
        self.assertEqual(len(pred), 3)
        np.testing.assert_array_equal(pred, [[1, 2, 3]]*3)
        np.testing.assert_array_equal(gt, pred)

    def test_clear_removes_previous_baseline_and_new_history_starts_at_zero(self):
        history = ForceHistory()
        history.append(10., [1, 2, 3], [4, 5, 6])
        history.clear()
        times, pred, gt = history.arrays()
        self.assertEqual(pred.shape, (0, 3))
        self.assertEqual(gt.shape, (0, 3))
        history.append(20., [0, 0, 0], None)
        np.testing.assert_array_equal(history.arrays()[0], [0.])

    def test_nonfinite_forces_become_gaps_and_bad_timestamps_do_not_corrupt_history(self):
        history = ForceHistory()
        history.append(10., [np.inf, 1, 2], [1, np.nan, 3])
        self.assertTrue(np.isnan(history.arrays()[1][0, 0]))
        for t in (9., 10., np.nan):
            with self.assertRaises(ValueError):
                history.append(t, [1, 2, 3], None)
        self.assertEqual(len(history.arrays()[0]), 1)

    def test_invalid_history_window_is_rejected(self):
        for seconds in (0., -1., np.nan, np.inf, 121.):
            with self.assertRaises(ValueError):
                ForceHistory(window_seconds=seconds)

    def test_render_empty_and_partial_history_on_same_scale_and_fits_on_screen(self):
        history = ForceHistory()
        empty = draw_history(history)
        self.assertEqual(empty.dtype, np.uint8)
        history.append(10., [1, -1, 2], None)
        history.append(10.1, [2, -2, 3], [1, -1, 2])
        full = draw_history(history)
        self.assertEqual(full.shape, empty.shape)
        self.assertFalse(np.array_equal(full, empty))
        for shape in ((320, 240, 3), (240, 320, 3)):
            panel, _ = comparison_panel(np.zeros(shape, np.uint8), [1, 2, 3],
                                        {'force_n': np.array([2, 3, 4]), 'delta_seconds': 0., 'status': 'OK'},
                                        history=history)
            self.assertLessEqual(panel.shape[0], 700)
            self.assertLessEqual(panel.shape[1], 1360)


if __name__ == '__main__':
    unittest.main()
