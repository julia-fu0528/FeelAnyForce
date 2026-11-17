import unittest
from unittest.mock import patch

import cv2
import numpy as np

from nano25_live import Nano25Reader, parse_force_map
import calibration_ui


class CalibrationControlsTests(unittest.TestCase):
    def controls(self, count=3, interval=1.):
        reader = Nano25Reader.__new__(Nano25Reader)
        reader.baseline_n = np.array([100., 200., 300.])
        reader.mapping = parse_force_map('-y,x,z')
        return calibration_ui.RecordingControls(reader, count, interval)

    def test_zero_uses_exactly_next_ten_readings_and_never_records_them(self):
        controls = self.controls()
        controls.command('toggle')
        controls.command('zero')
        for index in range(9):
            self.assertFalse(controls.consume(index, np.array([index, 2*index, 3*index])))
            np.testing.assert_array_equal(controls.reader.baseline_n, [100, 200, 300])
        self.assertFalse(controls.consume(9., np.array([9., 18., 27.])))
        np.testing.assert_allclose(controls.reader.baseline_n, [4.5, 9., 13.5])
        self.assertFalse(controls.recording)
        self.assertIsNone(controls.zero_samples)
        controls.command('toggle')
        self.assertTrue(controls.consume(10., np.array([5.5, 11., 16.5])))
        np.testing.assert_allclose(controls.force, [-2., 1., 3.])

    def test_bad_sample_cancels_zero_without_partial_baseline_update(self):
        controls = self.controls()
        controls.command('zero')
        controls.consume(1., np.array([1., 2., 3.]))
        self.assertFalse(controls.consume(2., None, 'Nano disconnected'))
        np.testing.assert_array_equal(controls.reader.baseline_n, [100, 200, 300])
        self.assertIsNone(controls.zero_samples)
        self.assertFalse(controls.recording)
        self.assertIn('Nano disconnected', controls.message)

    def test_preview_continues_between_due_records_and_limit_is_respected(self):
        controls = self.controls(count=2)
        raw = np.array([101., 202., 303.])
        self.assertFalse(controls.consume(0., raw))
        controls.command('toggle')
        self.assertTrue(controls.consume(1., raw))
        controls.saved(1.)
        self.assertFalse(controls.consume(1.5, raw+1))
        np.testing.assert_allclose(controls.force, [-3., 2., 4.])
        self.assertTrue(controls.consume(2., raw))
        controls.saved(2.)
        self.assertFalse(controls.consume(3., raw))
        self.assertEqual(controls.count, 2)
        self.assertFalse(controls.recording)

    def test_start_cannot_interrupt_zero_and_rezero_does_not_accumulate_offset(self):
        controls = self.controls()
        for baseline in (2., 5.):
            controls.command('zero')
            controls.command('toggle')
            self.assertFalse(controls.recording)
            for i in range(10):
                controls.consume(baseline*10+i, np.full(3, baseline))
            np.testing.assert_array_equal(controls.reader.baseline_n, [baseline]*3)

    def test_render_uses_mapped_forces_and_zero_button_is_clickable(self):
        controls = self.controls()
        controls.consume(1., np.array([101., 202., 303.]))
        ui = calibration_ui.CalibrationUI(controls)
        panel = ui.render(np.zeros((240, 320, 3), np.uint8), 1., .001)
        self.assertEqual(panel.dtype, np.uint8)
        self.assertLessEqual(panel.shape[1], 1360)
        np.testing.assert_array_equal(ui.history.arrays()[2][-1], [-2., 1., 3.])
        x, y, w, h = ui.buttons['zero']
        ui.on_mouse(cv2.EVENT_LBUTTONUP, x+w//2, y+h//2, 0, None)
        self.assertEqual(ui.actions.pop(), 'zero')

    def test_zero_after_saved_sample_marks_next_record_and_clears_old_plot(self):
        controls = self.controls()
        ui = calibration_ui.CalibrationUI(controls)
        frame = np.zeros((240, 320, 3), np.uint8)
        controls.command('toggle')
        controls.consume(1., np.array([101., 202., 303.]))
        controls.saved(1.)
        ui.render(frame, 1., 0.)
        controls.command('zero')
        for index in range(10):
            self.assertFalse(controls.consume(2.+index, np.full(3, 7.)))
        self.assertEqual(controls.zero_event['first_record_index'], 1)
        ui.render(frame, 11., 0.)
        self.assertEqual(len(ui.history.samples), 1)
        np.testing.assert_array_equal(ui.history.arrays()[2], [[0., 0., 0.]])

    def test_window_close_is_treated_as_quit(self):
        controls = self.controls()
        ui = calibration_ui.CalibrationUI(controls)
        with patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=-1), \
             patch.object(cv2, 'getWindowProperty', side_effect=cv2.error('window closed')):
            actions = ui.show(np.zeros((240, 320, 3), np.uint8), 1., None)
        self.assertIn('quit', actions)


if __name__ == '__main__':
    unittest.main()
