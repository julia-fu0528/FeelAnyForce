import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation_recording import EvaluationRecorder, summarize_csv, EvaluationRecordingUI


class EvaluationRecordingTests(unittest.TestCase):
    def measurement(self, force, status='OK', delta=.002):
        return dict(force_n=force, status=status, delta_seconds=delta)

    def test_saved_pairs_and_mse_use_exact_displayed_values_without_another_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EvaluationRecorder(tmp)
            recorder.start({'nano_force_map': '-y,x,z', 'checkpoint': 'original.pth'})
            recorder.append(10., [2., 4., 6.], self.measurement([1., 2., 3.]))
            recorder.append(11., [4., 2., 2.], self.measurement([1., 2., 3.]))
            report = recorder.stop()
            self.assertEqual(report['valid_samples'], 2)
            self.assertEqual(report['mse_per_axis_n2'], {'Fx': 5., 'Fy': 2., 'Fz': 5.})
            self.assertEqual(report['mse_overall_n2'], 4.)
            with (recorder.directory/'pairs.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]['gt_fx_n']), 1.)
            self.assertEqual(float(rows[0]['pred_fx_n']), 2.)
            self.assertAlmostEqual(float(rows[0]['nano_monotonic_s']), 10.002)
            self.assertEqual(json.loads((recorder.directory/'metrics.json').read_text()), report)
            recomputed = summarize_csv(recorder.directory/'pairs.csv')
            self.assertEqual(recomputed['mse_per_axis_n2'], report['mse_per_axis_n2'])
            self.assertEqual(recomputed['mse_overall_n2'], 4.)

    def test_missing_faulted_or_nonfinite_pairs_are_logged_but_excluded_from_mse(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EvaluationRecorder(tmp)
            recorder.start({})
            recorder.append(1., [1, 2, 3], self.measurement(None, 'No match', None))
            recorder.append(2., [1, 2, 3], self.measurement([1, 2, 3], 'Sensor fault'))
            recorder.append(3., [np.nan, 2, 3], self.measurement([1, 2, 3]))
            recorder.append(4., [1, 2, 3], self.measurement([1, np.inf, 3]))
            recorder.append(5., [1, 2, 3], self.measurement([1, 2, 3]))
            report = recorder.stop()
            self.assertEqual(report['total_samples'], 5)
            self.assertEqual(report['valid_samples'], 1)
            self.assertEqual(report['invalid_samples'], 4)
            self.assertEqual(report['mse_overall_n2'], 0.)
            self.assertEqual(summarize_csv(recorder.directory/'pairs.csv')['valid_samples'], 1)

    def test_start_stop_creates_separate_runs_and_ignores_frames_when_not_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EvaluationRecorder(tmp)
            recorder.append(0., [1, 2, 3], self.measurement([0, 0, 0]))
            self.assertEqual(list(Path(tmp).iterdir()), [])
            recorder.start({})
            first = recorder.directory
            with self.assertRaises(RuntimeError):
                recorder.start({})
            recorder.append(1., [1, 2, 3], self.measurement([0, 0, 0]))
            recorder.stop('re-zero')
            first_contents = (first/'pairs.csv').read_text()
            recorder.append(2., [10, 20, 30], self.measurement([0, 0, 0]))
            recorder.start({})
            self.assertNotEqual(recorder.directory, first)
            report = recorder.stop()
            self.assertEqual(report['valid_samples'], 0)
            self.assertIsNone(report['mse_overall_n2'])
            self.assertEqual((first/'pairs.csv').read_text(), first_contents)

    def test_record_button_has_click_target_and_metrics_render_without_samples(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            ui = EvaluationRecordingUI(EvaluationRecorder(tmp))
            panel = ui.render(np.zeros((660, 1280, 3), np.uint8))
            self.assertEqual(panel.dtype, np.uint8)
            self.assertLessEqual(panel.shape[0], 800)
            x, y, w, h = ui.button
            ui.on_mouse(cv2.EVENT_LBUTTONUP, x+w//2, y+h//2, 0, None)
            self.assertTrue(ui.take_toggle())
            self.assertFalse(ui.take_toggle())


if __name__ == '__main__':
    unittest.main()
