import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from online_test import Predictor
from force_history import ForceHistory
from sensors.gsrobotics.gs3drecon import Reconstruction3D
from evaluation_recording import EvaluationRecorder, EvaluationRecordingUI
from nano25_live import parse_force_map


class OnlineComparisonTests(unittest.TestCase):
    def predictor(self):
        p = Predictor.__new__(Predictor)
        p.args = SimpleNamespace(nano_force_map=None)
        p.nano = Mock()
        p.nano.match.return_value = {'force_n': np.array([1., 2., 3.]),
                                    'status': 'OK', 'delta_seconds': .001}
        p.dev = Mock()
        p.depth_nn = Reconstruction3D(SimpleNamespace(imgw=24, imgh=32))
        p.depth_nn.get_depthmap = Mock(return_value=np.zeros((24, 32)))
        p.bg = np.full((24, 32, 3), 100, np.uint8)
        p.transform_rgb = lambda image: torch.zeros((3, 24, 32))
        p.model = SimpleNamespace(tactile_backbone=lambda x: x.mean((2, 3)),
                                 regressor=lambda x: x + torch.tensor([1., 2., 3.]))
        p.quit_requested = False
        p.force_history = ForceHistory()
        p.evaluation_recorder = None
        p.evaluation_ui = None
        p.evaluation_mouse_registered = False
        return p

    def test_record_key_logs_the_plotted_pair_and_stop_calculates_mse(self):
        p = self.predictor()
        p.args = SimpleNamespace(nano_force_map=parse_force_map('-y,x,z'), checkpoint='original.pth',
                                 dataset_stats='mean_and_std.json', nano_max_gap_ms=50.)
        p.nano.baseline_n = np.array([10., 20., 30.])
        p.nano.match.return_value = dict(force_n=np.array([-1., 4., 2.]), status='OK', delta_seconds=.001)
        p.get_tacdepth = lambda *args: (np.full((24, 32, 3), 110, np.uint8), None)
        with tempfile.TemporaryDirectory() as tmp:
            p.evaluation_recorder = EvaluationRecorder(tmp)
            p.evaluation_ui = EvaluationRecordingUI(p.evaluation_recorder)
            with patch.object(torch.Tensor, 'cuda', lambda self, **kw: self), \
                 patch('online_test.cv2.imshow'), patch('online_test.cv2.setMouseCallback'), \
                 patch('online_test.cv2.getWindowProperty', return_value=1), \
                 patch('online_test.cv2.waitKey', side_effect=[ord('r'), ord('r')]):
                p.frame_monotonic = 10.
                p.predict()
                p.frame_monotonic = 11.
                p.predict()
            self.assertIsNotNone(p.evaluation_recorder.directory)
            report = json.loads((p.evaluation_recorder.directory/'metrics.json').read_text())
            self.assertEqual(report['valid_samples'], 1)
            self.assertEqual(report['mse_per_axis_n2'], {'Fx': 4., 'Fy': 4., 'Fz': 1.})
            self.assertEqual(report['mse_overall_n2'], 3.)

    def test_close_finalizes_active_recording_before_releasing_devices(self):
        p = self.predictor()
        with tempfile.TemporaryDirectory() as tmp:
            p.evaluation_recorder = EvaluationRecorder(tmp)
            p.evaluation_recorder.start({})
            p.evaluation_recorder.append(10., [1, 2, 3], p.nano.match.return_value)
            with patch('online_test.cv2.destroyAllWindows'):
                p.close()
            self.assertFalse(p.evaluation_recorder.active)
            report = json.loads((p.evaluation_recorder.directory/'metrics.json').read_text())
            self.assertEqual(report['valid_samples'], 1)
            self.assertEqual(report['stop_reason'], 'shutdown')

    def test_closing_live_window_saves_active_recording(self):
        p = self.predictor()
        p.get_tacdepth = lambda *args: (np.full((24, 32, 3), 110, np.uint8), None)
        p.frame_monotonic = 10.
        with tempfile.TemporaryDirectory() as tmp:
            p.evaluation_recorder = EvaluationRecorder(tmp)
            p.evaluation_recorder.start({})
            p.evaluation_ui = EvaluationRecordingUI(p.evaluation_recorder)
            with patch.object(torch.Tensor, 'cuda', lambda self, **kw: self), \
                 patch('online_test.cv2.imshow'), patch('online_test.cv2.setMouseCallback'), \
                 patch('online_test.cv2.getWindowProperty', return_value=0), \
                 patch('online_test.cv2.waitKey', return_value=-1):
                p.predict()
            self.assertTrue(p.quit_requested)
            self.assertFalse(p.evaluation_recorder.active)
            report = json.loads((p.evaluation_recorder.directory/'metrics.json').read_text())
            self.assertEqual(report['valid_samples'], 1)
            self.assertEqual(report['stop_reason'], 'quit')

    def test_rezero_finishes_recording_before_reference_frames(self):
        p = self.predictor()
        with tempfile.TemporaryDirectory() as tmp:
            p.evaluation_recorder = EvaluationRecorder(tmp)
            p.evaluation_recorder.start({})

            def capture(*args, zero_samples=None):
                self.assertFalse(p.evaluation_recorder.active)
                zero_samples.append(np.zeros(3))
                return np.zeros((24, 32, 3), np.uint8), None

            p.get_tacdepth = capture
            p.reset_reference()
            report = json.loads((p.evaluation_recorder.directory/'metrics.json').read_text())
            self.assertEqual(report['total_samples'], 0)
            self.assertEqual(report['stop_reason'], 're-zero')

    def test_camera_time_is_taken_before_depth_computation_and_inference(self):
        p = self.predictor()
        p.dev.get_image.return_value = np.zeros((24, 32, 3), np.uint8)
        with patch('online_test.time.monotonic', return_value=123.):
            p.get_tacdepth(p.dev, p.depth_nn)
        self.assertEqual(p.frame_monotonic, 123.)

    def test_predict_shows_comparison_and_quit_without_opening_second_camera(self):
        p = self.predictor()
        p.frame_monotonic = 123.
        p.get_tacdepth = lambda *args: (np.full((24, 32, 3), 110, np.uint8), np.zeros((24, 32)))
        with patch.object(torch.Tensor, 'cuda', lambda self, **kw: self), \
                patch('online_test.cv2.imshow') as show, patch('online_test.cv2.waitKey', return_value=ord('q')):
            output, image, depth = p.predict()
        np.testing.assert_allclose(output.numpy(), [[1., 2., 3.]])
        p.nano.match.assert_called_once_with(123.)
        self.assertTrue(p.quit_requested)
        self.assertEqual(show.call_args.args[0], 'FeelAnyForce + Nano25')
        self.assertGreater(show.call_args.args[1].shape[0], 24)

    def test_unloaded_rezero_updates_nano_and_rgb_reference_and_cleanup_releases_resources(self):
        p = self.predictor()
        frame = np.full((24, 32, 3), (10, 20, 30), np.uint8)
        def capture(*args, zero_samples=None):
            if zero_samples is not None:
                zero_samples.append(np.array([1., 2., 3.]))
            return frame, None
        p.get_tacdepth = capture
        p.force_history.append(10., [1, 2, 3], [4, 5, 6])
        p.reset_reference()
        p.nano.wait_ready.assert_called_once()
        np.testing.assert_array_equal(p.nano.zero_from_frames.call_args.args[0], [[1, 2, 3]]*50)
        np.testing.assert_array_equal(p.bg[0, 0], [30, 20, 10])
        self.assertEqual(len(p.force_history.arrays()[0]), 0)
        with patch('online_test.cv2.destroyAllWindows'):
            p.close()
        p.nano.close.assert_called_once()
        p.dev.cam.release.assert_called_once()

    def test_prediction_only_mode_does_not_require_nano(self):
        p = self.predictor()
        p.nano = None
        p.get_tacdepth = lambda *args: (np.full((24, 32, 3), 110, np.uint8), None)
        with patch.object(torch.Tensor, 'cuda', lambda self, **kw: self), \
                patch('online_test.cv2.imshow') as show, patch('online_test.cv2.waitKey', return_value=-1):
            output, _, _ = p.predict()
        self.assertEqual(output.shape, (1, 3))
        self.assertEqual(show.call_args.args[0], 'image_rgb_check')

    def test_live_history_tracks_frame_times_and_retains_invalid_gt_as_a_gap(self):
        p = self.predictor()
        p.get_tacdepth = lambda *args: (np.full((24, 32, 3), 110, np.uint8), None)
        with patch.object(torch.Tensor, 'cuda', lambda self, **kw: self), \
                patch('online_test.cv2.imshow') as show, patch('online_test.cv2.waitKey', return_value=-1):
            p.frame_monotonic = 10.
            p.predict()
            p.frame_monotonic = 10.2
            p.nano.match.return_value = {'force_n': None, 'status': 'Disconnected', 'delta_seconds': None}
            p.predict()
        times, pred, gt = p.force_history.arrays()
        np.testing.assert_allclose(times, [-.2, 0.])
        np.testing.assert_allclose(pred, [[1, 2, 3], [1, 2, 3]])
        self.assertTrue(np.isnan(gt[1]).all())
        self.assertEqual(show.call_args.args[1].shape, (660, 1280, 3))


if __name__ == '__main__':
    unittest.main()
