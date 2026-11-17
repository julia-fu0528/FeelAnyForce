import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from force_history import ForceHistory
from nano25_live import Nano25Reader
from online_test import Predictor
from sensors.gsrobotics.gs3drecon import Reconstruction3D


class SharedZeroTests(unittest.TestCase):
    def test_depth_baseline_averages_all_fifty_frames_and_resets_for_rezero(self):
        reconstruction = Reconstruction3D(SimpleNamespace(imgw=4, imgh=6))
        reconstruction.net = Mock(return_value=torch.stack((torch.linspace(-.2, .2, 24),
                                                            torch.linspace(.1, .3, 24)), dim=1))
        frame = np.zeros((4, 6, 3), np.uint8)
        with patch('sensors.gsrobotics.gs3drecon.poisson_dct_neumaan', return_value=np.full((4, 6), 5.)), \
                patch('builtins.print'):
            for _ in range(50):
                depth = reconstruction.get_depthmap(frame, False)
        np.testing.assert_allclose(depth, 0., atol=1e-12)
        np.testing.assert_array_equal(reconstruction.dm_zero, 5.)
        reconstruction.reset_zero()
        self.assertEqual(reconstruction.dm_zero_counter, 0)
        np.testing.assert_array_equal(reconstruction.dm_zero, 0.)

    def reader(self):
        reader = Nano25Reader.__new__(Nano25Reader)
        reader.scale = 4.4482216152605
        reader.mapping = ((1, -1), (0, 1), (2, 1))
        reader.max_gap = .05
        reader.baseline_n = np.array([100., 200., 300.])
        reader.acquisition = Mock()
        return reader

    def test_zero_sample_uses_original_nano_axes_without_previous_baseline(self):
        reader = self.reader()
        row = SimpleNamespace(monotonic=10., values=(1., 2., 3., 0., 0., 0.), status=0)
        reader.acquisition.snapshot.return_value = ([row], '', 0)
        np.testing.assert_allclose(reader.unloaded_force(10.001), [4.4482216152605, 8.896443230521, 13.3446648457815])
        reader.zero_from_frames([[1., 2., 3.], [3., 4., 5.]])
        np.testing.assert_array_equal(reader.baseline_n, [2., 3., 4.])

    def test_bad_zero_samples_fail_without_changing_previous_baseline(self):
        reader = self.reader()
        reader.acquisition.snapshot.return_value = ([], 'Nano disconnected', 1)
        with self.assertRaisesRegex(RuntimeError, 'Nano disconnected'):
            reader.unloaded_force(10.)
        for samples in ([], [[1, np.nan, 3]], [[1, 2]], [[np.inf, 0, 0]]):
            with self.assertRaises(ValueError):
                reader.zero_from_frames(samples)
        np.testing.assert_array_equal(reader.baseline_n, [100, 200, 300])

    def test_wait_ready_waits_for_healthy_current_sample_without_zeroing_early(self):
        reader = self.reader()
        row = SimpleNamespace(monotonic=10.02, values=(1., 2., 3., 0., 0., 0.), status=0)
        reader.acquisition.snapshot.side_effect = [([], 'Disconnected', 1), ([row], '', 1)]
        with patch('nano25_live.time.monotonic', side_effect=[10., 10.01, 10.01, 10.02, 10.02]), \
                patch('nano25_live.time.sleep'):
            reader.wait_ready()
        np.testing.assert_array_equal(reader.baseline_n, [100, 200, 300])

    def test_wait_ready_reports_timeout_without_modifying_baseline(self):
        reader = self.reader()
        reader.acquisition.snapshot.return_value = ([], 'Disconnected', 1)
        with patch('nano25_live.time.monotonic', side_effect=[10., 10.01, 10.01, 16.]), \
                patch('nano25_live.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Disconnected'):
                reader.wait_ready()
        np.testing.assert_array_equal(reader.baseline_n, [100, 200, 300])

    def test_nano_sample_is_frozen_before_depth_computation(self):
        p = Predictor.__new__(Predictor)
        p.nano = Mock()
        p.nano.unloaded_force.return_value = np.array([1., 2., 3.])
        camera, depth = Mock(), Mock()
        camera.get_image.return_value = np.zeros((4, 6, 3), np.uint8)
        captured = []
        def compute(*args):
            np.testing.assert_array_equal(captured, [[1., 2., 3.]])
            return np.zeros((4, 6))
        depth.get_depthmap.side_effect = compute
        with patch('online_test.time.monotonic', return_value=10.):
            p.get_tacdepth(camera, depth, zero_samples=captured)
        p.nano.unloaded_force.assert_called_once_with(10.)

    def predictor(self):
        p = Predictor.__new__(Predictor)
        p.evaluation_recorder = None
        p.nano = self.reader()
        p.nano.wait_ready = Mock()
        p.dev = Mock()
        p.depth_nn = Reconstruction3D(SimpleNamespace(imgw=4, imgh=6))
        p.depth_nn.dm_zero[:] = 7
        p.depth_nn.dm_zero_counter = 80
        p.bg = np.full((4, 6, 3), 90, np.uint8)
        p.force_history = ForceHistory()
        p.force_history.append(10., [1, 2, 3], [4, 5, 6])
        return p

    def test_reset_uses_same_fifty_frames_for_depth_and_nano_then_clears_history(self):
        p = self.predictor()
        counters = []
        def capture(*args, zero_samples=None):
            counter = p.depth_nn.dm_zero_counter
            counters.append(counter)
            zero_samples.append(np.array([counter, 2*counter, -counter], dtype=float))
            p.depth_nn.dm_zero_counter += 1
            return np.full((4, 6, 3), (10, 20, 30), np.uint8), None
        p.get_tacdepth = capture
        p.reset_reference()
        self.assertEqual(counters, list(range(50)))
        np.testing.assert_array_equal(p.nano.baseline_n, [24.5, 49., -24.5])
        np.testing.assert_array_equal(p.bg[0, 0], [30, 20, 10])
        self.assertEqual(len(p.force_history.arrays()[0]), 0)

    def test_failed_shared_zero_restores_previous_depth_image_and_force_baselines(self):
        p = self.predictor()
        def capture(*args, **kwargs):
            p.depth_nn.dm_zero[:] = 99
            p.depth_nn.dm_zero_counter = 3
            raise RuntimeError('Nano disconnected')
        p.get_tacdepth = capture
        with self.assertRaisesRegex(RuntimeError, 'Nano disconnected'):
            p.reset_reference()
        self.assertEqual(p.depth_nn.dm_zero_counter, 80)
        np.testing.assert_array_equal(p.depth_nn.dm_zero, 7.)
        np.testing.assert_array_equal(p.bg, 90)
        np.testing.assert_array_equal(p.nano.baseline_n, [100, 200, 300])
        self.assertEqual(len(p.force_history.arrays()[0]), 1)

    def test_depth_only_mode_still_calibrates_without_nano(self):
        p = self.predictor()
        p.nano = None
        p.force_history = None
        def capture(*args, zero_samples=None):
            self.assertIsNone(zero_samples)
            p.depth_nn.dm_zero_counter += 1
            return np.full((4, 6, 3), 42, np.uint8), None
        p.get_tacdepth = capture
        p.reset_reference()
        self.assertEqual(p.depth_nn.dm_zero_counter, 50)
        np.testing.assert_array_equal(p.bg, 42)


if __name__ == '__main__':
    unittest.main()
