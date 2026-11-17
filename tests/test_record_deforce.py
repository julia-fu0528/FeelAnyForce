import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time

import numpy as np
import pandas as pd
from click.testing import CliRunner

import record_deForce as recorder
from nano25_live import Nano25Reader


class Camera:
    def __init__(self):
        self.frames = 0
        self.closed = False
        self.cam = self

    def get_image(self):
        self.frames += 1
        return np.full((6, 4, 3), self.frames, dtype=np.uint8)

    def release(self):
        self.closed = True


class Depth:
    zero_frames = 50

    def reset_zero(self):
        self.frames = 0

    def get_depthmap(self, image, mask):
        self.frames += 1
        return np.full((6, 4), self.frames, dtype=float)


class Acquisition:
    def __init__(self, camera, fail_frame=None):
        self.camera = camera
        self.fail_frame = fail_frame
        self.closed = False
        self.observed_frames = []

    def snapshot(self):
        frame = self.camera.frames
        self.observed_frames.append(frame)
        sample = SimpleNamespace(monotonic=time.monotonic(), ft_sequence=frame,
                                 status=int(frame == self.fail_frame),
                                 values=(1., 2., 3., 0., 0., 0.) if frame <= 50
                                 else (3., 5., 7., 0., 0., 0.))
        return [sample], '', 0

    def close(self):
        self.closed = True


class RecordingTests(unittest.TestCase):
    def test_ui_zero_saves_mapped_labels_only_after_ten_new_frames(self):
        from nano25_live import parse_force_map
        camera, depth = Camera(), Depth()
        reader = Nano25Reader.__new__(Nano25Reader)
        reader.acquisition = Acquisition(camera)
        reader.scale, reader.max_gap = 1., .05
        reader.mapping = parse_force_map('-y,x,z')
        reader.baseline_n = None
        original_snapshot = reader.acquisition.snapshot

        def snapshot():
            samples, error, count = original_snapshot()
            if camera.frames > 50:
                samples[0].values = (camera.frames, 2*camera.frames, -camera.frames, 0., 0., 0.)
            return samples, error, count

        reader.acquisition.snapshot = snapshot
        displayed = {}

        def show(ui, tactile, stamp, delta):
            displayed[camera.frames] = ui.controls.force.copy()
            ui.render(tactile, stamp, delta)
            return {51: ['zero'], 61: ['toggle'], 63: ['quit']}.get(camera.frames, [])

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'
            with patch.object(recorder, 'Nano25Reader', return_value=reader), \
                 patch.object(recorder, 'init_mini', return_value=(camera, depth)), \
                 patch('calibration_ui.CalibrationUI.open'), \
                 patch('calibration_ui.CalibrationUI.close'), \
                 patch('calibration_ui.CalibrationUI.show', show):
                result = CliRunner().invoke(recorder.main, [
                    '--nano25', '--ui', '--nano-force-map=-y,x,z',
                    '--record_count', '2', '--interval', '0', '--output-dir', str(folder)])
            self.assertEqual(result.exit_code, 0, result.output+repr(result.exception))
            rows = pd.read_csv(folder/'calibration.csv')
            self.assertEqual(len(rows), 2)
            np.testing.assert_allclose(reader.baseline_n, [56.5, 113., -56.5])
            for index, frame in enumerate((62, 63)):
                labels = np.fromstring(rows.FT.iloc[index], sep=' ')
                np.testing.assert_allclose(labels, displayed[frame])
                np.testing.assert_array_equal(np.load(folder/rows.depth.iloc[index]), np.full((6, 4), 51+index))
            np.testing.assert_allclose(np.fromstring(rows.FT.iloc[0], sep=' '), [-11., 5.5, -5.5])
            metadata = json.loads((folder/'metadata.json').read_text())
            self.assertEqual(metadata['zero_events'][-1]['zero_frames'], 10)
            self.assertEqual(metadata['zero_events'][-1]['first_record_index'], 0)
            self.assertEqual(metadata['zero_events'][0]['baseline_n'], [1., 2., 3.])
            self.assertTrue(camera.closed and reader.acquisition.closed)

    def test_ui_requires_ethernet_nano(self):
        result = CliRunner().invoke(recorder.main, ['--ui'])
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn('--nano25', result.output)

    def run_recording(self, directory, *, fail_frame=None, count=2):
        camera, depth = Camera(), Depth()
        reader = Nano25Reader.__new__(Nano25Reader)
        reader.acquisition = Acquisition(camera, fail_frame)
        reader.scale, reader.max_gap = 1., .05
        reader.mapping = None
        reader.baseline_n = None
        with patch.object(recorder, 'Nano25Reader', return_value=reader, create=True), \
                patch.object(recorder, 'init_mini', return_value=(camera, depth)):
            result = CliRunner().invoke(recorder.main, [
                '--nano25', '--nano-bind', '192.168.1.100', '--record_count', str(count),
                '--interval', '0', '--output-dir', str(directory)])
        return result, camera, reader

    def test_network_recording_saves_zeroed_forces_and_matching_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'
            result, camera, reader = self.run_recording(folder)
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            rows = pd.read_csv(folder/'calibration.csv')
            self.assertEqual(len(rows), 2)
            np.testing.assert_allclose(np.fromstring(rows.FT.iloc[0], sep=' '), [2, 3, 4])
            np.testing.assert_allclose(reader.baseline_n, [1, 2, 3])
            for i, row in rows.iterrows():
                self.assertTrue((folder/row.tactile).is_file())
                np.testing.assert_array_equal(np.load(folder/row.depth), np.full((6, 4), 51+i))
            self.assertTrue((folder/'background.png').is_file())
            self.assertTrue(camera.closed and reader.acquisition.closed)

    def test_sensor_fault_keeps_completed_records_and_closes_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'
            result, camera, reader = self.run_recording(folder, fail_frame=72)
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('status', result.output)
            rows = pd.read_csv(folder/'calibration.csv')
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(list((folder/'tactile').glob('*.png'))), 1)
            self.assertTrue(camera.closed and reader.acquisition.closed)

    def test_fault_during_zero_does_not_record_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'
            result, camera, reader = self.run_recording(folder, fail_frame=20)
            self.assertNotEqual(result.exit_code, 0)
            self.assertIsNone(reader.baseline_n)
            self.assertFalse((folder/'calibration.csv').exists())
            self.assertTrue(camera.closed and reader.acquisition.closed)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'calibration.csv').write_text('existing calibration\n')
            result, camera, reader = self.run_recording(folder)
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('already exists', result.output)
            self.assertEqual((folder/'calibration.csv').read_text(), 'existing calibration\n')
            self.assertEqual(camera.frames, 0)

    def test_capture_freezes_force_before_depth_and_discards_buffered_frames(self):
        camera, depth = Camera(), Depth()
        depth.reset_zero()
        events = []
        def force_at(stamp):
            events.append(('force', camera.frames))
            return np.array([2., 3., 4.])
        original_depth = depth.get_depthmap
        def compute(image, mask):
            events.append(('depth', camera.frames))
            return original_depth(image, mask)
        depth.get_depthmap = compute
        image, _, force = recorder.capture_frame(camera, depth, force_at, discard_frames=10)
        self.assertEqual(events, [('force', 11), ('depth', 11)])
        self.assertEqual(int(image[0, 0, 0]), 11)
        np.testing.assert_array_equal(force, [2, 3, 4])

    def test_interruption_keeps_completed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'run'
            with patch.object(recorder.time, 'sleep', side_effect=KeyboardInterrupt):
                result, camera, reader = self.run_recording(folder)
            self.assertNotEqual(result.exit_code, 0)
            self.assertEqual(len(pd.read_csv(folder/'calibration.csv')), 1)
            self.assertTrue(camera.closed and reader.acquisition.closed)

    def test_invalid_counts_and_intervals_are_rejected(self):
        for options in (['--record_count', '0'], ['--interval', '-1'], ['--interval', 'nan']):
            result = CliRunner().invoke(recorder.main, options)
            self.assertEqual(result.exit_code, 2, result.output)


if __name__ == '__main__':
    unittest.main()
