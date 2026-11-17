import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np

from sensors.gsrobotics import gsdevice


class CameraSelectionTests(unittest.TestCase):
    def select(self, captures):
        with patch.object(gsdevice.os, 'listdir', return_value=['video18', 'video19']), \
             patch('builtins.open', mock_open(read_data='GelSight Mini\n')), \
             patch.object(gsdevice.cv2, 'VideoCapture', side_effect=lambda index, *args: captures[index]):
            return gsdevice.get_camera_id('GelSight Mini')

    def capture(self, opened=True, readable=True):
        capture = Mock()
        capture.isOpened.return_value = opened
        capture.read.return_value = (readable, np.zeros((8, 8, 3), dtype=np.uint8) if readable else None)
        return capture

    def test_selects_video_stream_instead_of_last_matching_metadata_node(self):
        self.assertEqual(self.select({18: self.capture(), 19: self.capture(opened=False)}), 18)

    def test_skips_node_that_opens_but_cannot_deliver_frames(self):
        self.assertEqual(self.select({18: self.capture(readable=False), 19: self.capture()}), 19)

    def test_no_working_stream_raises_camera_error(self):
        with self.assertRaisesRegex(RuntimeError, 'GelSight Mini'):
            self.select({18: self.capture(opened=False), 19: self.capture(opened=False)})

    def test_connect_failure_raises_before_image_processing(self):
        with patch.object(gsdevice, 'get_camera_id', return_value=18), \
             patch.object(gsdevice.cv2, 'VideoCapture', return_value=self.capture(opened=False)):
            camera = gsdevice.Camera('GelSight Mini')
            with self.assertRaisesRegex(RuntimeError, '18'):
                camera.connect()


if __name__ == '__main__':
    unittest.main()
