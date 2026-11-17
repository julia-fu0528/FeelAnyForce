import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from prepare_calibration import prepare_calibration
from args import get_parser
from dataset import TacForceDataset


class PrepareCalibrationTests(unittest.TestCase):
    def recording(self, root, background, count):
        (root/'tactile').mkdir(parents=True)
        (root/'depth').mkdir()
        Image.fromarray(np.full((4, 6, 3), background, dtype=np.uint8)).save(root/'background.png')
        rows = []
        for i in range(count):
            Image.fromarray(np.full((4, 6, 3), background+30, dtype=np.uint8)).save(root/f'tactile/{i}.png')
            np.save(root/f'depth/{i}.npy', np.zeros((4, 6)))
            rows.append([f'tactile/{i}.png', f'depth/{i}.npy', '1.000000 -2.000000 3.000000'])
        pd.DataFrame(rows, columns=['tactile', 'depth', 'FT']).to_csv(root/'calibration.csv', index=False)
        (root/'metadata.json').write_text(json.dumps({'force_unit': 'N', 'force_axes': 'native'}))

    def setup_runs(self, folder):
        train, val = folder/'train', folder/'val'
        self.recording(train, 100, 4)
        self.recording(val, 20, 2)
        stats = folder/'stats.json'
        stats.write_text('{"original.csv": {"nobg": [[0.3,0.3,0.3],[0.2,0.2,0.2]]}}')
        return train, val, stats

    def test_separate_runs_use_own_background_and_load_through_real_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val, stats = self.setup_runs(Path(tmp))
            prepare_calibration(train, val, stats)
            args = get_parser().parse_args(['--data_basedir', str(train)+'/',
                                          '--labels_train', 'calibration_train.csv',
                                          '--labels_val', 'calibration_val.csv'])
            for split, count in [('train', 4), ('val', 2)]:
                dataset = TacForceDataset(args, split)
                self.assertEqual(len(dataset), count)
                image, force, _ = dataset[0]
                np.testing.assert_array_equal(np.array(image), np.full((4, 6, 3), 157))
                np.testing.assert_array_equal(force.numpy(), [1, -2, 3])
            self.assertEqual((train/'normalization.json').read_text(), stats.read_text())
            self.assertEqual(len(pd.read_csv(train/'calibration.csv')), 4)

    def test_existing_preparation_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val, stats = self.setup_runs(Path(tmp))
            (train/'calibration_train.csv').write_text('keep me')
            with self.assertRaises(FileExistsError):
                prepare_calibration(train, val, stats)
            self.assertEqual((train/'calibration_train.csv').read_text(), 'keep me')
            self.assertFalse((train/'tactile_nobg').exists())

    def test_validation_must_be_a_different_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, _, stats = self.setup_runs(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'different'):
                prepare_calibration(train, train, stats)

    def test_incompatible_force_axes_are_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val, stats = self.setup_runs(Path(tmp))
            (val/'metadata.json').write_text(json.dumps({'force_unit': 'N', 'force_axes': 'x,-y,z'}))
            with self.assertRaisesRegex(ValueError, 'axes'):
                prepare_calibration(train, val, stats)
            self.assertFalse((train/'tactile_nobg').exists())


if __name__ == '__main__':
    unittest.main()
