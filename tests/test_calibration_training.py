import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from args import get_parser
import training_multi_head as training


class TinyModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def get_param_groups(self, **kwargs):
        return [{'params': [self.weight], 'lr': kwargs['lr_calibration']}]


class CalibrationTrainingTests(unittest.TestCase):
    def checkpoint(self, path):
        torch.save({'state_dict': {'weight': torch.ones(1)}, 'config': {
            'tactile_model': 'dinov2_vitb14', 'crop_size': 224,
            'labels_train': 'original.csv', 'labels_val': 'old_val.csv',
            'data_basedir': '/old/data/', 'ckpt_dir': '/old/output/',
            'batch_size_per_gpu': 128, 'lr_backbone': .5, 'lr_calibration': .5,
            'epochs': 100, 'training_name': 'old', 'tactile_backbone_training': 'finetune',
            'dataset_stats': '/old/stats.json', 'wandb': True}}, path)

    def args(self, folder, checkpoint):
        return get_parser().parse_args([
            '--tactile_backbone_training', 'calibration', '--checkpoint', str(checkpoint),
            '--ckpt_dir', str(folder/'output'), '--data_basedir', str(folder)+'/',
            '--labels_train', 'train.csv', '--labels_val', 'val.csv',
            '--dataset_stats', str(folder/'stats.json'), '--training_name', 'nano',
            '--batch_size_per_gpu', '4', '--num_workers', '0', '--epochs', '1',
            '--lr_calibration', '.0001', '--lr_backbone', '.00001'])

    def test_explicit_checkpoint_preserves_run_options_and_restores_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.checkpoint(folder/'source.pt')
            args = self.args(folder, folder/'source.pt')
            before = vars(args).copy()
            args.crop_size = 999
            checkpoint = training.load_calibration_checkpoint(args)
            self.assertEqual(args.crop_size, 224)
            for key in ('labels_train', 'labels_val', 'data_basedir', 'ckpt_dir', 'epochs',
                        'batch_size_per_gpu', 'dataset_stats', 'lr_backbone', 'lr_calibration',
                        'wandb', 'training_name', 'tactile_backbone_training'):
                self.assertEqual(getattr(args, key), before[key], key)
            self.assertEqual(checkpoint['config']['labels_train'], 'original.csv')

    def test_default_checkpoint_can_still_be_loaded_from_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'output').mkdir()
            self.checkpoint(folder/'output/best_val.pth.tar')
            args = self.args(folder, 'best_val.pth.tar')
            checkpoint = training.load_calibration_checkpoint(args)
            self.assertIn('state_dict', checkpoint)

    def test_calibration_saves_best_checkpoint_without_optional_test_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.checkpoint(folder/'source.pt')
            args = self.args(folder, folder/'source.pt')
            stats = {'nobg': [[.3]*3, [.2]*3], 'depth': [[.4]*3, [.1]*3]}
            (folder/'stats.json').write_text(json.dumps({'original.csv': stats}))
            with patch.object(training, 'ComposedModel', TinyModel), \
                    patch.object(training, 'TacForceDataset', return_value=[(0, 0, 0)]), \
                    patch.object(training, 'train', return_value={'loss': 1.}), \
                    patch.object(training, 'validate_network', return_value={'loss': .25}):
                training.train_regressor(args)
            saved = torch.load(Path(args.exp_dir)/'best_val.pth.tar', weights_only=True)
            self.assertEqual(saved['best_val_loss'], .25)
            self.assertEqual(saved['config']['labels_train'], 'train.csv')
            self.assertEqual(saved['config']['batch_size_per_gpu'], 4)
            self.assertEqual(json.loads((folder/'stats.json').read_text())['train.csv'], stats)
            log = json.loads((Path(args.exp_dir)/'log.txt').read_text())
            self.assertEqual(log['val_loss'], .25)
            self.assertNotIn('test_seen_loss', log)


if __name__ == '__main__':
    unittest.main()
