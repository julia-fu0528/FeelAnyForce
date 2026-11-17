import argparse
import os
import time
from pathlib import Path

import cv2

from sensors.gsrobotics import gsdevice, gs3drecon
from sensors import ATISensor

import torch
from torch import nn
from torchvision import transforms as pth_transforms
from PIL import Image
import numpy as np
import json

from args import get_parser
from composed_model import ComposedModel
from nano25_live import Nano25Reader, add_nano_arguments, comparison_panel
from force_history import ForceHistory
from evaluation_recording import EvaluationRecorder, EvaluationRecordingUI

class Predictor:
    def __init__(self):
        self.nano = None
        self.dev = None
        self.evaluation_recorder = None
        self.evaluation_ui = None
        self.evaluation_mouse_registered = False
        self.quit_requested = False
        parser = add_nano_arguments(get_parser())
        parser.add_argument('--eval-output-dir', type=Path, default=Path('evaluation'),
                            help='Parent folder for new live evaluation recordings (Record button or R).')
        self.args = parser.parse_args()
        ckpt = torch.load(self.args.checkpoint)
        if "config" in ckpt.keys():
            config = argparse.Namespace(**ckpt["config"])
            self.args = parser.parse_args(namespace=config)
        self.force_history = ForceHistory(self.args.nano_history_seconds) if self.args.nano25 else None

        # Load the model
        self.model = ComposedModel(self.args)

        self.model.cuda()
        self.model.eval()
        self.model.load_state_dict(ckpt["state_dict"])


        with open(self.args.dataset_stats) as convert_file:
            dataset_mean_std = json.load(convert_file)
        mean_rgb, std_rgb = dataset_mean_std[self.args.labels_train][self.args.tactile_mode]

        self.transform_rgb = pth_transforms.Compose([
            pth_transforms.Pad([0, 40, 0, 40]),
            pth_transforms.Resize(self.args.crop_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean_rgb, std_rgb),
        ])

        mean_depth = (0.234, 0.234, 0.234)
        std_depth = (0.138, 0.138, 0.138)
        self.transform_depth = pth_transforms.Compose([
            pth_transforms.Pad([0, 40, 0, 40]),
            pth_transforms.Resize(self.args.crop_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean_depth, std_depth),
        ])
        self.max_val_depth, self.min_val_depth = 27.03226967220051, -8.311515796355053
        self.dev, self.depth_nn = self.init_mini()
        try:
            if self.args.nano25:
                self.nano = Nano25Reader(self.args)
            self.reset_reference()
            if self.nano is not None:
                self.evaluation_recorder = EvaluationRecorder(self.args.eval_output_dir)
                self.evaluation_ui = EvaluationRecordingUI(self.evaluation_recorder)
        except BaseException:
            self.close()
            raise

    def reset_reference(self):
        self.finish_evaluation('re-zero')
        print(f'Keep the gel unloaded: zeroing depth and Nano over {self.depth_nn.zero_frames} camera frames.')
        if self.nano is not None:
            self.nano.wait_ready()
        previous_zero = self.depth_nn.dm_zero.copy()
        previous_counter = self.depth_nn.dm_zero_counter
        self.depth_nn.reset_zero()
        zero_samples = [] if self.nano is not None else None
        try:
            for _ in range(self.depth_nn.zero_frames):
                bg, _ = self.get_tacdepth(self.dev, self.depth_nn, zero_samples=zero_samples)
            new_bg = cv2.cvtColor(np.array(bg), cv2.COLOR_BGR2RGB)
            if self.nano is not None:
                self.nano.zero_from_frames(zero_samples)
        except BaseException:
            self.depth_nn.dm_zero = previous_zero
            self.depth_nn.dm_zero_counter = previous_counter
            raise
        self.bg = new_bg
        if self.force_history is not None:
            self.force_history.clear()
        print('Ready. R: record/stop evaluation; Z: unloaded re-zero; Q/Esc: quit.')

    def toggle_evaluation(self):
        if self.evaluation_recorder is None:
            return
        if self.evaluation_recorder.active:
            self.finish_evaluation('user')
            return
        mapping = self.args.nano_force_map
        axes = ','.join(('-' if sign < 0 else '+')+'xyz'[index] for index, sign in mapping) if mapping else 'native Nano axes (alignment unverified)'
        self.evaluation_recorder.start(dict(
            checkpoint=str(Path(self.args.checkpoint).resolve()),
            dataset_stats=str(Path(self.args.dataset_stats).resolve()),
            nano_force_map=axes, baseline_n=self.nano.baseline_n.tolist(),
            nano_max_gap_ms=self.args.nano_max_gap_ms))
        print(f'Evaluation recording started: {self.evaluation_recorder.directory.resolve()}')

    def finish_evaluation(self, reason):
        if self.evaluation_recorder is None or not self.evaluation_recorder.active:
            return
        report = self.evaluation_recorder.stop(reason)
        print(f'Evaluation saved: {self.evaluation_recorder.directory.resolve()}')
        print(f'Valid pairs: {report["valid_samples"]}; excluded: {report["invalid_samples"]}')
        if report['valid_samples']:
            axes = ', '.join(f'{axis}={value:.6g}' for axis, value in report['mse_per_axis_n2'].items())
            print(f'MSE (N^2): {axes}; overall={report["mse_overall_n2"]:.6g}')
        else:
            print('No valid paired samples; MSE is unavailable.')

    def close(self):
        try:
            self.finish_evaluation('shutdown')
        finally:
            try:
                if self.nano is not None:
                    self.nano.close()
            finally:
                if self.dev is not None and self.dev.cam is not None:
                    self.dev.cam.release()
                cv2.destroyAllWindows()

    def init_mini(self):
        dev = gsdevice.Camera("GelSight Mini")
        net_file_path = 'nnmini.pt'
        GPU = False
        dev.connect()

        ''' Load neural network '''
        net_path = os.path.join("sensors/gsrobotics", net_file_path)
        print('mini net path = ', net_path)

        if GPU:
            gpuorcpu = "cuda"
        else:
            gpuorcpu = "cpu"

        depth_nn = gs3drecon.Reconstruction3D(dev)
        net = depth_nn.load_nn(net_path, gpuorcpu)

        f0 = dev.get_raw_image()
        roi = (0, 0, f0.shape[1], f0.shape[0])

        return dev, depth_nn

    def get_tacdepth(self, dev, nn, *, zero_samples=None):
        for _ in range(1):
            f1 = dev.get_image()
            self.frame_monotonic = time.monotonic()
            if zero_samples is not None:
                # Match before depth computation can delay or evict the Nano sample.
                zero_samples.append(self.nano.unloaded_force(self.frame_monotonic))
            dm = nn.get_depthmap(f1, False)
        return f1, dm


    def eval(self):
        im, depth = self.get_tacdepth(self.dev, self.depth_nn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(Image.fromarray(im) - self.bg + 127)
        image = self.transform_rgb(im)

        force_vector = torch.FloatTensor(self.ft_sensor.get_ft())[:3]

        image = image.cuda(non_blocking=True)
        force_vector = force_vector.cuda(non_blocking=True)

        with torch.no_grad():
            intermediate_output_rgb = self.model.tactile_backbone(image.unsqueeze(0))
            intermediate_output = intermediate_output_rgb
            output = self.model.regressor(intermediate_output)

        loss = torch.sqrt(nn.MSELoss()(output, force_vector))
        loss = loss.cpu()
        force_vector = force_vector.cpu()
        output = output.cpu()
        cv2.imshow('image_rgb_check.png', np.array(im))
        cv2.waitKey(1)
        print(f"Force predicted: {output}, ground truth {force_vector}, error = {loss}")
        return output, force_vector, im, depth

    def predict(self):
        im, depth = self.get_tacdepth(self.dev, self.depth_nn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(Image.fromarray(im) - self.bg + 127)
        image = self.transform_rgb(im)
        image = image.cuda(non_blocking=True)

        with torch.no_grad():
            intermediate_output_rgb = self.model.tactile_backbone(image.unsqueeze(0))
            intermediate_output = intermediate_output_rgb
            output = self.model.regressor(intermediate_output)

        output = output.cpu()
        if self.nano is not None:
            measured = self.nano.match(self.frame_monotonic)
            if self.evaluation_recorder is not None:
                self.evaluation_recorder.append(self.frame_monotonic, output.numpy(), measured)
            self.force_history.append(self.frame_monotonic, output.numpy(), measured['force_n'])
            panel, _ = comparison_panel(np.array(im), output.numpy(), measured,
                                        mapping=self.args.nano_force_map, history=self.force_history)
            if self.evaluation_ui is not None:
                panel = self.evaluation_ui.render(panel)
            cv2.imshow('FeelAnyForce + Nano25', panel)
            if self.evaluation_ui is not None and not self.evaluation_mouse_registered:
                cv2.setMouseCallback('FeelAnyForce + Nano25', self.evaluation_ui.on_mouse)
                self.evaluation_mouse_registered = True
        else:
            cv2.imshow('image_rgb_check', np.array(im))
            print(f"Force predicted: {output}")
        key = cv2.waitKey(1) & 0xff
        window_closed = False
        toggle_requested = False
        if self.evaluation_ui is not None:
            toggle_requested = self.evaluation_ui.take_toggle()
            try:
                window_closed = cv2.getWindowProperty('FeelAnyForce + Nano25', cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                window_closed = True
        if key in (ord('q'), ord('Q'), 27) or window_closed:
            self.finish_evaluation('quit')
            self.quit_requested = True
        elif key in (ord('z'), ord('Z')):
            self.reset_reference()
        elif key in (ord('r'), ord('R')) or toggle_requested:
            self.toggle_evaluation()
        return output, im, depth


if __name__ == '__main__':
    p = Predictor()
    try:
        while not p.quit_requested:
            p.predict()
    except KeyboardInterrupt:
        pass
    finally:
        p.close()
