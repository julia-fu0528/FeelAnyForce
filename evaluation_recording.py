"""Record live predicted/GT force pairs and calculate unshifted, per-frame MSE."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile

import cv2
import numpy as np


AXES = ('Fx', 'Fy', 'Fz')
PRED_COLUMNS = ('pred_fx_n', 'pred_fy_n', 'pred_fz_n')
GT_COLUMNS = ('gt_fx_n', 'gt_fy_n', 'gt_fz_n')
FIELDS = ('camera_monotonic_s', 'nano_monotonic_s', 'host_match_ms',
          *PRED_COLUMNS, *GT_COLUMNS, 'valid', 'status')


class ErrorMetrics:
    def __init__(self):
        self.total = self.valid = 0
        self.squared_error = np.zeros(3, dtype=np.float64)

    def add(self, prediction, ground_truth, valid):
        self.total += 1
        if valid:
            self.squared_error += np.square(prediction-ground_truth)
            self.valid += 1

    def report(self):
        mse = self.squared_error/self.valid if self.valid else None
        return dict(total_samples=self.total, valid_samples=self.valid,
                    invalid_samples=self.total-self.valid,
                    mse_per_axis_n2=dict(zip(AXES, mse.tolist() if mse is not None else [None]*3)),
                    mse_overall_n2=float(mse.mean()) if mse is not None else None)


class EvaluationRecorder:
    def __init__(self, output_root='evaluation'):
        self.output_root = Path(output_root)
        self.directory = None
        self.stream = None
        self.metrics = ErrorMetrics()
        self.last_report = None

    @property
    def active(self):
        return self.stream is not None

    def start(self, metadata):
        if self.active:
            raise RuntimeError('An evaluation recording is already active')
        self.output_root.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().strftime('%Y%m%d_%H%M%S_')
        self.directory = Path(tempfile.mkdtemp(prefix=prefix, dir=self.output_root))
        details = dict(metadata, started_utc=datetime.now(timezone.utc).isoformat(),
                       force_unit='N', mse_unit='N^2',
                       pairing='Nearest host timestamp per camera frame; no lag compensation',
                       mse_definition='Mean squared component error over valid recorded frames; overall is the mean of Fx, Fy, Fz MSE')
        (self.directory/'metadata.json').write_text(json.dumps(details, indent=2, allow_nan=False)+'\n')
        stream = (self.directory/'pairs.csv').open('x', newline='')
        try:
            self.writer = csv.writer(stream)
            self.writer.writerow(FIELDS)
            stream.flush()
        except BaseException:
            stream.close()
            raise
        self.stream = stream
        self.metrics = ErrorMetrics()
        self.last_report = None

    def append(self, camera_time, prediction, measured):
        if not self.active:
            return
        prediction = np.asarray(prediction, dtype=float).reshape(3)
        gt = (np.full(3, np.nan) if measured['force_n'] is None
              else np.asarray(measured['force_n'], dtype=float).reshape(3))
        delta = measured['delta_seconds']
        reason = measured['status'] if measured['status'] != 'OK' else ''
        if not reason and (not np.isfinite(prediction).all() or not np.isfinite(gt).all()):
            reason = 'Missing or nonfinite prediction/GT'
        if not reason and (not math.isfinite(camera_time) or delta is None or not math.isfinite(delta)):
            reason = 'Missing or nonfinite host timestamp'
        valid = not reason
        nano_time = camera_time+delta if delta is not None else math.nan
        match_ms = delta*1000 if delta is not None else math.nan
        values = [camera_time, nano_time, match_ms, *prediction, *gt]
        self.writer.writerow([value if math.isfinite(value) else '' for value in values]
                             + [int(valid), reason or 'OK'])
        self.stream.flush()
        self.metrics.add(prediction, gt, valid)

    def stop(self, reason='user'):
        if not self.active:
            return self.last_report
        try:
            self.stream.close()
        finally:
            self.stream = None
        self.last_report = dict(self.metrics.report(), stop_reason=reason,
                                ended_utc=datetime.now(timezone.utc).isoformat())
        temporary = self.directory/'metrics.json.tmp'
        temporary.write_text(json.dumps(self.last_report, indent=2, allow_nan=False)+'\n')
        temporary.replace(self.directory/'metrics.json')
        return self.last_report


def summarize_csv(path):
    """Recompute MSE from a saved pairs.csv; invalid rows remain excluded."""
    metrics = ErrorMetrics()
    with Path(path).open(newline='') as stream:
        rows = csv.DictReader(stream)
        required = {*PRED_COLUMNS, *GT_COLUMNS, 'valid'}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError('Expected a pairs.csv written by the live evaluation recorder')
        for row in rows:
            prediction = np.array([float(row[key]) if row[key] else np.nan for key in PRED_COLUMNS])
            gt = np.array([float(row[key]) if row[key] else np.nan for key in GT_COLUMNS])
            valid = row['valid'] == '1' and np.isfinite(prediction).all() and np.isfinite(gt).all()
            metrics.add(prediction, gt, valid)
    return metrics.report()


class EvaluationRecordingUI:
    def __init__(self, recorder):
        self.recorder = recorder
        self.button = (0, 0, 0, 0)
        self.toggle_requested = False

    def on_mouse(self, event, x, y, flags, param):
        left, top, width, height = self.button
        if event == cv2.EVENT_LBUTTONUP and left <= x < left+width and top <= y < top+height:
            self.toggle_requested = True

    def take_toggle(self):
        requested = self.toggle_requested
        self.toggle_requested = False
        return requested

    def render(self, panel):
        height, width = panel.shape[:2]
        result = np.full((height+96, width, 3), 24, np.uint8)
        result[:height] = panel
        self.button = (16, height+12, 180, 46)
        color = (60, 60, 155) if self.recorder.active else (70, 90, 60)
        cv2.rectangle(result, (16, height+12), (196, height+58), color, -1)

        def text(label, x, y, scale=.52):
            cv2.putText(result, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (235, 235, 235), 1, cv2.LINE_AA)

        text('Stop / MSE (R)' if self.recorder.active else 'Record (R)', 27, height+41, .56)
        report = self.recorder.metrics.report()
        state = 'RECORDING' if self.recorder.active else ('SAVED' if self.recorder.directory else 'READY')
        text(f'{state} | Valid pairs: {report["valid_samples"]} | Excluded: {report["invalid_samples"]}', 220, height+24)
        if report['valid_samples']:
            axes = '   '.join(f'{axis}: {report["mse_per_axis_n2"][axis]:.5g}' for axis in AXES)
            text(f'MSE (N^2)   {axes}   Overall: {report["mse_overall_n2"]:.5g}', 220, height+49)
        else:
            text('Record paired predictions and Nano GT to calculate MSE.', 220, height+49)
        if self.recorder.directory:
            text(f'Run: {self.recorder.directory.name} | pairs.csv + metrics.json on stop', 220, height+76, .46)
        text('New folder per recording', 16, height+81, .4)
        return result


def main():
    parser = argparse.ArgumentParser(description='Calculate per-axis and overall force MSE from a live evaluation pairs.csv.')
    parser.add_argument('csv_path', type=Path)
    args = parser.parse_args()
    try:
        report = summarize_csv(args.csv_path)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
