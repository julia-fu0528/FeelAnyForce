"""Bounded, camera-timestamped force histories and lightweight OpenCV plots."""
from collections import deque
import math

import numpy as np


def history_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 1 <= seconds <= 120:
        raise ValueError('History window must be between 1 and 120 seconds')
    return seconds


class ForceHistory:
    def __init__(self, window_seconds=10., max_samples=10000):
        self.window_seconds = history_seconds(window_seconds)
        if not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError('History must hold at least two samples')
        self.samples = deque(maxlen=max_samples)

    def append(self, camera_time, prediction, ground_truth):
        if not math.isfinite(camera_time) or (self.samples and camera_time <= self.samples[-1][0]):
            raise ValueError('Camera timestamps must be finite and strictly increasing')
        pred = np.asarray(prediction, dtype=float).reshape(3).copy()
        gt = (np.full(3, np.nan) if ground_truth is None else
              np.asarray(ground_truth, dtype=float).reshape(3).copy())
        pred[~np.isfinite(pred)] = np.nan
        gt[~np.isfinite(gt)] = np.nan
        self.samples.append((camera_time, pred, gt))
        cutoff = camera_time-self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def clear(self):
        self.samples.clear()

    def arrays(self):
        if not self.samples:
            return np.empty(0), np.empty((0, 3)), np.empty((0, 3))
        times = np.array([row[0] for row in self.samples])-self.samples[-1][0]
        return times, np.stack([row[1] for row in self.samples]), np.stack([row[2] for row in self.samples])


def trace_segments(times, values, max_gap=1.):
    """Split at unavailable samples and long capture gaps; never interpolate them."""
    segments, current = [], []
    previous = None
    for t, value in zip(times, values):
        valid = math.isfinite(t) and math.isfinite(value)
        if not valid or (previous is not None and t-previous > max_gap):
            if current:
                segments.append(np.asarray(current))
                current = []
        if valid:
            current.append((t, value))
        previous = t if valid else None
    if current:
        segments.append(np.asarray(current))
    return segments


def force_limits(prediction, ground_truth):
    """One shared scale per axis, covering both series and zero."""
    values = np.concatenate((np.asarray(prediction).ravel(), np.asarray(ground_truth).ravel(), [0.]))
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())
    middle, span = (low+high)/2, max(high-low, .2)
    return middle-.58*span, middle+.58*span


def draw_history(history, width=780, height=660, *, gt_only=False):
    import cv2

    panel = np.full((height, width, 3), (29, 26, 23), np.uint8)
    pred_color, gt_color = (255, 180, 60), (60, 170, 255)
    text_color, grid_color = (225, 225, 225), (62, 57, 51)

    def text(label, position, color=text_color, scale=.48):
        cv2.putText(panel, label, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text('Force history (N)', (16, 26), scale=.6)
    legend = ((450, 'Nano25', gt_color),) if gt_only else ((260, 'Predicted', pred_color), (450, 'Nano25 GT', gt_color))
    for x, label, color in legend:
        cv2.line(panel, (x, 20), (x+30, 20), color, 2, cv2.LINE_AA)
        text(label, (x+40, 26), color)
    subtitle = 'Live Nano measurements matched to camera frames' if gt_only else 'Matched samples per camera frame | common scale for each pair'
    text(subtitle, (16, 51), scale=.43)
    times, prediction, gt = history.arrays()
    left, right = 76, width-22
    row_height = (height-108)//3
    for axis, name in enumerate(('Fx', 'Fy', 'Fz')):
        top = 78+axis*row_height
        bottom = top+row_height-32
        low, high = force_limits(prediction[:, axis], gt[:, axis])
        text(name, (14, top-12), scale=.6)

        def point(t, value):
            x = left+(t+history.window_seconds)/history.window_seconds*(right-left)
            y = bottom-(value-low)/(high-low)*(bottom-top)
            return int(round(x)), int(round(y))

        for value in (low, (low+high)/2, high):
            y = point(0., value)[1]
            cv2.line(panel, (left, y), (right, y), grid_color, 1)
            text(f'{value:.3g}', (14, y+5), scale=.39)
        zero = point(0., 0.)[1]
        cv2.line(panel, (left, zero), (right, zero), (100, 91, 79), 1)
        for t in np.linspace(-history.window_seconds, 0., 5):
            x = point(t, 0.)[0]
            cv2.line(panel, (x, top), (x, bottom), grid_color, 1)
            if axis == 2:
                text(f'{t:g}', (x-12, bottom+20), scale=.4)
        # GT has a thinner line and point markers to distinguish coincident curves.
        for values, color, thickness in ((prediction[:, axis], pred_color, 2), (gt[:, axis], gt_color, 1)):
            for segment in trace_segments(times, values):
                points = np.array([point(t, value) for t, value in segment], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(panel, [points], False, color, thickness, cv2.LINE_AA)
                if len(points) == 1 or thickness == 1:
                    # Sparse markers leave overlapping prediction lines visible.
                    for p in points[::max(1, len(points)//30)]:
                        cv2.circle(panel, tuple(p), 2, color, 1, cv2.LINE_AA)
    text(f'Time relative to latest camera frame (s) | last {history.window_seconds:g} s',
         (76, height-24), scale=.43)
    return panel
