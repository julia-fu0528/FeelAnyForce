"""Live calibration controls and an OpenCV Nano25 force display."""
from collections import deque

import cv2
import numpy as np

from force_history import ForceHistory, draw_history


class RecordingControls:
    zero_frames = 10

    def __init__(self, reader, target, interval):
        self.reader, self.target, self.interval = reader, target, interval
        self.count = 0
        self.recording = False
        self.quit = False
        self.next_record = 0.
        self.zero_samples = None
        self.zero_event = None
        self.zero_generation = 0
        self.force = None
        self.message = 'Paused. Zero while unloaded, then press Start.'

    def command(self, action):
        if action == 'quit':
            self.quit = True
        elif action == 'zero' and self.count < self.target:
            self.recording = False
            self.zero_samples = []
            self.message = 'Zeroing 0/10. Keep the gel unloaded.'
        elif action == 'toggle' and self.zero_samples is None and self.count < self.target:
            self.recording = not self.recording
            self.next_record = 0.
            self.message = 'Recording. Hold contacts steady.' if self.recording else 'Paused.'

    def consume(self, camera_time, raw_force, error=''):
        """Consume one new camera-matched reading; return whether to save this frame."""
        self.zero_event = None
        if raw_force is not None:
            raw_force = np.asarray(raw_force, dtype=float)
        if (error or raw_force is None or raw_force.shape != (3,)
                or not np.isfinite(raw_force).all()):
            was_zeroing = self.zero_samples is not None
            self.recording = False
            self.zero_samples = None
            self.force = None
            self.message = ('Zero cancelled; previous baseline kept. ' if was_zeroing else 'Paused. ')
            self.message += error or 'No valid Nano force sample.'
            return False

        was_zeroing = self.zero_samples is not None
        if was_zeroing:
            self.zero_samples.append(raw_force.copy())
            self.message = f'Zeroing {len(self.zero_samples)}/10. Keep the gel unloaded.'
            if len(self.zero_samples) == self.zero_frames:
                self.reader.zero_from_frames(self.zero_samples)
                self.zero_samples = None
                self.zero_generation += 1
                self.zero_event = dict(first_record_index=self.count,
                                       camera_monotonic=camera_time,
                                       zero_frames=self.zero_frames,
                                       baseline_n=self.reader.baseline_n.tolist())
                self.message = 'Zero complete (10/10). Press Start to record.'

        self.force = raw_force - self.reader.baseline_n
        if self.reader.mapping is not None:
            self.force = np.array([sign*self.force[index] for index, sign in self.reader.mapping])
        return (not was_zeroing and self.recording and self.count < self.target
                and camera_time >= self.next_record)

    def saved(self, camera_time):
        self.count += 1
        self.next_record = camera_time + self.interval
        if self.count == self.target:
            self.recording = False
            self.message = 'Recording complete. Q/Esc or Quit to close.'


class CalibrationUI:
    title = 'FeelAnyForce calibration | Nano25'

    def __init__(self, controls):
        self.controls = controls
        self.history = ForceHistory()
        self.zero_generation = controls.zero_generation
        self.actions = deque()
        self.buttons = {'toggle': (16, 585, 145, 46),
                        'zero': (177, 585, 145, 46),
                        'quit': (338, 585, 145, 46)}

    def open(self):
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.title, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            for action, (left, top, width, height) in self.buttons.items():
                if left <= x < left+width and top <= y < top+height:
                    self.actions.append(action)

    def render(self, tactile, camera_time, delta_seconds):
        controls = self.controls
        if self.zero_generation != controls.zero_generation:
            self.history.clear()
            self.zero_generation = controls.zero_generation
        self.history.append(camera_time, np.full(3, np.nan), controls.force)
        panel = np.full((660, 1280, 3), (29, 26, 23), np.uint8)

        def text(label, x, y, color=(230, 230, 230), scale=.52):
            cv2.putText(panel, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

        text('Calibration collection', 16, 28, scale=.7)
        image = cv2.resize(tactile, (400, 300))
        panel[42:342, 50:450] = image
        text(f'Saved {controls.count} / {controls.target}', 16, 369, scale=.65)
        state = 'ZEROING' if controls.zero_samples is not None else ('RECORDING' if controls.recording else 'PAUSED')
        if controls.count == controls.target:
            state = 'COMPLETE'
        text(state, 310, 369, (80, 210, 255))
        mapping = controls.reader.mapping
        axes = ', '.join(('-' if sign < 0 else '+')+'xyz'[index] for index, sign in mapping) if mapping else 'native x, y, z'
        text(f'Label Fx, Fy, Fz = Nano {axes}', 16, 397, scale=.48)
        for i, axis in enumerate(('Fx', 'Fy', 'Fz')):
            value = '---' if controls.force is None else f'{controls.force[i]:+.3f} N'
            text(f'{axis}: {value}', 16+i*162, 430, (60, 180, 255), scale=.57)
        timing = 'No host match' if delta_seconds is None else f'Host match {delta_seconds*1000:+.1f} ms'
        text(timing, 16, 462, scale=.48)
        # Wrap status messages without hiding faults behind the graph.
        words, lines = controls.message.split(), ['']
        for word in words:
            if len(lines[-1])+len(word)+1 > 58:
                lines.append('')
            lines[-1] += (' ' if lines[-1] else '') + word
        for i, line in enumerate(lines[:3]):
            text(line, 16, 493+i*21, (80, 210, 255), scale=.45)
        text('Zero: next 10 frames; Nano forces only.', 16, 567, scale=.48)
        for action, (x, y, width, height) in self.buttons.items():
            cv2.rectangle(panel, (x, y), (x+width, y+height), (65, 65, 65), -1)
            label = {'toggle': 'Pause' if controls.recording else 'Start', 'zero': 'Zero (Z)', 'quit': 'Quit (Q)'}[action]
            text(label, x+18, y+29, scale=.62)
        text('Space: start/pause   Z: zero   Q/Esc: quit', 16, 651, scale=.45)
        panel[:, 500:] = draw_history(self.history, gt_only=True)
        return panel

    def show(self, tactile, camera_time, delta_seconds):
        cv2.imshow(self.title, self.render(tactile, camera_time, delta_seconds))
        key = cv2.waitKey(1) & 0xff
        action = {ord(' '): 'toggle', ord('z'): 'zero', ord('Z'): 'zero',
                  ord('q'): 'quit', ord('Q'): 'quit', 27: 'quit'}.get(key)
        if action:
            self.actions.append(action)
        try:
            visible = cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if not visible:
            self.actions.append('quit')
        actions = list(self.actions)
        self.actions.clear()
        return actions

    def close(self):
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass  # The user may already have closed the window.
