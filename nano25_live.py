"""Optional live Nano25 comparison using tac_calib's established Net F/T driver."""
import importlib.util
import ipaddress
import math
from pathlib import Path
import sys
import time

import numpy as np
from force_history import draw_history, history_seconds


def force_scale(unit):
    scales = {'n': 1., 'kn': 1000., 'lbf': 4.4482216152605,
              'klbf': 4448.2216152605, 'kgf': 9.80665}
    key = unit.strip().lower()
    if key not in scales:
        raise ValueError(f'Unsupported ATI force unit: {unit!r}')
    return scales[key]


def parse_force_map(value):
    """Signed Nano axes in model Fx,Fy,Fz order, e.g. -y,x,z."""
    parts = [part.strip().lower() for part in value.split(',')]
    valid = {'x', 'y', 'z', '+x', '+y', '+z', '-x', '-y', '-z'}
    if len(parts) != 3 or any(part not in valid for part in parts):
        raise ValueError('Use three signed axes, e.g. x,y,z or -y,x,z')
    mapping = tuple(('xyz'.index(part[-1]), -1 if part.startswith('-') else 1) for part in parts)
    if len({index for index, _ in mapping}) != 3:
        raise ValueError('Each Nano axis must appear exactly once')
    return mapping


def baseline_from_samples(samples, *, since, now, scale, count=10):
    seen, values = set(), []
    for sample in samples:
        if (sample.ft_sequence in seen or not since < sample.monotonic <= now
                or now-sample.monotonic > .5 or sample.status
                or not np.isfinite(sample.values).all()):
            continue
        seen.add(sample.ft_sequence)
        values.append(sample.values[:3])
    if len(values) < count:
        raise ValueError(f'Waiting for {count} fresh unloaded Nano samples ({len(values)} available)')
    return np.mean(values[-count:], axis=0) * scale


def match_force(samples, camera_time, baseline_n, scale, mapping=None, *, max_gap=.05, error=''):
    sample = min(samples, key=lambda s: abs(s.monotonic-camera_time)) if samples else None
    delta = sample.monotonic-camera_time if sample is not None else None
    reason = error
    if not reason:
        if sample is None:
            reason = 'No F/T sample'
        elif sample.status:
            reason = f'Nano sensor status 0x{sample.status:08X}'
        elif not np.isfinite(sample.values).all():
            reason = 'Nonfinite Nano reading'
        elif abs(delta) > max_gap:
            reason = f'No F/T match within {max_gap*1000:g} ms'
    force = None
    if not reason:
        force = np.asarray(sample.values[:3])*scale - baseline_n
        if mapping is not None:
            force = np.array([sign*force[index] for index, sign in mapping])
    return dict(force_n=force, delta_seconds=delta, status=reason or 'OK')


def add_nano_arguments(parser):
    parser.add_argument('--nano25', action='store_true', help='Show live Nano25 ground truth beside predicted forces')
    parser.add_argument('--nano-ip', default='192.168.1.1', help='ATI Net F/T IP address')
    parser.add_argument('--nano-bind', default=None, help='Local IP of the Nano-connected adapter')
    parser.add_argument('--tac-calib-root', type=Path, default=Path(__file__).resolve().parent.parent/'tac_calib',
                        help='Path to tac_calib containing ft_test/ati_live.py')
    parser.add_argument('--nano-force-map', type=parse_force_map, default=None,
                        help='Verified Nano axes in model Fx,Fy,Fz order, e.g. --nano-force-map=-y,x,z')
    parser.add_argument('--nano-max-gap-ms', type=float, default=50., help='Maximum host timestamp matching gap')
    parser.add_argument('--nano-history-seconds', type=history_seconds, default=10.,
                        help='Scrolling force plot window, 1–120 seconds (default: 10)')
    return parser


class Nano25Reader:
    def __init__(self, args):
        ipaddress.IPv4Address(args.nano_ip)
        if args.nano_bind:
            ipaddress.IPv4Address(args.nano_bind)
        if not math.isfinite(args.nano_max_gap_ms) or args.nano_max_gap_ms <= 0:
            raise ValueError('--nano-max-gap-ms must be finite and positive')
        path = Path(args.tac_calib_root)/'ft_test/ati_live.py'
        if not path.is_file():
            raise ValueError(f'Nano25 driver missing: {path}. Set --tac-calib-root.')
        spec = importlib.util.spec_from_file_location('_feelanyforce_netft', path)
        driver = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = driver  # dataclasses resolves its module during import.
        spec.loader.exec_module(driver)
        calibration = driver.read_calibration(args.nano_ip, args.nano_bind)
        self.scale = force_scale(calibration.force_unit)
        self.mapping, self.max_gap = args.nano_force_map, args.nano_max_gap_ms/1000
        source = driver.NetFT(args.nano_ip, calibration, args.nano_bind)
        try:
            self.acquisition = driver.Acquisition(source, calibration, 100, 10)
            self.acquisition.thread.start()
        except Exception:
            source.close()
            raise
        self.baseline_n = None
        print(f'Nano25 {calibration.serial}: {calibration.force_unit} converted to N')

    def unloaded_force(self, camera_time):
        """Freeze an un-tared force sample for one unloaded camera frame, in N."""
        samples, error, _ = self.acquisition.snapshot()
        matched = match_force(samples, camera_time, np.zeros(3), self.scale,
                              max_gap=self.max_gap, error=error)
        if matched['force_n'] is None:
            raise RuntimeError(f"Cannot zero Nano25: {matched['status']}")
        return matched['force_n']

    def wait_ready(self, timeout=5.):
        start = time.monotonic()
        reason = 'No fresh Nano25 reading'
        while time.monotonic()-start < timeout:
            try:
                self.unloaded_force(time.monotonic())
                return
            except RuntimeError as exc:
                reason = str(exc)
            time.sleep(.01)
        raise RuntimeError(reason)

    def zero_from_frames(self, forces_n):
        """Commit the mean of the samples frozen during the depth-zero window."""
        values = np.asarray(forces_n, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Nano zero needs finite three-axis samples from unloaded frames')
        baseline = values.mean(axis=0)
        if not np.isfinite(baseline).all():
            raise ValueError('Nano zero produced a nonfinite baseline')
        self.baseline_n = baseline
        print(f'Nano25 zeroed from {len(values)} camera-matched frames; baseline (N): {baseline}')

    def zero(self, timeout=5.):
        since = time.monotonic()
        reason = 'Waiting for Nano25'
        while time.monotonic()-since < timeout:
            samples, error, _ = self.acquisition.snapshot()
            if error:
                reason = error
            else:
                try:
                    self.baseline_n = baseline_from_samples(samples, since=since,
                                                            now=time.monotonic(), scale=self.scale)
                    return
                except ValueError as exc:
                    reason = str(exc)
            time.sleep(.01)
        raise RuntimeError(f'Cannot zero Nano25: {reason}')

    def match(self, camera_time):
        if self.baseline_n is None:
            raise RuntimeError('Zero Nano25 while unloaded before comparison')
        samples, error, _ = self.acquisition.snapshot()
        return match_force(samples, camera_time, self.baseline_n, self.scale, self.mapping,
                           max_gap=self.max_gap, error=error)

    def close(self):
        self.acquisition.close()


def comparison_panel(rgb, prediction, measured, mapping=None, *, history=None):
    """Return an OpenCV BGR panel and the exact displayed text for logging/tests."""
    import cv2

    prediction = np.asarray(prediction).reshape(3)
    gt = measured['force_n']
    lines = ['FeelAnyForce prediction / Nano25 ground truth (N)']
    if mapping is None:
        lines.append('Axes UNVERIFIED: model predictions vs native Nano axes')
    else:
        labels = ','.join(('-' if sign < 0 else '+') + 'xyz'[index] for index, sign in mapping)
        lines.append(f'GT mapping to model axes: {labels}')
    lines.append('Axis       Predicted        Nano GT' + ('         Pred-GT' if mapping is not None else ''))
    for index, axis in enumerate(('Fx', 'Fy', 'Fz')):
        value = f'{gt[index]:+10.3f}' if gt is not None else '       ---'
        row = f'{axis}         {prediction[index]:+10.3f}      {value}'
        if mapping is not None:
            row += f'      {prediction[index]-gt[index]:+10.3f}' if gt is not None else '             ---'
        lines.append(row)
    delta = measured['delta_seconds']
    timing = f' | host match {delta*1000:+.1f} ms' if delta is not None else ''
    lines += [measured['status'] + timing, 'Z: unloaded reference + Nano zero   Q/Esc: quit']
    width = 500 if history is not None else 940
    image_height = 300 if history is not None else 400
    scale = min(width/rgb.shape[1], image_height/rgb.shape[0])
    image = cv2.resize(rgb, (max(1, round(rgb.shape[1]*scale)), max(1, round(rgb.shape[0]*scale))))
    panel = np.full((image.shape[0]+len(lines)*30+20, width, 3), 24, np.uint8)
    left = (width-image.shape[1])//2
    panel[:image.shape[0], left:left+image.shape[1]] = image[..., ::-1]
    text_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1., 1)[0][0] for line in lines)
    font_scale = min(.53, (width-24)/text_width)
    for i, line in enumerate(lines):
        color = (80, 200, 255) if i == 1 or (i == 6 and gt is None) else (230, 230, 230)
        cv2.putText(panel, line, (12, image.shape[0]+28+i*30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)
    if history is not None:
        plots = draw_history(history)
        combined = np.full((max(panel.shape[0], plots.shape[0]), width+plots.shape[1], 3), 24, np.uint8)
        combined[:panel.shape[0], :width] = panel
        combined[:plots.shape[0], width:] = plots
        panel = combined
    return panel, lines
