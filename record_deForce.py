"""Record GelSight frames with serial ATI or Ethernet Nano25 force labels."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import click
import cv2
import numpy as np

from nano25_live import Nano25Reader, match_force, parse_force_map
from sensors import ATISensor
from sensors.gsrobotics import gsdevice, gs3drecon


def init_mini():
    dev = gsdevice.Camera('GelSight Mini')
    try:
        dev.connect()
        net_path = Path(__file__).resolve().parent/'sensors/gsrobotics/nnmini.pt'
        nn = gs3drecon.Reconstruction3D(dev)
        nn.load_nn(str(net_path), 'cpu')
        return dev, nn
    except BaseException:
        if getattr(dev, 'cam', None) is not None:
            dev.cam.release()
        raise


def capture_frame(dev, nn, read_force=None, *, discard_frames=0):
    for _ in range(discard_frames):
        dev.get_image()
    tactile = dev.get_image()
    camera_time = time.monotonic()
    # Freeze the force before depth reconstruction can delay the match.
    force = read_force(camera_time) if read_force is not None else None
    depth = nn.get_depthmap(tactile, False)
    return tactile, depth, force


def matched_force(reader, camera_time):
    measured = reader.match(camera_time)
    if measured['force_n'] is None:
        raise RuntimeError(f"Cannot record Nano25: {measured['status']}")
    return measured['force_n']


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not save image: {path}')


def record_interactively(dev, nn, reader, record_count, interval, save_record, save_zero):
    from calibration_ui import CalibrationUI, RecordingControls

    controls = RecordingControls(reader, record_count, interval)
    ui = CalibrationUI(controls)
    try:
        ui.open()
        while not controls.quit:
            tactile = dev.get_image()
            camera_time = time.monotonic()
            if tactile is None:
                raise RuntimeError('Cannot read GelSight camera frame')
            # Freeze one native, un-tared reading. Both the preview and saved label
            # derive from this same reading, even if depth reconstruction is slow.
            samples, error, _ = reader.acquisition.snapshot()
            measured = match_force(samples, camera_time, np.zeros(3), reader.scale,
                                   max_gap=reader.max_gap, error=error)
            ready = controls.consume(camera_time, measured['force_n'],
                                     '' if measured['status'] == 'OK' else measured['status'])
            if controls.zero_event is not None:
                save_zero(controls.zero_event)
            if ready:
                depth = nn.get_depthmap(tactile, False)
                save_record(controls.count, tactile, depth, controls.force)
                controls.saved(time.monotonic())
            for action in ui.show(tactile, camera_time, measured['delta_seconds']):
                controls.command(action)
    finally:
        ui.close()


@click.command()
@click.option('-c', '--record_count', type=click.IntRange(min=1), default=1000, show_default=True)
@click.option('-i', '--interval', type=click.FloatRange(min=0), default=1., show_default=True)
@click.option('--nano25', is_flag=True, help='Record Ethernet Nano25 forces instead of serial ATI.')
@click.option('--ui', is_flag=True, help='Live Nano25 preview with Start/Pause and a 10-frame force Zero button.')
@click.option('--nano-ip', default='192.168.1.1', show_default=True)
@click.option('--nano-bind', default=None, help='Local IP of the Nano-connected adapter.')
@click.option('--tac-calib-root', type=click.Path(path_type=Path),
              default=Path(__file__).resolve().parent.parent/'tac_calib', show_default=True)
@click.option('--nano-force-map', default=None,
              help='Verified Nano axes in label Fx,Fy,Fz order, e.g. --nano-force-map=-y,x,z.')
@click.option('--nano-max-gap-ms', type=float, default=50., show_default=True)
@click.option('--output-dir', type=click.Path(path_type=Path), default=None,
              help='New recording folder; default: dataset/calibration_sphere/<timestamp>.')
def main(record_count, interval, nano25, ui, nano_ip, nano_bind, tac_calib_root,
         nano_force_map, nano_max_gap_ms, output_dir):
    if ui and not nano25:
        raise click.UsageError('--ui requires --nano25.')
    if not math.isfinite(interval):
        raise click.BadParameter('must be finite', param_hint='--interval')
    if not math.isfinite(nano_max_gap_ms) or nano_max_gap_ms <= 0:
        raise click.BadParameter('must be finite and positive', param_hint='--nano-max-gap-ms')
    try:
        mapping = parse_force_map(nano_force_map) if nano_force_map is not None else None
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint='--nano-force-map') from exc
    base_dir = output_dir or Path('dataset/calibration_sphere')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if base_dir.exists():
        raise click.ClickException(f'Output folder already exists: {base_dir}. Choose a new --output-dir.')
    args = SimpleNamespace(nano_ip=nano_ip, nano_bind=nano_bind, tac_calib_root=tac_calib_root,
                           nano_force_map=mapping, nano_max_gap_ms=nano_max_gap_ms)
    reader = dev = None
    try:
        reader = Nano25Reader(args) if nano25 else ATISensor.Receiver()
        if nano25:
            reader.wait_ready()
        else:
            reader.tare()
        dev, nn = init_mini()
        nn.reset_zero()
        click.echo(f'Keep the gel unloaded: zeroing over {nn.zero_frames} camera frames.')
        zero_samples = []
        for _ in range(nn.zero_frames):
            background, _, force = capture_frame(dev, nn, reader.unloaded_force if nano25 else None)
            if nano25:
                zero_samples.append(force)
        if nano25:
            reader.zero_from_frames(zero_samples)
        base_dir.mkdir(parents=True, exist_ok=False)
        (base_dir/'tactile').mkdir()
        (base_dir/'depth').mkdir()
        write_image(base_dir/'background.png', background)
        metadata = dict(sensor='nano25_ethernet' if nano25 else 'ati_serial',
                        force_unit='N' if nano25 else 'sensor-configured',
                        force_axes=nano_force_map or 'native sensor axes (alignment unverified)',
                        zero_frames=nn.zero_frames)
        if nano25:
            metadata.update(nano_ip=nano_ip, nano_bind=nano_bind,
                            nano_max_gap_ms=nano_max_gap_ms,
                            baseline_n=reader.baseline_n.tolist())
            metadata['zero_events'] = [dict(first_record_index=0, zero_frames=nn.zero_frames,
                                           baseline_n=reader.baseline_n.tolist(),
                                           source='startup')]
        (base_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        csv_path = base_dir/'calibration.csv'
        click.echo(f'Ready. Output: {csv_path}; Ctrl+C stops and keeps completed records.')
        if ui:
            click.echo('Preview paused. Zero/Z: average next 10 frames (Nano only). Start/Space: record.')
        if nano25 and mapping is None:
            click.echo('Labels use native Nano axes; alignment with model axes is unverified.')
        read_force = (lambda stamp: matched_force(reader, stamp)) if nano25 else (lambda stamp: reader.get_ft())
        with csv_path.open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['tactile', 'depth', 'FT'])
            stream.flush()

            def save_record(index, tactile, depth, force):
                force = np.asarray(force, dtype=float)
                if force.ndim != 1 or len(force) < 3 or not np.isfinite(force).all():
                    raise RuntimeError('Cannot record invalid force reading')
                image_path = Path('tactile')/f'{index:06d}.png'
                depth_path = Path('depth')/f'{index:06d}.npy'
                write_image(base_dir/image_path, tactile)
                np.save(base_dir/depth_path, depth)
                writer.writerow([image_path.as_posix(), depth_path.as_posix(),
                                 ' '.join(f'{value:.6f}' for value in force)])
                stream.flush()
                click.echo(f'{index+1}/{record_count} FT: {force}')

            def save_zero(event):
                metadata['zero_events'].append(dict(event, source='ui'))
                metadata['baseline_n'] = event['baseline_n']
                temporary = base_dir/'metadata.json.tmp'
                temporary.write_text(json.dumps(metadata, indent=2)+'\n')
                temporary.replace(base_dir/'metadata.json')

            if ui:
                record_interactively(dev, nn, reader, record_count, interval, save_record, save_zero)
            else:
                for index in range(record_count):
                    tactile, depth, force = capture_frame(dev, nn, read_force, discard_frames=10)
                    save_record(index, tactile, depth, force)
                    if index+1 < record_count:
                        time.sleep(interval)
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        try:
            if reader is not None:
                reader.close()
        finally:
            if dev is not None and getattr(dev, 'cam', None) is not None:
                dev.cam.release()


if __name__ == '__main__':
    main()
