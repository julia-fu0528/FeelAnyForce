"""Prepare two Nano25 recorder sessions for FeelAnyForce calibration training."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image


def load_recording(root):
    rows = pd.read_csv(root/'calibration.csv')
    if rows.empty or not {'tactile', 'depth', 'FT'}.issubset(rows.columns):
        raise ValueError(f'Empty or invalid recording: {root}')
    background = np.array(Image.open(root/'background.png').convert('RGB'))
    for row in rows.itertuples():
        image_path, depth_path = Path(row.tactile), Path(row.depth)
        if (image_path.is_absolute() or depth_path.is_absolute()
                or '..' in image_path.parts or '..' in depth_path.parts
                or image_path.parts[0] != 'tactile' or depth_path.parts[0] != 'depth'):
            raise ValueError(f'Expected recorder-relative tactile/depth paths in {root}')
        image = np.array(Image.open(root/image_path).convert('RGB'))
        depth = np.load(root/depth_path)
        force = np.fromstring(str(row.FT), sep=' ')
        if image.shape != background.shape or depth.shape != image.shape[:2]:
            raise ValueError(f'Image/background/depth size mismatch: {root/image_path}')
        if force.size != 3 or not np.isfinite(force).all() or not np.isfinite(depth).all():
            raise ValueError(f'Invalid Nano25 force or depth: {root/image_path}')
    return rows, background


def prepare_calibration(train_run, validation_run, stats_path):
    train_run, validation_run, stats_path = map(Path, (train_run, validation_run, stats_path))
    train_run, validation_run = train_run.resolve(), validation_run.resolve()
    if train_run == validation_run:
        raise ValueError('Training and validation must use different recordings')
    outputs = [train_run/'tactile_nobg', validation_run/'tactile_nobg',
               train_run/'calibration_train.csv', train_run/'calibration_val.csv',
               train_run/'normalization.json']
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Already prepared or output exists: {path}')
    metadata = [json.loads((root/'metadata.json').read_text()) for root in (train_run, validation_run)]
    if any(item.get('force_unit') != 'N' for item in metadata):
        raise ValueError('Both recordings must contain Nano25 forces in N')
    if not metadata[0].get('force_axes') or metadata[0]['force_axes'] != metadata[1].get('force_axes'):
        raise ValueError('Training and validation force axes must match')
    # Validate every input before writing derived files; originals are retained.
    json.loads(stats_path.read_text())
    recordings = [load_recording(root) for root in (train_run, validation_run)]
    for root, (rows, background) in zip((train_run, validation_run), recordings):
        (root/'tactile_nobg').mkdir()
        for row in rows.itertuples():
            image = np.array(Image.open(root/row.tactile).convert('RGB'))
            destination = root/'tactile_nobg'/Path(row.tactile).relative_to('tactile')
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Match the existing live predictor's uint8 background subtraction.
            Image.fromarray(image - background + 127).save(destination)
    train_rows, val_rows = recordings[0][0], recordings[1][0].copy()
    # Both label CSVs are loaded relative to --data_basedir (the training folder).
    for column in ('tactile', 'depth'):
        val_rows[column] = [Path(os.path.relpath(validation_run/value, train_run)).as_posix()
                            for value in val_rows[column]]
    train_rows.to_csv(train_run/'calibration_train.csv', index=False)
    val_rows.to_csv(train_run/'calibration_val.csv', index=False)
    shutil.copyfile(stats_path, train_run/'normalization.json')
    return len(train_rows), len(val_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_run', type=Path)
    parser.add_argument('--validation-run', type=Path, required=True)
    parser.add_argument('--stats', type=Path, default=Path(__file__).with_name('mean_and_std.json'))
    args = parser.parse_args()
    try:
        train_count, val_count = prepare_calibration(args.train_run, args.validation_run, args.stats)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Prepared {train_count} training and {val_count} validation frames.')
    print(f'Labels and normalization: {args.train_run.resolve()}')


if __name__ == '__main__':
    main()
