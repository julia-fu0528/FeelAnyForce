<div align="center">
  <img src="assets/overview.png"
  width="85%">
</div>
<br>

<h4 align="center">
  FeelAnyForce: Estimating Contact Force Feedback from Tactile Sensation for Vision-Based Tactile Sensors
</h4>

<div align="center">
  <a href="https://prg.cs.umd.edu/FeelAnyForce"><b>Website</b></a> |
  <a href="https://arxiv.org/abs/2410.02048"><b>Arxiv</b></a> |
  <a href="https://www.youtube.com/watch?v=IfQCkUj2xbU"><b>YouTube</b></a>
</div>

## Table of Contents

- [Update](#update)
- [Installation](#gear-installation)
- [Real-Time prediction on GelSight Mini](#-Real-Time-prediction-on-GelSight-Mini)
- [Calibration](#-Calibration)
- [Citing this paper](#citing-this-paper)

## Update
- 01/30/25 Accepted to ICRA25!
- 03/01/25 Data release
- 03/24/25 First release of the code
## :gear: Installation

### :one: Clone this repository
To clone this repository you can launch the following command
```bash
git clone https://github.com/prgumd/FeelAnyForce.git
cd FeelAnyForce
```

### :two: Create and activate the virtual environment
To create and activate the virtual environment to work with this repository you can launch the following command
```bash
conda env create -f env.yml
conda activate anyforce_env
```

or
```bash
virtualenv anyforce_env
source anyforce_env/bin/activate
pip install -r requirements.txt
```

### :three: Download the data
Create dataset folder and download the data with

```bash
bash dataset_downloader.sh
```

### :four: Training from scratch
To train the network,
```bash
python training_multi_head.py --epochs 100 --labels_train ./dataset/TacForce_train_set.csv --labels_val ./dataset/TacForce_val_set.csv --ckpt_dir $ckpt_dir --data_basedir ./dataset/ --training_name TacForce_v1 
``` 

## 🎬 Real-Time prediction on GelSight Mini 

```bash
wget -P ckpt/ https://huggingface.co/amirsh1376/FeelAnyForce/resolve/main/checkpoint_v1.pth.tar   # download the checkpoint
python online_test.py --checkpoint ckpt/checkpoint_v1.pth.tar
``` 

To display predicted and measured **Fx, Fy, Fz together**, use the Nano25 Net F/T
comparison mode. It reuses the network reader in the sibling `tac_calib` project:

```bash
python online_test.py --checkpoint ckpt/checkpoint_v1.pth.tar \
  --nano25 --nano-bind 192.168.1.100
```

Close other GelSight/Nano viewers first. Keep the gel unloaded throughout startup:
the same **50 unloaded camera frames** establish the depth zero and Nano baseline.
For each frame, a healthy Nano sample is matched at camera receipt time before
depth computation; their mean is subtracted from subsequent Nano forces. The
last unloaded frame becomes the model image reference. The window shows predicted and
measured forces in newtons. Press **Z while unloaded** to refresh both references,
or **Q/Esc** to quit and release the devices. **Z** repeats the shared 50-frame depth,
Nano, and image calibration; keep the gel unloaded until `Ready` appears. Missing
or faulted Nano samples stop calibration rather than creating a partial baseline.
Zeroing removes the mean unloaded offset; small sensor noise can remain around zero.
Prediction-only operation still works
without `--nano25`. The existing CUDA model requirements remain unchanged.

The same window also shows scrolling **Fx, Fy, Fz plots**, with predicted force
in blue and Nano25 ground truth in orange. Each pair shares a vertical scale in
newtons; the horizontal axis shows seconds relative to the latest camera frame.
The default history is 10 seconds. Change it with `--nano-history-seconds 20`
(supported range: 1–120 seconds). Nano traces use the matched sample for each
camera frame, rather than displaying the independent full-rate Nano stream.
Missing/invalid readings and capture gaps over one second break the traces.
**Z** clears the plots after re-zeroing so different baselines are never mixed.
Axis/sign mapping and the UNVERIFIED label apply to the plots too.

In Nano comparison mode, click **Record** (or press **R**) to record an evaluation
segment. Click **Stop / MSE** or press **R** again to finish and calculate MSE.
The window displays running per-axis and overall MSE; stopping also prints the
results and output folder in the terminal. The recorded GT is exactly the mapped,
zeroed GT shown on screen; predictions are stored without another axis transform.

Each segment gets a new folder under `evaluation/<timestamp>_<unique suffix>/`:

- `pairs.csv`: predicted/GT Fx, Fy, Fz in N, camera/Nano host timestamps, matching
  gap, validity flag, and status for every recorded camera frame.
- `metrics.json`: valid/excluded counts, MSE for each axis, and overall MSE in N².
- `metadata.json`: checkpoint, normalization path, Nano axis mapping and baseline,
  matching limit, and metric definition.

Use `--eval-output-dir /path/to/evaluations` to change the parent folder. Existing
recordings are never overwritten. Quit, Ctrl+C, and window close finalize active
recordings. **Z finishes the current segment before re-zeroing**; press R afterward
to begin a new one. Startup and reference frames are not evaluation samples.

Per-axis MSE is the average of `(prediction - GT)^2` over valid recorded frames.
Overall MSE is the mean of the three per-axis MSE values, not their sum. Invalid
GT matches and nonfinite predictions are logged but excluded; zero valid samples
produce `null` MSE rather than zero. These metrics use the displayed host-matched
pairs without shifting curves to compensate for latency. They include unloaded
frames if those are recorded. Evaluation CSVs are separate from calibration data.

To recalculate MSE later from the recorded forces:

```bash
python evaluation_recording.py ./evaluation/RUN_FOLDER/pairs.csv
```

The default ATI address is `192.168.1.1`; use `--nano-ip` if different.
`--nano-bind` must be the local adapter's assigned IP. If `tac_calib` is elsewhere,
pass `--tac-calib-root /path/to/tac_calib` (it must contain `ft_test/ati_live.py`).
The reader converts configured ATI force units, including lbf, to N. Ground truth
is matched to camera host receipt time before depth computation/inference, with a
50 ms maximum gap (`--nano-max-gap-ms`). Missing, faulted or unmatched readings
show as unavailable, not zero. These are host timestamps, not hardware synchronization.

Initially, predictions use model axes and GT uses Nano axes, clearly labeled
**UNVERIFIED**. Check axis order/signs with steady contacts before interpreting
differences. Once verified, provide the signed Nano axes in model Fx,Fy,Fz order,
for example `--nano-force-map=x,y,z` for verified identity alignment or
`--nano-force-map=-y,x,z` for a verified permutation/sign change. These examples
are not a calibration of your mounting. Supplying a map also enables the
`Pred-GT` column. Arbitrary sensor rotations need a separate rotation calibration.

Comparison helpers and camera-loop behavior can be tested without devices:

```bash
python -m unittest discover -s tests -v
```

## 🩹 Calibration
To record calibration data, you can 3D print a fixture for your force sensor along with a mount that allows the tactile sensor to be attached securely to the force sensor.

We’ve designed an [indenter](https://cad.onshape.com/documents/49421fb35cd6f8baa3cf3824/w/b05d7621015714dc8ed84cee/e/5af7331f10bbd3e8d1a7921a) to replicate the indentation from a robot’s end effector. This calibration setup allows for stable indentation on the tactile sensor, ensuring the force sensor readings can stabilize. Additionally, we included placeholders for calibration weights to apply various forces on the indenter.

<div align="center"> <img src="assets/calib.png" width="60%"> </div>

For the **Ethernet Nano25**, close other camera/sensor viewers and run:

```bash
python record_deForce.py --nano25 --nano-bind 192.168.1.100 \
  --record_count 100 --interval 1
```

For a live camera preview, Nano force values/history, and clickable recording
controls, add `--ui`. To use `Fx = -Nano Y`, `Fy = Nano X`, `Fz = Nano Z`:

```bash
python record_deForce.py --nano25 --nano-bind 192.168.1.100 \
  --nano-force-map=-y,x,z --ui --record_count 100 --interval 1
```

After startup zeroing, UI mode begins **paused**. Click **Zero** (or press `Z`)
with the gel unloaded to average the **next 10 camera-matched Nano readings**.
The previous Nano baseline is replaced only after all 10 readings are valid.
Zeroing pauses recording and saves no training samples; click **Start** (or
press Space) to begin/resume. The Zero button changes only the Nano baseline;
the GelSight image/depth reference remains the one captured at startup.
An invalid Nano reading cancels zeroing, keeps the old baseline, and pauses
recording with a visible error. Each successful zero is logged in
`metadata.json` under `zero_events`, with `first_record_index` indicating the
first CSV row that uses it (zero-based, excluding the header).

The displayed force values use the same mapping and baseline as the saved labels.
Preview continues between saved samples; `--interval` sets the pause between
saves, not the preview rate. At the requested sample count, recording stops and
the live preview remains open. Click **Quit**, press `Q`/Esc, or close the window.
Closing early keeps completed samples. Without `--ui`, automatic console
recording works as before.

Keep the gel unloaded until `Ready`: the same 50 camera frames establish the
depth and Nano force baselines. The recorder writes `Fx Fy Fz` in **newtons**,
matched at camera receipt time before depth computation (maximum gap 50 ms).
Console mode discards buffered camera frames before each recorded frame; UI mode
reads camera frames continuously between saves. Matching uses host
receipt times, not hardware synchronization; keep contacts steady while recording.
Missing, stale, or faulted force samples stop the run without adding an invalid row.
Ctrl+C keeps completed records and closes both devices.

Each run creates `dataset/calibration_sphere/<timestamp>/` with `calibration.csv`,
`tactile/`, `depth/`, `background.png`, and `metadata.json`. CSV image/depth paths
are relative to that run folder. Use `--output-dir /path/to/new-run` to choose a
location; existing folders are refused to protect previous data. `--interval`
is the pause in seconds between saved records, in addition to capture/processing.

The IP/bind and `--tac-calib-root` options work as in the live viewer. Labels use
native Nano axes unless you supply a verified `--nano-force-map`; the mapping and
unloaded baseline are saved in metadata. This mode records three forces, not torques.
For the original serial ATI sensor at `/dev/ttyUSB0`, omit `--nano25`:

```bash
python record_deForce.py --record_count 100 --interval 1
```

### 📊 Training with Calibration Data

It’s recommended to balance your dataset for a more uniform force distribution, as discussed on the project website.

Calibration starts from the released `ckpt/checkpoint_v1.pth.tar` (or your own
trained checkpoint). `--checkpoint` selects input weights; `--ckpt_dir` selects
where new checkpoints are saved. Do not use an unset shell variable for either.
The checkpoint supplies model/preprocessing settings and its original normalization;
the new run keeps its CLI dataset paths, batch size, learning rates, and output path.
Only the selected final regressor layers are optimized, as in the original calibration mode.

Record separate training and validation sessions with the same sensor mounting
and force-axis settings. Include positive/negative shear in both axes, several
contact positions and normal pressures, and unloaded samples. Use new contacts
covering the same range for validation. After both recordings complete, prepare
them with:

```bash
python prepare_calibration.py ./dataset/calibration_sphere/TRAIN_RUN \
  --validation-run ./dataset/calibration_sphere/VALIDATION_RUN
```

This creates `tactile_nobg/` in both sessions using each session's own background
and the repo's uint8 `image - background + 127` operation. It writes
`calibration_train.csv`, `calibration_val.csv`, and `normalization.json` into the
training folder, with validation paths referring to the separate session.
Original recordings are retained; existing prepared outputs are refused.
Training does not require the optional `TacForce_test_set.csv`.

Replace `RUN_TIMESTAMP` with the prepared run folder name:

```bash
calibration_run="./dataset/calibration_sphere/RUN_TIMESTAMP"
python training_multi_head.py \
  --checkpoint ./ckpt/checkpoint_v1.pth.tar \
  --ckpt_dir ./ckpt/ \
  --data_basedir "$calibration_run/" \
  --labels_train calibration_train.csv \
  --labels_val calibration_val.csv \
  --dataset_stats "$calibration_run/normalization.json" \
  --training_name nano25 \
  --batch_size_per_gpu 4 --num_workers 0 \
  --lr_backbone 0.00001 --lr_calibration 0.0001 \
  --tactile_backbone_training calibration
```

The default is 100 epochs; use `--epochs 1` for a short check. New weights go to
`ckpt/nano25_calibration/best_val.pth.tar` and `last.pth.tar` (a timestamp is added
to the experiment directory if it already exists). The run's normalization file
also gets an entry for its training CSV so the calibrated checkpoint works with
the live predictor.

## Citing this paper
```bibtex
@misc{shahidzadeh2024feelanyforceestimatingcontactforce,
      title={FeelAnyForce: Estimating Contact Force Feedback from Tactile Sensation for Vision-Based Tactile Sensors}, 
      author={Amir-Hossein Shahidzadeh and Gabriele Caddeo and Koushik Alapati and Lorenzo Natale and Cornelia Fermüller and Yiannis Aloimonos},
      year={2024},
      eprint={2410.02048},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.02048}, 
}
```
## 🧔 Maintainer

This repository is maintained by:

|                                                                                                          |                                                                             |
|:--------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------:|
|        [<img src="https://github.com/AmirShahid.png" width="40">](https://github.com/AmirShahid)         |                [@AmirShahid](https://github.com/AmirShahid)                 | | | |
|    [<img src="https://github.com/gabrielecaddeo.png" width="40">](https://github.com/gabrielecaddeo)     |            [@gabrielecaddeo](https://github.com/gabrielecaddeo)             |


## Acknowledegments
Thanks to these great repositories: [DinoV2](https://github.com/facebookresearch/dinov2), [Dino](https://github.com/facebookresearch/dino), [Sim2Surf](https://github.com/hsp-iit/sim2real-surface-classification), [gsrobotics](https://github.com/gelsightinc/gsrobotics)



conda activate anyforce_env

run="./dataset/calibration_sphere/nano25_$(date +%Y%m%d_%H%M%S)"
val="${run}_val"

python record_deForce.py --nano25 --nano-bind 192.168.1.100 \
  --record_count 500 --interval 0.5 --output-dir "$run"

Include both directions of X and Y shear, several contact locations, different normal pressures, and occasional unloaded samples. Move slowly and hold contacts steady; don’t collect only downward presses.

Next, record 100 validation samples using new contacts across the same range:

python record_deForce.py --nano25 --nano-bind 192.168.1.100 \
  --record_count 100 --interval 0.5 --output-dir "$val"

Prepare background-subtracted images and training/validation CSVs:

python prepare_calibration.py "$run" --validation-run "$val"

Retrain from the original pretrained checkpoint:

python training_multi_head.py \
  --checkpoint ./ckpt/checkpoint_v1.pth.tar \
  --ckpt_dir "$run/checkpoints" \
  --data_basedir "$run/" \
  --labels_train calibration_train.csv \
  --labels_val calibration_val.csv \
  --dataset_stats "$run/normalization.json" \
  --training_name nano25 \
  --batch_size_per_gpu 4 --num_workers 0 \
  --epochs 100 \
  --lr_backbone 0.00001 --lr_calibration 0.0001 \
  --tactile_backbone_training calibration

After training finishes, test the best checkpoint:

python online_test.py \
  --checkpoint "$run/checkpoints/nano25_calibration/best_val.pth.tar" \
  --nano25 --nano-bind 192.168.1.100

  ok apparently x y are misaligned
nano -y becomes predicted +x
