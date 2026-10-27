# Cooperative Face Liveness Detection from Optical Flow

Reference PyTorch implementation of the paper:

- Sokolov et al., "Cooperative Face Liveness Detection from Optical Flow"

The implementation follows the paper pipeline:

1. Read a cooperative video where the user slowly moves a frontal face closer to the camera.
2. Select three key frames:
   - `f1`: face height close to `0.500 * frame_height`
   - `f2`: face height close to `0.625 * frame_height`
   - `f3`: face height close to `0.750 * frame_height`
3. Detect facial landmarks.
4. Align `f1` and `f3` by roll angle, crop face with `10%` margin, resize to `256x256`.
5. Crop `f2` identically but without roll alignment.
6. Run pretrained RAFT on `f1` and `f3`.
7. Convert the flow to magnitude and clip values above `0.2 * crop_size = 51.2 px`.
8. Feed:
   - clipped 1-channel optical-flow magnitude to the flow backbone
   - RGB `f2` crop to the RGB backbone
9. Fuse the two ResNet18 embeddings and classify live/spoof.

## What data should be fed to the model

### Raw collection protocol

Each sample should be a short video recorded in the paper's cooperative scenario:

- the face is mostly frontal
- the user starts with face height around `50%` of frame height
- the user slowly approaches the camera
- the final face height is around `75%` of frame height
- the face should not disappear from the frame
- there should be no backward movement during capture

This exact protocol matters because the method relies on the optical-flow geometry created by the controlled forward motion.

### Labels

- `1` for a real live face
- `0` for a spoof / presentation attack

Examples of spoof classes discussed in the paper:

- screen photos
- printed photos
- printed masks
- dynamic video replays
- static video replays

### Training manifest

Use either `JSONL` or `CSV`.

`JSONL` example:

```json
{"id": "real_0001", "video_path": "/abs/path/real_0001.mp4", "label": 1}
{"id": "spoof_0042", "video_path": "/abs/path/spoof_0042.mp4", "label": 0}
```

`CSV` example:

```csv
id,video_path,label
real_0001,/abs/path/real_0001.mp4,1
spoof_0042,/abs/path/spoof_0042.mp4,0
```

## Which pretrained networks to use

### Required by the paper

1. Optical flow:
   - pretrained `RAFT`
   - in this implementation: `torchvision.models.optical_flow.raft_large`
   - recommended weights: `Raft_Large_Weights.DEFAULT`
   - paper setting: `256x256` input and `3` flow refinement iterations

2. Facial keypoints:
   - the paper says "neural facial key-point detector" but does not fix one exact network
   - in this implementation the default choice is `face-alignment` with pretrained `2D FAN` landmarks and `SFD` face detector

3. Liveness classifier:
   - dual `ResNet18`
   - RGB backbone: initialize from ImageNet pretrained `ResNet18`
   - flow backbone: also initialize from ImageNet pretrained `ResNet18`, adapting the first convolution from 3 channels to 1 by averaging RGB filters

### Practical recommendation

- Keep RAFT pretrained and frozen.
- Start the RGB backbone from ImageNet weights.
- Start the flow backbone from ImageNet weights adapted to 1 channel.
- Fine-tune both classifier backbones jointly on your liveness dataset.

The paper explicitly notes that the RGB backbone can also be pre-trained separately on larger single-shot anti-spoofing datasets before joint training.

## Data preparation

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare `.pt` training samples from raw videos:

```bash
python -m cooperative_face_liveness.prepare \
  --manifest /abs/path/train.jsonl \
  --out-dir /abs/path/prepared/train \
  --device cuda \
  --random-frame \
  --multi-resolution \
  --perspective-prob 0.5
```

The prepared sample contains:

- `flow`: tensor `[1, 256, 256]`
- `rgb`: tensor `[3, 256, 256]`
- `label`: scalar `0` or `1`
- `meta`: frame ids and face-height ratios

## Training

```bash
python -m cooperative_face_liveness.train \
  --train-dir /abs/path/prepared/train \
  --val-dir /abs/path/prepared/val \
  --out /abs/path/checkpoints/cooperative_face_liveness.pt \
  --device cuda
```

## Inference

```bash
python -m cooperative_face_liveness.infer \
  --video /abs/path/sample.mp4 \
  --checkpoint /abs/path/checkpoints/cooperative_face_liveness.pt \
  --device cuda
```

## Notes about fidelity to the paper

- The paper does not provide public training code or exact landmark-detector architecture.
- The paper does specify:
  - frame selection targets `0.500 / 0.625 / 0.750`
  - `10%` crop margin
  - crop size `256x256`
  - pretrained RAFT
  - clipped flow magnitude
  - clipping threshold `20%` of crop size
  - dual-backbone `ResNet18`
  - random-frame, multi-resolution, and perspective augmentations

So the classifier and preprocessing here are faithful to the published design, while the specific landmark backend is a practical PyTorch-native choice.

## Facial landmarks with Wing loss

Package `landmark_wing` trains a compact MobileNetV2 regressor on public face-keypoint datasets with Wing loss (Feng et al., CVPR 2018). One run uses one layout:

| `--dataset` | Points | Source |
|---|---|---|
| `300w` | 68 | [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/) |
| `wflw` | 98 | [WFLW](https://wywu.github.io/projects/LAB/WFLW.html) |
| `lapa` | 106 | [LaPa](https://github.com/jd-opensource/lapa-dataset) |

Download the dataset yourself (licenses forbid redistributing images) and point `--root` at the unpacked folder.

```bash
python -m landmark_wing.train \
  --dataset 300w \
  --root /abs/path/300W \
  --out checkpoints/wing_mobilenetv2_300w.pt \
  --device cuda
```

After every epoch the trainer writes `vis/epoch_XXX.png` next to the checkpoint: green = ground truth, red = prediction on a fixed validation subset. Use `--vis-samples 0` to disable.

Inference on a face bbox:

```bash
python -m landmark_wing.infer \
  --checkpoint checkpoints/wing_mobilenetv2_300w.pt \
  --image /abs/path/face.jpg \
  --bbox 100 80 400 420 \
  --out /abs/path/overlay.png
```

Smoke tests (no dataset required):

```bash
python -m pytest landmark_wing/tests -q
```
