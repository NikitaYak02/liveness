# Face artifacts generation (prototype)

Generate natural face artifacts — **beard**, **glasses with glare**, **bangs**, **occluders** — on a folder of images for keypoints / detector augmentation.

Geometry outside the artifact mask is preserved. Markup keypoints under the mask are annotated (`occluded`) or marked missing (`--occluded-policy`).

## Methods

| ID | Method | GPU | Notes |
|---|---|---|---|
| `flux_fill` | FLUX.1 Fill [dev] masked inpaint | ~24GB | Best quality for beard/bangs/glasses |
| `sdxl_cn` | SDXL inpaint + ControlNet (canny/depth) | ~12–24GB | Faster, OpenRAIL++ |
| `instruct_edit` | Qwen-Image-Edit / FLUX Kontext + diff paste-back | ~24GB | Best glare / instruction following |
| `stylegan` | e4e + InterFaceGAN / HairCLIP (or aligned fallback) | varies | Fast baseline; weak on large pose |
| `procedural` | Glasses compositing + NatOcc-style paste | CPU | Deterministic, no NN |

## Quick start

```bash
cd python/face_artifacts_gen
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-base.txt

# CPU procedural smoke run
python run_face_artifacts.py \
  -i /path/to/dataset \
  -o /path/to/out \
  --method procedural \
  --artifact glasses_glare \
  --device cpu \
  --skip-qa \
  --limit 20 \
  --seed 0
```

Diffusion backends:

```bash
pip install -r requirements-diffusion.txt
huggingface-cli login   # FLUX / some gated models

python run_face_artifacts.py -i ... -o ... \
  --method flux_fill --artifact beard --device cuda --cpu-offload
```

## Dataset layout

Same conventions as `face_kp_train_tools`:

```
dataset/
  images/**/*.jpg
  markup/**/*.json   # optional; InsightFace 2d106 used as fallback
```

Output:

```
out/<method>/<artifact>/
  images/...
  markup/...
  masks/...
  manifest.jsonl
  contact_sheet.html
```

## CLI

```
python run_face_artifacts.py -i INPUT -o OUTPUT \
  --method {flux_fill,sdxl_cn,instruct_edit,stylegan,procedural} \
  --artifact {beard,glasses_glare,bangs,occluder} \
  [--device cuda] [--limit N] [--seed S] \
  [--occluded-policy keep|missing] [--skip-qa] [--use-clip]
```

## License notes

- FLUX.1 [dev] / Kontext [dev]: non-commercial — check before product training use.
- SDXL / ControlNet Union: OpenRAIL++.
- Qwen-Image-Edit: Apache-2.0 (preferred instruct path).

## Tests

```bash
pip install -r requirements-base.txt pytest
python -m pytest tests/ -q
```
