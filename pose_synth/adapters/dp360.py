"""DiffPortrait360 adapter (experimental on 24 GB cards).

Pipeline, all on the most frontal photo:
  1. PanoHead-style 512 crop (face fairly small, hair in frame).
  2. In parallel: PanoHead projection + rendering of pose conditions and 3D-aware
     noise at our cameras (second GPU), back-of-head generation (first GPU).
  3. The video diffusion model on the first GPU, in chunks of `chunk` frames.
     Neighbouring chunks share OVERLAP frames whose second copy is dropped.
     The ControlNet temporal module is trained with ``video_length: 8``, so the
     only valid chunk size is 8 (16/12 break einops rearrange). With
     options.chunk == 0 we use 8, first fp16 then fp32 on non-shape failures.
  The outcome {fits, chunk, reason} goes to options.status_file, which the
  orchestrator reads to skip the method on later runs.

Cameras: PanoHead LookAtPoseSampler, frontal = (pi/2, pi/2). A head pose with
3DDFA-V2 angles (yaw, pitch) maps to h = pi/2 + YAW_SIGN*yaw, v = pi/2 + PITCH_SIGN*pitch;
the input photo gets the camera of its own measured pose.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import cv2

from common import face_crop, job_args, load_job, log, read_bgr, repo_dir, save_frame, save_meta, sign, source_box

# Must match motion_module_kwargs.video_length in
# cldm_v15_reference_only_temporal_pose.yaml (trained on continuous 8-view clips).
CHUNKS = (8,)
VIDEO_LENGTH = 8
SIZE = 512
CROP_SCALE = 1.9
CROP_SHIFT_Y = -0.12
YAW_SIGN = sign("dp360_yaw_sign", 1.0)
PITCH_SIGN = sign("dp360_pitch_sign", 1.0)
NAME = "input.png"
WEIGHTS = {
    "panohead": "easy-khair-180-gpc0.8-trans10-025000.pkl",
    "back_head": "back_head-230000.th",
    "main": "model_state-340000.th",
}


def to_camera(yaw: float, pitch: float) -> list[float]:
    return [math.pi / 2 + YAW_SIGN * math.radians(yaw), math.pi / 2 + PITCH_SIGN * math.radians(pitch)]


def windows(n: int, chunk: int) -> list[list[int]]:
    """Frame indices per chunk; every chunk is full length (the tail repeats the last frame)."""
    overlap = 2 if chunk >= 12 else 1
    result, start = [], 0
    while True:
        idx = list(range(start, min(start + chunk, n)))
        result.append(idx + [n - 1] * (chunk - len(idx)))
        if start + chunk >= n:
            return result
        start += chunk - overlap


def run(cmd: list[str], cwd: Path, gpu: str, log_path: Path) -> tuple[int, str]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
    log(f"[gpu {gpu}] {' '.join(cmd)}")
    with open(log_path, "w") as fp:
        code = subprocess.call(cmd, cwd=cwd, env=env, stdout=fp, stderr=subprocess.STDOUT)
    text = log_path.read_text(errors="replace")
    sys.stdout.write(text[-4000:])
    return code, text


def is_oom(text: str) -> bool:
    return "out of memory" in text.lower() or "OutOfMemoryError" in text


def is_chunk_mismatch(text: str) -> bool:
    """Model hardcodes video_length=8; wrong --nSample shows up as einops errors."""
    low = text.lower()
    return "einopserror" in low or "shape mismatch" in low or "can't divide axis" in low


def write_status(path: str | None, status: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(status, indent=1))


def assemble(work: Path, n_frames: int, chunk: int) -> tuple[Path, list[list[int]]]:
    """Lay out one chunked run the way inference.py's full_head_clean dataset expects."""
    run_dir = work / f"run_c{chunk}"
    shutil.rmtree(run_dir, ignore_errors=True)
    cond = run_dir / "conditions"
    noise = run_dir / "noise" / Path(NAME).stem
    cond.mkdir(parents=True)
    noise.mkdir(parents=True)
    (run_dir / "data").mkdir()
    shutil.copytree(work / "input_image", run_dir / "data" / "input_image")
    shutil.copytree(work / "Back_Head", run_dir / "data" / "Back_Head")

    wins = windows(n_frames, chunk)
    order = [i for w in wins for i in w]
    video = None
    for k, i in enumerate(order):
        shutil.copy(work / "cond" / f"{i:07d}.png", cond / f"{k:07d}.png")
        img = cv2.imread(str(work / "noise" / f"{i:07d}.png"))
        if video is None:
            video = cv2.VideoWriter(str(noise / f"condition{len(order)}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    10, (img.shape[1], img.shape[0]))
        video.write(img)
    video.release()
    return run_dir, wins


def read_video(path: Path) -> list:
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, img = cap.read()
        if not ok:
            return frames
        frames.append(img)


def main() -> int:
    args = job_args(lambda p: (p.add_argument("--projection-steps", type=int, default=200),
                               p.add_argument("--pti-steps", type=int, default=50)))
    job = load_job(args.job)
    opts = job.get("options", {})
    status_file = opts.get("status_file")
    requested = int(opts["chunk"]) if opts.get("chunk") else 0
    if requested and requested != VIDEO_LENGTH:
        log(f"ignoring unsupported dp360 chunk={requested}; "
            f"model video_length is {VIDEO_LENGTH}")
        requested = 0
    chunks = [requested] if requested else list(CHUNKS)

    code_dir = repo_dir("Diffportrait360") / "diffportrait360_release" / "code"
    weights = repo_dir("Diffportrait360") / "pretrained_weights"
    for w in WEIGHTS.values():
        if not (weights / w).is_file():
            raise SystemExit(f"missing {weights / w}; run envs/install_dp360.sh")
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if g.strip()]
    main_gpu, aux_gpu = gpus[0], gpus[-1]
    here = Path(__file__).resolve().parent
    py = sys.executable

    out = Path(args.out).resolve()
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "input_image").mkdir(parents=True)
    source = job["frontal"]
    cv2.imwrite(str(work / "input_image" / NAME),
                face_crop(read_bgr(source), source_box(job, source), SIZE, CROP_SCALE, CROP_SHIFT_Y))
    src_pose = job["source_poses"].get(source) or {"yaw": 0.0, "pitch": 0.0}
    frames = job["frames"]

    panohead_cmd = [py, str(here / "dp360_panohead.py"), "--network", str(weights / WEIGHTS["panohead"]),
                    "--image", str(work / "input_image" / NAME),
                    "--input-camera", json.dumps(to_camera(src_pose["yaw"], src_pose["pitch"])),
                    "--cameras", json.dumps([to_camera(f["yaw"], f["pitch"]) for f in frames]),
                    "--cond-dir", str(work / "cond"), "--noise-dir", str(work / "noise"),
                    "--num-steps", str(args.projection_steps), "--num-steps-pti", str(args.pti_steps)]
    back_cmd = [py, str(here / "dp360_infer.py"),
                "--model_config", "./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml",
                "--test_dataset", "back_head_generation", "--control_mode", "controlnet_important",
                "--local_image_dir", str(work / "Back_Head"), "--resume_dir", str(weights / WEIGHTS["back_head"]),
                "--control_type", "GAN_Generated", "--inference_image_path", str(work / "input_image"),
                "--nSample", "1", "--condition_path", str(code_dir.parent / "sample_data" / "cam_condition" / "sphere32")]

    results = {}
    threads = [
        threading.Thread(target=lambda: results.__setitem__(
            "panohead", run(panohead_cmd, code_dir / "3DNoise", aux_gpu, out / "panohead.log"))),
        threading.Thread(target=lambda: results.__setitem__(
            "back_head", run(back_cmd, code_dir, main_gpu, out / "back_head.log"))),
    ]
    if aux_gpu == main_gpu:
        for t in threads:
            t.start()
            t.join()
    else:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    for step, (code, text) in results.items():
        if code != 0:
            if is_oom(text):
                write_status(status_file, {"fits": False, "chunk": None, "reason": f"{step}: out of memory"})
            raise SystemExit(f"{step} failed with exit code {code}, see {out / (step + '.log')}")
    if not (work / "Back_Head" / NAME).is_file():
        raise SystemExit("back-head generation produced no image")

    for chunk in chunks:
        run_dir, wins = assemble(work, len(frames), chunk)
        for half in (True, False):
            result_dir = run_dir / "result"
            cmd = [py, str(here / "dp360_infer.py"),
                   "--model_config", "./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml",
                   "--test_dataset", "full_head_clean_inference_final_face", "--control_mode", "controlnet_important",
                   "--local_image_dir", str(result_dir), "--resume_dir", str(weights / WEIGHTS["main"]),
                   "--control_type", "GAN_Generated", "--inference_image_path", str(run_dir / "data"),
                   "--nSample", str(chunk), "--condition_path", str(run_dir / "conditions"),
                   "--denoise_from_guidance", "--initial_image_path", str(run_dir / "noise")]
            if half:
                cmd.append("--half")
            code, text = run(cmd, code_dir, main_gpu, out / f"main_c{chunk}{'_fp16' if half else ''}.log")
            videos = sorted(result_dir.glob("*.mp4")) if result_dir.is_dir() else []
            if code == 0 and videos:
                generated = read_video(videos[0])
                written = set()
                for w, start in zip(wins, range(0, len(generated), chunk)):
                    for i, img in zip(w, generated[start:start + chunk]):
                        if i not in written:
                            written.add(i)
                            save_frame(str(out), frames[i]["index"], cv2.resize(img, (SIZE, SIZE)))
                write_status(status_file, {"fits": True, "chunk": chunk, "fp16": half, "reason": ""})
                save_meta(str(out), {"source": source, "chunk": chunk, "fp16": half, "yaw_sign": YAW_SIGN,
                                     "pitch_sign": PITCH_SIGN, "frames_generated": len(generated)})
                return 0
            if is_chunk_mismatch(text):
                log(f"chunk {chunk} incompatible with model video_length={VIDEO_LENGTH}; skipping")
                break
            if is_oom(text):
                log(f"out of memory at chunk {chunk} (fp16={half})")
                break
            if not half:
                raise SystemExit(f"inference failed with exit code {code} at chunk {chunk}")
            log(f"fp16 run failed (exit {code}), retrying chunk {chunk} in fp32")

    write_status(status_file, {"fits": False, "chunk": None,
                               "reason": f"out of memory even at chunk {chunks[-1]} on one 24 GB card"})
    raise SystemExit(f"DiffPortrait360 does not fit: out of memory at chunk {chunks[-1]}")


if __name__ == "__main__":
    raise SystemExit(main())
