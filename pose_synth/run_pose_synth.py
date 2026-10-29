#!/usr/bin/env python3
"""Render a person at new yaw/pitch with several pose-synthesis methods and compare them.

Input is one photo or a folder of photos of the same person. Each method runs in its own venv under third_party/.venv_<method>
(see envs/; no conda required). Results land in OUT/<method>/ and a
side-by-side OUT/compare.mp4.

Host needs numpy + OpenCV (`bash envs/install_host.sh` or
`uv pip install -r requirements.txt`). Adapters use third_party/.venv_<method>.

On Arch (CUDA 13 / GCC 16) prefer Docker with CUDA 12.4:
    bash envs/docker_run.sh bash envs/install_all.sh
    bash envs/docker_run.sh python run_pose_synth.py -i /data/frames -o /out/r1 --gpus 0,1

Example:
    python run_pose_synth.py -i face.jpg -o out --gpus 0,1
    python run_pose_synth.py -i person_dir/ -o out --mode grid --step 10 \\
        --methods 3ddfa liveportrait gpavatar
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps import video  # noqa: E402
from ps.poses import MODES, build_trajectory  # noqa: E402
from ps.runner import (  # noqa: E402
    GPU_NEEDS,
    METHODS,
    THIRD_PARTY_DIR,
    GpuPool,
    adapter_command,
    run_logged,
    tail,
)
from ps.source_select import list_images, most_frontal, nearest_source  # noqa: E402

DP360_STATUS = THIRD_PARTY_DIR / "dp360_status.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="image file or folder with photos of one person")
    p.add_argument("-o", "--out", required=True, help="output folder")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--mode", default="sweep", choices=MODES,
                   help="sweep: yaw pass then pitch pass; grid: serpentine over yaw x pitch")
    p.add_argument("--yaw", nargs=2, type=float, default=(-75.0, 75.0), metavar=("MIN", "MAX"))
    p.add_argument("--pitch", nargs=2, type=float, default=(-30.0, 30.0), metavar=("MIN", "MAX"))
    p.add_argument("--step", type=float, default=5.0, help="degrees between frames")
    p.add_argument("--fps", type=float, default=12.0)
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids shared between methods")
    p.add_argument("--python", action="append", default=[], metavar="METHOD=CMD",
                   help="interpreter for a method instead of `conda run -n ps_<method> python`, "
                        "e.g. --python liveportrait=/opt/envs/lp/bin/python")
    p.add_argument("--dp360-chunk", type=int, default=0,
                   help="frames per DiffPortrait360 chunk; 0 = probe 16, 12, 8 and remember what fits")
    p.add_argument("--force-dp360", action="store_true",
                   help="run DiffPortrait360 even if an earlier probe found it does not fit")
    p.add_argument("--skip-measure", action="store_true", help="do not measure pose on generated frames")
    return p.parse_args(argv)


def parse_overrides(items: list[str]) -> dict[str, str]:
    overrides = {}
    for item in items:
        method, sep, cmd = item.partition("=")
        if not sep or method not in METHODS:
            raise SystemExit(f"bad --python value {item!r}, expected METHOD=CMD with METHOD in {METHODS}")
        overrides[method] = cmd
    return overrides


def measure_poses(images: list[str], work: Path, overrides: dict[str, str], tag: str) -> dict[str, dict | None]:
    """Head pose of each image via the 3DDFA-V2 adapter; empty dict if it is unavailable."""
    if not images:
        return {}
    list_path = work / f"measure_{tag}_in.json"
    result_path = work / f"measure_{tag}_out.json"
    list_path.write_text(json.dumps(images))
    code = run_logged(
        adapter_command("3ddfa", overrides, ["--measure", str(list_path), "--result", str(result_path)]),
        work / f"measure_{tag}.log",
        gpus=[],
    )
    if code != 0 or not result_path.exists():
        print(f"[measure:{tag}] 3DDFA-V2 unavailable (exit {code}), see {work / f'measure_{tag}.log'}")
        return {}
    return json.loads(result_path.read_text())


def dp360_blocked() -> str | None:
    if not DP360_STATUS.exists():
        return None
    status = json.loads(DP360_STATUS.read_text())
    if status.get("fits") is False:
        return status.get("reason", "earlier probe: does not fit into GPU memory")
    return None


def build_job(method: str, images: list[str], source_poses: dict, trajectory, args) -> dict:
    frontal = most_frontal(images, source_poses)
    frames = []
    for index, pose in enumerate(trajectory):
        target = pose.as_dict()
        if method in ("3ddfa", "liveportrait"):
            source = nearest_source(target, images, source_poses)
        else:
            source = frontal
        frames.append({"index": index, **target, "source": source})
    options = {}
    if method == "dp360":
        chunk = args.dp360_chunk
        if not chunk and DP360_STATUS.exists():
            chunk = json.loads(DP360_STATUS.read_text()).get("chunk") or 0
        options = {"chunk": chunk, "status_file": str(DP360_STATUS)}
    return {
        "method": method,
        "images": images,
        "source_poses": source_poses,
        "frontal": frontal,
        "frames": frames,
        "options": options,
    }


def run_method(method: str, job: dict, out_dir: Path, overrides, pool: GpuPool, errors: dict, lock):
    method_dir = out_dir / method
    (method_dir / "frames").mkdir(parents=True, exist_ok=True)
    job_path = method_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=1))
    log_path = method_dir / "log.txt"
    log_path.write_text("")

    need = GPU_NEEDS[method]
    gpus = pool.acquire(need) if need else []
    started = time.time()
    try:
        print(f"[{method}] start on GPUs {gpus or 'cpu'}")
        code = run_logged(adapter_command(method, overrides, ["--job", str(job_path), "--out", str(method_dir)]),
                          log_path, gpus)
    finally:
        if need:
            pool.release(gpus)
    elapsed = time.time() - started

    if code != 0:
        last_line = tail(log_path, 1).strip()
        headline = f"exit code {code}: {last_line}" if last_line else f"exit code {code}"
        (method_dir / "error.txt").write_text(headline + "\n\n" + tail(log_path) + "\n")
        with lock:
            errors[method] = headline
        print(f"[{method}] FAILED in {elapsed:.0f}s, see {method_dir / 'error.txt'}")
    else:
        (method_dir / "error.txt").unlink(missing_ok=True)
        print(f"[{method}] done in {elapsed:.0f}s")


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = parse_overrides(args.python)
    out_dir = Path(args.out).resolve()
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    images = [str(p.resolve()) for p in list_images(args.input)]
    trajectory = build_trajectory(args.mode, tuple(args.yaw), tuple(args.pitch), args.step)
    requested = [p.as_dict() for p in trajectory]
    print(f"{len(images)} input image(s), {len(trajectory)} target poses, methods: {', '.join(args.methods)}")

    source_poses = measure_poses(images, work, overrides, "sources")
    source_poses = {img: source_poses.get(img) for img in images}
    (out_dir / "sources.json").write_text(json.dumps(source_poses, indent=1))

    methods = list(args.methods)
    errors: dict[str, str] = {}
    if "dp360" in methods and not args.force_dp360:
        reason = dp360_blocked()
        if reason:
            errors["dp360"] = f"skipped: {reason} (use --force-dp360 to retry)"
            (out_dir / "dp360").mkdir(exist_ok=True)
            (out_dir / "dp360" / "error.txt").write_text(errors["dp360"] + "\n")
            print(f"[dp360] {errors['dp360']}")

    pool = GpuPool([int(g) for g in args.gpus.split(",") if g.strip() != ""])
    if not pool.gpus:
        raise SystemExit("--gpus must list at least one GPU id")
    lock = threading.Lock()
    threads = []
    for method in methods:
        if method in errors:
            continue
        job = build_job(method, images, source_poses, trajectory, args)
        t = threading.Thread(target=run_method, args=(method, job, out_dir, overrides, pool, errors, lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    succeeded = [m for m in methods if m not in errors]
    measured: dict[str, dict | None] = {}
    if not args.skip_measure:
        frames = [str(video.frame_path(out_dir / m / "frames", i))
                  for m in succeeded for i in range(len(trajectory))
                  if video.frame_path(out_dir / m / "frames", i).exists()]
        measured = measure_poses(frames, work, overrides, "outputs")

    records_by_method: dict[str, list[dict] | None] = {}
    for method in methods:
        if method in errors:
            records_by_method[method] = None
            continue
        frames_dir = out_dir / method / "frames"
        records = [{
            "index": i,
            "requested": req,
            "measured": measured.get(str(video.frame_path(frames_dir, i))),
            "present": video.frame_path(frames_dir, i).exists(),
        } for i, req in enumerate(requested)]
        (out_dir / method / "poses.json").write_text(json.dumps(records, indent=1))
        present = video.write_method_video(method, frames_dir, records, out_dir / method / "video.mp4", args.fps)
        if present == 0:
            errors[method] = "adapter finished but wrote no frames"
            records_by_method[method] = None
        else:
            records_by_method[method] = records

    video.write_compare_video(methods, out_dir, records_by_method, errors, requested,
                              out_dir / "compare.mp4", args.fps)

    from compare import summarize  # local module, imported late to keep startup light
    summary = summarize(out_dir)
    print(summary["table"])
    print(f"compare video: {out_dir / 'compare.mp4'}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
