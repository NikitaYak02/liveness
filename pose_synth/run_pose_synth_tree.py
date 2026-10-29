#!/usr/bin/env python3
"""Run pose synthesis on every image folder under a directory tree.

Walks INPUT recursively. Any directory that contains at least one image
(jpg/png/…) is treated as one person/sequence job. Nested folders without
images are only traversed. Output mirrors the relative path of each job:

    INPUT/a/b/seq/*.jpg  ->  OUT/a/b/seq/<method>/...

Extra arguments after ``--`` (or any unknown flags) are forwarded to
``run_pose_synth.py`` unchanged.

Examples:
    python run_pose_synth_tree.py -i /data/dataset -o /out/dataset \\
        --methods 3ddfa liveportrait --gpus 0,1 --step 10

    bash envs/docker_run.sh python run_pose_synth_tree.py \\
        -i /data/dataset -o /out/dataset --methods 3ddfa --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ps.runner import METHODS  # noqa: E402
from ps.source_select import IMAGE_EXTS  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_pose_synth.py"


def has_images(directory: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in directory.iterdir())


def discover_jobs(root: Path) -> list[Path]:
    """Directories under root (inclusive) that directly contain images, sorted."""
    root = root.resolve()
    if root.is_file():
        return [root.parent] if root.suffix.lower() in IMAGE_EXTS else []
    if not root.is_dir():
        raise FileNotFoundError(root)
    jobs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        folder = Path(dirpath)
        if any(Path(name).suffix.lower() in IMAGE_EXTS for name in filenames):
            jobs.append(folder)
    return jobs


def job_done(out_dir: Path, methods: list[str]) -> bool:
    """True when every requested method has frames or an error.txt (previous attempt finished)."""
    if not out_dir.is_dir():
        return False
    for method in methods:
        method_dir = out_dir / method
        if (method_dir / "error.txt").is_file():
            continue
        frames = method_dir / "frames"
        if not frames.is_dir() or not any(frames.glob("*.png")):
            return False
    return (out_dir / "summary.json").is_file() or (out_dir / "compare.mp4").is_file()


def parse_args(argv: list[str] | None = None):
    # Split so unknown run_pose_synth flags can follow after -- or be collected as passthrough.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path, help="root folder with nested frame folders")
    p.add_argument("-o", "--out", required=True, type=Path, help="root output folder (mirrors input layout)")
    p.add_argument("--methods", nargs="+", default=["3ddfa", "liveportrait"], choices=METHODS)
    p.add_argument("--gpus", default="0")
    p.add_argument("--force", action="store_true", help="re-run jobs that already have results")
    p.add_argument("--dry-run", action="store_true", help="only list discovered jobs")
    p.add_argument("--stop-on-error", action="store_true", help="abort the tree walk on the first failing job")
    p.add_argument("--python-bin", default=sys.executable,
                   help="interpreter for run_pose_synth.py (default: current python)")
    args, passthrough = p.parse_known_args(argv)
    # Allow either `... -- --step 10` or `... --step 10` mixed in.
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    root_in = args.input.expanduser().resolve()
    root_out = args.out.expanduser().resolve()
    jobs = discover_jobs(root_in)
    if not jobs:
        print(f"no image folders under {root_in}", file=sys.stderr)
        return 1

    print(f"found {len(jobs)} job(s) under {root_in}")
    results = []
    for index, job in enumerate(jobs, 1):
        rel = job.relative_to(root_in) if job != root_in else Path(".")
        out_dir = root_out / rel
        status = "skip" if (not args.force and job_done(out_dir, args.methods)) else "run"
        print(f"[{index}/{len(jobs)}] {status}: {job} -> {out_dir}")
        if args.dry_run or status == "skip":
            results.append({"input": str(job), "out": str(out_dir), "status": status, "code": 0})
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin, str(RUNNER),
            "-i", str(job),
            "-o", str(out_dir),
            "--methods", *args.methods,
            "--gpus", args.gpus,
            *passthrough,
        ]
        t0 = time.time()
        code = subprocess.call(cmd)
        elapsed = time.time() - t0
        results.append({"input": str(job), "out": str(out_dir), "status": "ok" if code == 0 else "fail",
                        "code": code, "seconds": round(elapsed, 1)})
        print(f"  -> exit {code} in {elapsed:.0f}s")
        if code != 0 and args.stop_on_error:
            break

    root_out.mkdir(parents=True, exist_ok=True)
    summary_path = root_out / "tree_summary.json"
    summary_path.write_text(json.dumps({"input": str(root_in), "jobs": results}, indent=1))
    failed = [r for r in results if r["status"] == "fail"]
    print(f"done: {len(results) - len(failed)} ok/skip, {len(failed)} failed; summary {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
