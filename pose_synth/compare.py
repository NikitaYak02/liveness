#!/usr/bin/env python3
"""Per-method angle error summary for a pose_synth output folder.

Compares requested yaw/pitch with the pose 3DDFA-V2 measured on each generated
frame. Beyond roughly 70 degrees of yaw 3DDFA-V2 itself is unreliable, so the
large-yaw bin is a rough guide rather than a precise score.

    python compare.py out/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

YAW_BINS = ((0, 30), (30, 60), (60, 91))


def _method_stats(records: list[dict]) -> dict:
    present = [r for r in records if r.get("present")]
    measured = [r for r in present if r.get("measured")]
    stats = {
        "frames": len(records),
        "present": len(present),
        "measured": len(measured),
    }
    if not measured:
        return stats
    req_yaw = np.array([r["requested"]["yaw"] for r in measured])
    req_pitch = np.array([r["requested"]["pitch"] for r in measured])
    got_yaw = np.array([r["measured"]["yaw"] for r in measured])
    got_pitch = np.array([r["measured"]["pitch"] for r in measured])
    stats["yaw_mae"] = float(np.mean(np.abs(got_yaw - req_yaw)))
    stats["pitch_mae"] = float(np.mean(np.abs(got_pitch - req_pitch)))
    stats["max_abs_yaw_reached"] = float(np.max(np.abs(got_yaw)))
    for lo, hi in YAW_BINS:
        mask = (np.abs(req_yaw) >= lo) & (np.abs(req_yaw) < hi)
        if mask.any():
            stats[f"yaw_mae_{lo}_{hi}"] = float(np.mean(np.abs(got_yaw[mask] - req_yaw[mask])))
    for name, req, got in (("yaw", req_yaw, got_yaw), ("pitch", req_pitch, got_pitch)):
        if np.std(req) > 1e-6 and np.std(got) > 1e-6:
            corr = float(np.corrcoef(req, got)[0, 1])
            stats[f"{name}_corr"] = corr
            if corr < -0.5:
                stats.setdefault("warnings", []).append(
                    f"{name} sign looks inverted (corr {corr:+.2f}): adapter convention mismatch")
    return stats


def _fmt(value, spec=".1f") -> str:
    return "-" if value is None else format(value, spec)


def summarize(out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    methods = {}
    for method_dir in sorted(p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
        error_file = method_dir / "error.txt"
        poses_file = method_dir / "poses.json"
        if error_file.exists():
            methods[method_dir.name] = {"error": error_file.read_text().strip().splitlines()[0]}
        elif poses_file.exists():
            methods[method_dir.name] = _method_stats(json.loads(poses_file.read_text()))

    header = ("method", "frames", "measured", "yaw MAE", "|yaw|<30", "30-60", ">=60", "max |yaw|", "pitch MAE")
    rows = [header]
    for name, s in methods.items():
        if "error" in s:
            rows.append((name, "-", "-", "error: " + s["error"][:60], "", "", "", "", ""))
            continue
        rows.append((
            name,
            f"{s['present']}/{s['frames']}",
            str(s["measured"]),
            _fmt(s.get("yaw_mae")),
            _fmt(s.get("yaw_mae_0_30")),
            _fmt(s.get("yaw_mae_30_60")),
            _fmt(s.get("yaw_mae_60_91")),
            _fmt(s.get("max_abs_yaw_reached")),
            _fmt(s.get("pitch_mae")),
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows]
    for name, s in methods.items():
        for warning in s.get("warnings", []):
            lines.append(f"! {name}: {warning}")
    table = "\n".join(lines)

    summary = {"methods": methods, "table": table}
    (out_dir / "summary.json").write_text(json.dumps(methods, indent=1))
    (out_dir / "summary.txt").write_text(table + "\n")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    args = p.parse_args(argv)
    print(summarize(args.out_dir)["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
