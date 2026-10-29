"""Running adapters in their own environments and sharing GPUs between them."""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = PACKAGE_DIR / "adapters"
THIRD_PARTY_DIR = Path(os.environ.get("PS_THIRD_PARTY", PACKAGE_DIR / "third_party"))

METHODS = ("3ddfa", "liveportrait", "gpavatar", "facelift", "dp360")

# How many GPUs each method takes. DiffPortrait360 keeps its auxiliary models
# (back-head generator, PanoHead) on the second card when one is available.
GPU_NEEDS = {"3ddfa": 0, "liveportrait": 1, "gpavatar": 1, "facelift": 1, "dp360": 2}

ADAPTER_FILES = {
    "3ddfa": "tddfa.py",
    "liveportrait": "liveportrait.py",
    "gpavatar": "gpavatar.py",
    "facelift": "facelift.py",
    "dp360": "dp360.py",
}


def python_command(method: str, overrides: dict[str, str]) -> list[str]:
    """Interpreter for a method: --python override, then third_party/.venv_<method>, then conda."""
    if method in overrides:
        return shlex.split(overrides[method])
    venv_python = THIRD_PARTY_DIR / f".venv_{method}" / "bin" / "python"
    if venv_python.is_file():
        return [str(venv_python)]
    return ["conda", "run", "--no-capture-output", "-n", f"ps_{method}", "python"]


def adapter_command(method: str, overrides: dict[str, str], args: list[str]) -> list[str]:
    return python_command(method, overrides) + [str(ADAPTERS_DIR / ADAPTER_FILES[method]), *args]


def run_logged(cmd: list[str], log_path: Path, gpus: list[int] | None) -> int:
    env = os.environ.copy()
    env["PS_THIRD_PARTY"] = str(THIRD_PARTY_DIR)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ADAPTERS_DIR), env.get("PYTHONPATH")]))
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus)) if gpus else ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("$ " + shlex.join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        except FileNotFoundError as exc:
            log.write(f"cannot start: {exc}\n")
            return 127
    return proc.returncode


def tail(path: Path, lines: int = 25) -> str:
    """Last non-empty lines of a log."""
    try:
        text = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except FileNotFoundError:
        return ""
    return "\n".join(text[-lines:])


@dataclass
class GpuPool:
    """Hands out GPU ids; a job asking for more cards than exist gets all of them."""

    gpus: list[int]
    _free: list[int] = field(init=False)
    _cond: threading.Condition = field(init=False, default_factory=threading.Condition)

    def __post_init__(self):
        self._free = list(self.gpus)

    def acquire(self, need: int) -> list[int]:
        need = min(need, len(self.gpus))
        with self._cond:
            self._cond.wait_for(lambda: len(self._free) >= need)
            taken, self._free = self._free[:need], self._free[need:]
            return taken

    def release(self, taken: list[int]) -> None:
        with self._cond:
            self._free.extend(taken)
            self._free.sort()
            self._cond.notify_all()
