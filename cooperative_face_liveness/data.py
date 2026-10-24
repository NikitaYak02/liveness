from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class ManifestEntry:
    sample_id: str
    video_path: str
    label: int


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path)
    if path.suffix == ".jsonl":
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            entries.append(
                ManifestEntry(
                    sample_id=str(payload.get("id", Path(payload["video_path"]).stem)),
                    video_path=str(payload["video_path"]),
                    label=int(payload["label"]),
                )
            )
        return entries

    if path.suffix == ".csv":
        entries = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                entries.append(
                    ManifestEntry(
                        sample_id=str(row.get("id") or Path(row["video_path"]).stem),
                        video_path=str(row["video_path"]),
                        label=int(row["label"]),
                    )
                )
        return entries

    raise ValueError(f"Unsupported manifest format: {path}")


class PreparedLivenessDataset(Dataset):
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No prepared `.pt` samples were found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        sample = torch.load(self.files[index], map_location="cpu")
        return {
            "flow": sample["flow"].float(),
            "rgb": sample["rgb"].float(),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "meta": sample.get("meta", {}),
        }
