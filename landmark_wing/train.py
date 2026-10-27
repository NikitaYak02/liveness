"""Train MobileNetV2 facial landmarks with Wing loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DATASET_LANDMARKS, default_train_val_splits, make_dataset
from .loss import WingLoss
from .metrics import nme
from .model import MobileNetV2Landmarks
from .vis import save_epoch_grid


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LANDMARKS))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--wing-w", type=float, default=10.0)
    parser.add_argument("--wing-eps", type=float, default=2.0)
    parser.add_argument("--vis-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, loader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    nmes: list[float] = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["landmarks"].to(device)
        preds = model(images)
        loss = criterion(preds, targets)
        losses.append(float(loss.item()))
        nmes.extend(nme(preds, targets).cpu().tolist())
    return float(np.mean(losses)), float(np.mean(nmes))


def _fixed_vis_batch(dataset, count: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    if count <= 0 or len(dataset) == 0:
        return None
    count = min(count, len(dataset))
    images = []
    landmarks = []
    for index in range(count):
        item = dataset[index]
        images.append(item["image"])
        landmarks.append(item["landmarks"])
    return torch.stack(images, dim=0), torch.stack(landmarks, dim=0)


def main() -> None:
    args = build_argparser().parse_args()
    _seed_everything(args.seed)
    device = args.device
    train_split, val_split = default_train_val_splits(args.dataset)
    num_landmarks = DATASET_LANDMARKS[args.dataset]

    train_set = make_dataset(
        args.dataset,
        args.root,
        train_split,
        image_size=args.image_size,
        augment=True,
        seed=args.seed,
    )
    val_set = make_dataset(
        args.dataset,
        args.root,
        val_split,
        image_size=args.image_size,
        augment=False,
        seed=args.seed,
    )
    if train_set.num_landmarks != num_landmarks:
        raise RuntimeError(
            f"expected {num_landmarks} landmarks for {args.dataset}, "
            f"got {train_set.num_landmarks}"
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = MobileNetV2Landmarks(
        num_landmarks=num_landmarks,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = WingLoss(width=args.wing_w, epsilon=args.wing_eps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_dir = out_path.parent / "vis"
    vis_batch = _fixed_vis_batch(val_set, args.vis_samples)

    best_nme = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device)
            targets = batch["landmarks"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=float(np.mean(losses[-50:])))

        scheduler.step()
        val_loss, val_nme = evaluate(model, val_loader, criterion, device)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_nme={val_nme:.4f}"
        )

        if vis_batch is not None:
            save_epoch_grid(
                model,
                vis_batch[0],
                vis_batch[1],
                vis_dir / f"epoch_{epoch:03d}.png",
            )

        payload = {
            "state_dict": model.state_dict(),
            "num_landmarks": num_landmarks,
            "image_size": args.image_size,
            "dataset": args.dataset,
            "epoch": epoch,
            "val_nme": val_nme,
        }
        torch.save(payload, out_path.with_name(out_path.stem + "_last" + out_path.suffix))
        if val_nme < best_nme:
            best_nme = val_nme
            torch.save(payload, out_path)
            print(f"saved best checkpoint to {out_path} (nme={best_nme:.4f})")


if __name__ == "__main__":
    main()
