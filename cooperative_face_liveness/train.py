from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PreparedLivenessDataset
from .model import DualResNet18Liveness


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-rgb-pretrain", action="store_true")
    parser.add_argument("--no-flow-pretrain", action="store_true")
    return parser


def _roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    ranks = np.zeros_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end

    pos_mask = sorted_labels == 1
    neg_mask = sorted_labels == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pos_rank_sum = ranks[pos_mask].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(model, loader, criterion, device: str) -> tuple[float, float]:
    model.eval()
    losses = []
    labels_all = []
    scores_all = []
    with torch.inference_mode():
        for batch in loader:
            flow = batch["flow"].to(device)
            rgb = batch["rgb"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            logits = model(flow, rgb)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            labels_all.append(labels.squeeze(1).cpu().numpy())
            scores_all.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())

    auc = _roc_auc_score(np.concatenate(labels_all), np.concatenate(scores_all))
    return float(np.mean(losses)), auc


def main() -> None:
    args = build_argparser().parse_args()
    device = args.device

    train_set = PreparedLivenessDataset(args.train_dir)
    val_set = PreparedLivenessDataset(args.val_dir)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DualResNet18Liveness(
        pretrained_rgb=not args.no_rgb_pretrain,
        pretrained_flow=not args.no_flow_pretrain,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = float("-inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            flow = batch["flow"].to(device)
            rgb = batch["rgb"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(flow, rgb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            progress.set_postfix(train_loss=f"{np.mean(losses):.4f}")

        val_loss, val_auc = evaluate(model, val_loader, criterion, device=device)
        mean_train_loss = float(np.mean(losses))
        print(
            f"epoch={epoch} train_loss={mean_train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint = {
                "model_kwargs": {
                    "pretrained_rgb": not args.no_rgb_pretrain,
                    "pretrained_flow": not args.no_flow_pretrain,
                },
                "state_dict": model.state_dict(),
                "val_auc": val_auc,
            }
            torch.save(checkpoint, out_path)
            print(f"Saved best checkpoint to {out_path}")


if __name__ == "__main__":
    main()
