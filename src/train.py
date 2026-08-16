from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import (
    CharacterPresenceModel,
    SUPPORTED_ARCHES,
    TomJerryDataset,
    class_logits_from_character_logits,
    find_image_dir,
    make_eval_transform,
    make_train_transform,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--arch", choices=SUPPORTED_ARCHES, default="efficientnet_b3")
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-weight", type=float, default=0.35)
    parser.add_argument("--blend-aux", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--use-aspect-pad", action="store_true", default=True)
    return parser.parse_args()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def make_loaders(args, train_df, val_df, image_dir):
    y = train_df["appearance"].to_numpy()
    counts = np.bincount(y, minlength=4)
    sample_weights = 1.0 / np.maximum(counts[y], 1)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = TomJerryDataset(
        train_df,
        image_dir,
        make_train_transform(args.image_size, use_aspect_pad=args.use_aspect_pad),
        labeled=True,
    )
    val_ds = TomJerryDataset(
        val_df,
        image_dir,
        make_eval_transform(args.image_size, use_aspect_pad=args.use_aspect_pad),
        labeled=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model, loader, device, blend_aux):
    model.eval()
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            class_logits, character_logits = model(images)
            blended_logits = (1.0 - blend_aux) * class_logits + blend_aux * class_logits_from_character_logits(character_logits)
            probs = torch.softmax(blended_logits, dim=1)
            preds = probs.argmax(dim=1)
            
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    raw_f1 = f1_score(y_true, np.array(all_preds), average="macro")
    probs_np = np.array(all_probs)
    return raw_f1, probs_np, y_true


def train_one_epoch(model, loader, optimizer, scaler, device, ce_loss, bce_loss, aux_weight):
    model.train()
    total_loss = 0.0
    for images, targets, character_targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        character_targets = character_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            class_logits, character_logits = model(images)
            loss = ce_loss(class_logits, targets) + aux_weight * bce_loss(character_logits, character_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv if args.train_csv is not None else args.data_dir / "train.csv"
    train_df = pd.read_csv(train_csv)
    image_dir = find_image_dir(args.data_dir)

    splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(train_df["filename"], train_df["appearance"]))
    train_idx, val_idx = splits[args.fold]
    fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

    class_counts = np.bincount(fold_train_df["appearance"].to_numpy(), minlength=4)
    character_targets = np.array([[(label in [1, 3]), (label in [2, 3])] for label in fold_train_df["appearance"]], dtype=np.float32)
    pos_counts = character_targets.sum(axis=0)
    neg_counts = len(character_targets) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterPresenceModel(args.arch, pretrained=True).to(device)
    train_loader, val_loader = make_loaders(args, fold_train_df, fold_val_df, image_dir)

    ce_loss = FocalLoss(gamma=2.0, label_smoothing=0.05)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"[*] Training {args.arch} | Image Size: {args.image_size} | Fold {args.fold}/{args.n_folds} on {device}")
    print(f"[*] Train Samples: {len(fold_train_df)} | Val Samples: {len(fold_val_df)} | Class Counts: {class_counts.tolist()}")

    best_f1 = -1.0
    stale_epochs = 0
    history = []
    best_path = checkpoint_dir / f"{args.arch}_fold{args.fold}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, ce_loss, bce_loss, args.aux_weight)
        scheduler.step()
        val_f1, val_probs, y_val = evaluate(model, val_loader, device, args.blend_aux)

        current_lr = scheduler.get_last_lr()[0]
        row = {"epoch": epoch, "train_loss": round(train_loss, 4), "val_macro_f1": round(val_f1, 5), "lr": current_lr}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_f1:.5f} | LR: {current_lr:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": args.arch,
                    "image_size": args.image_size,
                    "fold": args.fold,
                    "n_folds": args.n_folds,
                    "blend_aux": args.blend_aux,
                    "best_f1": best_f1,
                },
                best_path,
            )
            # Save fold validation predictions for rapid threshold optimization
            np.save(args.out_dir / f"oof_{args.arch}_fold{args.fold}.npy", val_probs)
            print(f"  [+] New Best Model Saved -> {best_path} (Val F1: {best_f1:.5f})")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[*] Early stopping at epoch {epoch}. Best Val F1: {best_f1:.5f}")
                break

    pd.DataFrame(history).to_csv(args.out_dir / f"history_{args.arch}_fold{args.fold}.csv", index=False)
    print(f"\n[✓] Training complete. Best Validation Macro-F1: {best_f1:.5f}")


if __name__ == "__main__":
    main()
