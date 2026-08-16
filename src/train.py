from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--arch", choices=SUPPORTED_ARCHES, default="convnext_tiny")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-weight", type=float, default=0.35)
    parser.add_argument("--blend-aux", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--groups-csv", type=Path, default=Path("outputs/duplicate_clusters.csv"))
    return parser.parse_args()


def assign_groups(train_df: pd.DataFrame, groups_csv: Path | None) -> np.ndarray | None:
    if groups_csv is None or not groups_csv.exists():
        print(f"warning: groups csv missing ({groups_csv}); falling back to StratifiedKFold")
        return None

    groups_df = pd.read_csv(groups_csv)
    if "filename" not in groups_df.columns or "cluster_id" not in groups_df.columns:
        raise ValueError("--groups-csv must contain filename and cluster_id columns")

    mapping = groups_df.drop_duplicates("filename").set_index("filename")["cluster_id"]
    groups = train_df["filename"].map(mapping)
    missing = groups.isna()
    if missing.any():
        start = int(groups.max()) + 1 if groups.notna().any() else 0
        groups = groups.copy()
        groups.loc[missing] = np.arange(start, start + int(missing.sum()))
        print(f"warning: {int(missing.sum())} train files missing cluster_id; assigned unique groups")
    return groups.astype(int).to_numpy()


def make_loaders(args, train_df, val_df, image_dir):
    y = train_df["appearance"].to_numpy()
    counts = np.bincount(y, minlength=4)
    sample_weights = 1.0 / np.maximum(counts[y], 1)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ds = TomJerryDataset(train_df, image_dir, make_train_transform(args.image_size), labeled=True)
    val_ds = TomJerryDataset(val_df, image_dir, make_eval_transform(args.image_size), labeled=True)

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
    all_targets, all_preds, all_logits = [], [], []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            class_logits, character_logits = model(images)
            blended_logits = (1.0 - blend_aux) * class_logits + blend_aux * class_logits_from_character_logits(character_logits)
            preds = blended_logits.argmax(dim=1)
            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_logits.append(blended_logits.float().cpu().numpy())
    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    logits = np.concatenate(all_logits)
    macro = f1_score(targets, preds, average="macro")
    per_class = f1_score(targets, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    return macro, per_class, preds, logits, targets


def train_one_epoch(model, loader, optimizer, scaler, device, ce_loss, bce_loss, aux_weight):
    model.train()
    total_loss = 0.0
    for images, targets, character_targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        character_targets = character_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            class_logits, character_logits = model(images)
            loss = ce_loss(class_logits, targets) + aux_weight * bce_loss(character_logits, character_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def make_scheduler(optimizer, epochs: int, warmup_epochs: int):
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    warmup_epochs = min(warmup_epochs, epochs)
    cosine_epochs = max(epochs - warmup_epochs, 1)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def save_oof(path: Path, fold: int, val_df: pd.DataFrame, preds, logits, targets):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    oof = val_df[["filename"]].copy()
    oof["appearance"] = targets
    oof["fold"] = fold
    oof["pred"] = preds
    for i in range(4):
        oof[f"logit_{i}"] = logits[:, i]
        oof[f"prob_{i}"] = probs[:, i]
    if path.exists():
        previous = pd.read_csv(path)
        previous = previous[previous["fold"] != fold]
        oof = pd.concat([previous, oof], ignore_index=True)
    oof.to_csv(path, index=False)
    print(f"wrote {path} fold={fold} rows={len(oof)}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv if args.train_csv is not None else args.data_dir / "train.csv"
    train_df = pd.read_csv(train_csv)
    image_dir = find_image_dir(args.data_dir)

    if "is_pseudo" in train_df.columns:
        pseudo_mask = train_df["is_pseudo"].astype(int).eq(1)
        real_df = train_df.loc[~pseudo_mask].reset_index(drop=True)
        pseudo_df = train_df.loc[pseudo_mask].reset_index(drop=True)
    else:
        real_df = train_df
        pseudo_df = train_df.iloc[0:0].copy()

    groups = assign_groups(real_df, args.groups_csv)
    if groups is None:
        splitter = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(real_df["filename"], real_df["appearance"]))
    else:
        splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(real_df["filename"], real_df["appearance"], groups))

    train_idx, val_idx = splits[args.fold]
    fold_train_df = pd.concat([real_df.iloc[train_idx], pseudo_df], ignore_index=True)
    fold_val_df = real_df.iloc[val_idx].reset_index(drop=True)

    class_counts = np.bincount(fold_train_df["appearance"].to_numpy(), minlength=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterPresenceModel(args.arch, pretrained=True).to(device)
    train_loader, val_loader = make_loaders(args, fold_train_df, fold_val_df, image_dir)

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
            {
                "params": list(model.class_head.parameters()) + list(model.character_head.parameters()),
                "lr": args.lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    print(f"device={device} arch={args.arch} image_size={args.image_size} fold={args.fold}/{args.n_folds}")
    print(f"train_rows={len(fold_train_df)} val_rows={len(fold_val_df)} pseudo_rows={len(pseudo_df)} class_counts={class_counts.tolist()}")
    print(f"grouped_folds={groups is not None} warmup_epochs={args.warmup_epochs} backbone_lr_mult={args.backbone_lr_mult}")

    best_f1 = -1.0
    stale_epochs = 0
    history = []
    best_path = checkpoint_dir / f"{args.arch}_fold{args.fold}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, ce_loss, bce_loss, args.aux_weight)
        scheduler.step()
        val_f1, per_class, _, _, _ = evaluate(model, val_loader, device, args.blend_aux)
        lrs = scheduler.get_last_lr()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_f1,
            "val_f1_0": float(per_class[0]),
            "val_f1_1": float(per_class[1]),
            "val_f1_2": float(per_class[2]),
            "val_f1_3": float(per_class[3]),
            "lr_backbone": lrs[0],
            "lr_head": lrs[-1],
        }
        history.append(row)
        print(json.dumps(row))

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
                    "grouped": groups is not None,
                },
                best_path,
            )
            print(f"saved {best_path} best_f1={best_f1:.5f}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} best_f1={best_f1:.5f}")
                break

    pd.DataFrame(history).to_csv(args.out_dir / f"history_{args.arch}_fold{args.fold}.csv", index=False)
    print(f"best_macro_f1={best_f1:.5f}")

    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        _, per_class, preds, logits, targets = evaluate(model, val_loader, device, args.blend_aux)
        print(f"oof_per_class_f1={np.round(per_class, 5).tolist()}")
        save_oof(args.out_dir / f"oof_{args.arch}.csv", args.fold, fold_val_df, preds, logits, targets)


if __name__ == "__main__":
    main()
