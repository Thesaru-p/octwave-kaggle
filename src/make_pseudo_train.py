from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/train_pseudo.csv"))
    parser.add_argument("--min-confidence", type=float, default=0.98)
    parser.add_argument("--min-margin", type=float, default=0.75)
    parser.add_argument("--max-per-class", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    probs = pd.read_csv(args.probs)

    pseudo = probs[
        (probs["confidence"] >= args.min_confidence)
        & (probs["margin"] >= args.min_margin)
    ][["filename", "appearance", "confidence", "margin"]].copy()

    pseudo = (
        pseudo.sort_values(["appearance", "confidence", "margin"], ascending=[True, False, False])
        .groupby("appearance", group_keys=False)
        .head(args.max_per_class)
        .reset_index(drop=True)
    )

    train_out = train.copy()
    train_out["is_pseudo"] = 0
    pseudo_train = pseudo[["filename", "appearance"]].copy()
    pseudo_train["is_pseudo"] = 1
    combined = pd.concat([train_out, pseudo_train], ignore_index=True)
    combined.to_csv(args.out, index=False)

    print(f"real_train_rows={len(train)}")
    print(f"pseudo_rows={len(pseudo)}")
    print(f"combined_rows={len(combined)}")
    print("pseudo distribution:")
    print(pseudo["appearance"].value_counts().sort_index().to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
