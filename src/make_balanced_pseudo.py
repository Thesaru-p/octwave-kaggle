from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--probs", type=Path, required=True, help="Test probability CSV file from predict_super_ensemble.py")
    parser.add_argument("--out", type=Path, default=Path("outputs/train_pseudo_balanced.csv"))
    parser.add_argument("--max-per-class", type=int, default=300)
    return parser.parse_args()


# Adaptive thresholds per class to avoid starving minority classes (Class 0 and Class 3)
CLASS_THRESHOLDS = {
    0: {"min_conf": 0.92, "min_margin": 0.60},
    1: {"min_conf": 0.96, "min_margin": 0.70},
    2: {"min_conf": 0.94, "min_margin": 0.65},
    3: {"min_conf": 0.86, "min_margin": 0.45},
}


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    probs = pd.read_csv(args.probs)

    pseudo_dfs = []
    for cls, th in CLASS_THRESHOLDS.items():
        cls_df = probs[probs["appearance"] == cls].copy()
        
        # Check if character probabilities are present
        if "prob_tom" in cls_df.columns and "prob_jerry" in cls_df.columns and cls == 3:
            # Special high precision boost for Class 3 (both)
            mask = (cls_df["prob_tom"] >= 0.85) & (cls_df["prob_jerry"] >= 0.80)
            selected = cls_df[mask]
        else:
            mask = (cls_df["confidence"] >= th["min_conf"]) & (cls_df["margin"] >= th["min_margin"])
            selected = cls_df[mask]

        selected = selected.sort_values(["confidence", "margin"], ascending=[False, False]).head(args.max_per_class)
        pseudo_dfs.append(selected)

    pseudo = pd.concat(pseudo_dfs, ignore_index=True)

    train_out = train.copy()
    train_out["is_pseudo"] = 0
    pseudo_train = pseudo[["filename", "appearance"]].copy()
    pseudo_train["is_pseudo"] = 1

    combined = pd.concat([train_out, pseudo_train], ignore_index=True)
    combined.to_csv(args.out, index=False)

    print(f"[*] Real Train Samples: {len(train)}")
    print(f"[*] High-Confidence Pseudo Samples Added: {len(pseudo)}")
    print(f"[*] Total Combined Training Samples: {len(combined)}")
    print("\nPseudo-Label Distribution:")
    print(pseudo["appearance"].value_counts().sort_index().to_string())
    print(f"\n[✓] Saved Balanced Pseudo Dataset: {args.out}")


if __name__ == "__main__":
    main()
