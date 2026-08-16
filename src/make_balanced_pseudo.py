from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from make_pseudo_train import exclude_matched


CLASS_THRESHOLDS = {
    0: {"min_conf": 0.92, "min_margin": 0.60},
    1: {"min_conf": 0.96, "min_margin": 0.70},
    2: {"min_conf": 0.94, "min_margin": 0.65},
    3: {"min_conf": 0.86, "min_margin": 0.45},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/train_pseudo_balanced.csv"))
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--exclude-csv", type=Path, default=None)
    parser.add_argument("--exclude-sources", nargs="*", default=["md5", "phash", "dino"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.data_dir / "train.csv")
    probs = exclude_matched(pd.read_csv(args.probs), args.exclude_csv, args.exclude_sources)

    pseudo_dfs = []
    for cls, threshold in CLASS_THRESHOLDS.items():
        cls_df = probs[probs["appearance"] == cls].copy()
        if "prob_tom" in cls_df.columns and "prob_jerry" in cls_df.columns and cls == 3:
            selected = cls_df[(cls_df["prob_tom"] >= 0.85) & (cls_df["prob_jerry"] >= 0.80)]
        else:
            selected = cls_df[
                (cls_df["confidence"] >= threshold["min_conf"]) & (cls_df["margin"] >= threshold["min_margin"])
            ]
        selected = selected.sort_values(["confidence", "margin"], ascending=[False, False]).head(args.max_per_class)
        pseudo_dfs.append(selected)

    non_empty = [frame for frame in pseudo_dfs if len(frame)]
    pseudo = (
        pd.concat(non_empty, ignore_index=True)
        if non_empty
        else pd.DataFrame(columns=["filename", "appearance"])
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
    print(pseudo["appearance"].value_counts().sort_index().to_string() if len(pseudo) else "empty")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
