from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.read_csv(args.submission)
    corrections = pd.read_csv(args.corrections)

    required = {"filename", "appearance"}
    if not required.issubset(corrections.columns):
        raise ValueError("corrections CSV must contain filename and appearance columns")

    valid_labels = {0, 1, 2, 3}
    bad_labels = set(corrections["appearance"].astype(int)) - valid_labels
    if bad_labels:
        raise ValueError(f"invalid appearance labels: {sorted(bad_labels)}")

    corrections = corrections[["filename", "appearance"]].copy()
    corrections["appearance"] = corrections["appearance"].astype(int)

    corrected = submission.copy()
    correction_map = dict(zip(corrections["filename"], corrections["appearance"]))
    corrected["appearance"] = corrected.apply(
        lambda row: correction_map.get(row["filename"], row["appearance"]),
        axis=1,
    )

    changed = int((corrected["appearance"] != submission["appearance"]).sum())
    corrected.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(corrected)} corrections={len(corrections)} changed={changed}")


if __name__ == "__main__":
    main()
