from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from common import find_image_dir
from duplicates import (
    agreement_by_threshold,
    cluster_phash,
    extract_dinov2_embeddings,
    hamming_block,
    hex_to_uint64,
    index_images,
    nearest_other,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--dino-threshold", type=float, default=0.97)
    parser.add_argument("--use-dino", action="store_true")
    parser.add_argument("--dino-model", type=str, default="dinov2_vitb14")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


def leakage_rate(val_idx: np.ndarray, train_idx: np.ndarray, cluster_ids: np.ndarray) -> float:
    train_clusters = set(cluster_ids[train_idx].tolist())
    leaked = 0
    for i in val_idx:
        cluster = int(cluster_ids[i])
        train_members = int(np.sum((cluster_ids[train_idx] == cluster)))
        if cluster in train_clusters and train_members > 0:
            # A singleton cluster cannot leak a different frame.
            cluster_size = int(np.sum(cluster_ids == cluster))
            if cluster_size > 1:
                leaked += 1
    return leaked / max(len(val_idx), 1)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)

    train_names = train_df["filename"].tolist()
    test_names = test_df["filename"].tolist()
    all_names = train_names + test_names

    cache_path = args.out_dir / "hash_index.csv"
    index = index_images(all_names, image_dir, cache_path=cache_path)
    index = index.drop_duplicates("filename").set_index("filename").loc[all_names].reset_index()

    split = np.array(["train"] * len(train_names) + ["test"] * len(test_names))
    appearance = np.array(train_df["appearance"].tolist() + [np.nan] * len(test_names), dtype=np.float32)
    phash_u = np.array([hex_to_uint64(h) for h in index["phash"]], dtype=np.uint64)
    cluster_ids = cluster_phash(phash_u, threshold=args.phash_threshold)

    clusters = pd.DataFrame(
        {
            "filename": all_names,
            "split": split,
            "appearance": appearance,
            "md5": index["md5"],
            "phash": index["phash"],
            "cluster_id": cluster_ids,
        }
    )
    clusters_path = args.out_dir / "duplicate_clusters.csv"
    clusters.to_csv(clusters_path, index=False)

    train_mask = split == "train"
    train_clusters = clusters.loc[train_mask]
    n_clusters = int(train_clusters["cluster_id"].nunique())
    sizes = train_clusters.groupby("cluster_id").size()
    mixed = (
        train_clusters.groupby("cluster_id")["appearance"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"train_images={int(train_mask.sum())} test_images={int((~train_mask).sum())}")
    print(f"train_clusters={n_clusters} singletons={int((sizes == 1).sum())} mixed_label_clusters={int(mixed)}")

    train_u = phash_u[train_mask]
    train_labels = train_df["appearance"].to_numpy()
    nearest, dist = nearest_other(train_u)
    agree = (train_labels == train_labels[nearest]).astype(np.int32)
    calibration = pd.DataFrame(
        {
            "filename": train_names,
            "nearest_train": np.array(train_names)[nearest],
            "phash_dist": dist,
            "label": train_labels,
            "nearest_label": train_labels[nearest],
            "agree": agree,
        }
    )
    calibration.to_csv(args.out_dir / "phash_calibration.csv", index=False)
    summary = agreement_by_threshold(dist, agree, [0, 1, 2, 3, 4, 6, 8, 12, 16])
    print("phash nearest-neighbor label agreement:")
    print(summary.to_string(index=False))

    recommended = None
    for _, row in summary.iterrows():
        if row["pairs"] >= 20 and row["label_agreement"] >= 0.99:
            recommended = int(row["threshold"])
    print(f"recommended_phash_threshold={recommended if recommended is not None else args.phash_threshold}")

    y = train_df["appearance"].to_numpy()
    old = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    grouped = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    old_split = next(old.split(train_df["filename"], y))
    grouped_split = next(grouped.split(train_df["filename"], y, train_clusters["cluster_id"].to_numpy()))
    old_leak = leakage_rate(old_split[1], old_split[0], train_clusters["cluster_id"].to_numpy())
    grouped_leak = leakage_rate(grouped_split[1], grouped_split[0], train_clusters["cluster_id"].to_numpy())
    print(f"fold0_near_dup_leakage stratified={old_leak:.4f} grouped={grouped_leak:.4f}")

    test_index = index.iloc[len(train_names) :].reset_index(drop=True)
    train_index = index.iloc[: len(train_names)].reset_index(drop=True)
    train_md5 = set(train_index["md5"])
    md5_hits = int(test_index["md5"].isin(train_md5).sum())
    test_u = phash_u[~train_mask]
    phash_hits = 0
    chunk = 256
    for start in range(0, len(test_u), chunk):
        block = hamming_block(test_u[start : start + chunk], train_u)
        phash_hits += int((block.min(axis=1) <= args.phash_threshold).sum())
    print(
        f"test_coverage md5={md5_hits}/{len(test_u)} ({md5_hits / max(len(test_u), 1):.3f}) "
        f"phash<={args.phash_threshold}={phash_hits}/{len(test_u)} ({phash_hits / max(len(test_u), 1):.3f})"
    )

    dino_summary = None
    if args.use_dino:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            embeddings = extract_dinov2_embeddings(
                all_names,
                image_dir,
                device,
                model_name=args.dino_model,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            train_emb = embeddings[: len(train_names)]
            sims = train_emb @ train_emb.T
            np.fill_diagonal(sims, -1.0)
            dino_nearest = sims.argmax(axis=1)
            dino_sim = sims.max(axis=1)
            dino_agree = (train_labels == train_labels[dino_nearest]).astype(np.int32)
            dino_cal = pd.DataFrame(
                {
                    "filename": train_names,
                    "nearest_train": np.array(train_names)[dino_nearest],
                    "cosine": dino_sim,
                    "label": train_labels,
                    "nearest_label": train_labels[dino_nearest],
                    "agree": dino_agree,
                }
            )
            dino_cal.to_csv(args.out_dir / "dino_calibration.csv", index=False)
            rows = []
            for threshold in [0.99, 0.98, 0.97, 0.96, 0.95, 0.90]:
                mask = dino_sim >= threshold
                n = int(mask.sum())
                rate = float(dino_agree[mask].mean()) if n else float("nan")
                rows.append({"threshold": threshold, "pairs": n, "label_agreement": rate})
            dino_summary = pd.DataFrame(rows)
            print("dino nearest-neighbor label agreement:")
            print(dino_summary.to_string(index=False))
            test_emb = embeddings[len(train_names) :]
            test_sims = test_emb @ train_emb.T
            dino_hits = int((test_sims.max(axis=1) >= args.dino_threshold).sum())
            print(
                f"test_coverage dino>={args.dino_threshold}={dino_hits}/{len(test_u)} "
                f"({dino_hits / max(len(test_u), 1):.3f})"
            )
        except Exception as exc:
            print(f"dino failed ({exc}); continuing with md5+phash only")

    stats = {
        "phash_threshold": args.phash_threshold,
        "recommended_phash_threshold": recommended,
        "train_clusters": n_clusters,
        "mixed_label_clusters": int(mixed),
        "fold0_leakage_stratified": old_leak,
        "fold0_leakage_grouped": grouped_leak,
        "test_md5_hits": md5_hits,
        "test_phash_hits": phash_hits,
        "test_rows": int(len(test_u)),
        "phash_agreement": summary.to_dict(orient="records"),
    }
    if dino_summary is not None:
        stats["dino_agreement"] = dino_summary.to_dict(orient="records")
    (args.out_dir / "duplicate_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {clusters_path}")
    print(f"wrote {args.out_dir / 'duplicate_stats.json'}")


if __name__ == "__main__":
    main()
