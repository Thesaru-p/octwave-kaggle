from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import find_image_dir
from duplicates import assign_matches, extract_dinov2_embeddings, index_images
from inference import apply_decision, load_decision_config, predict_frame_probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/submission_match.csv"))
    parser.add_argument("--diagnostics-out", type=Path, default=Path("outputs/match_diagnostics.csv"))
    parser.add_argument("--decision-json", type=Path, default=None)
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--dino-threshold", type=float, default=0.97)
    parser.add_argument("--use-dino", action="store_true")
    parser.add_argument("--dino-model", type=str, default="dinov2_vitb14")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-multiscale", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)

    train_names = train_df["filename"].tolist()
    test_names = test_df["filename"].tolist()
    index = index_images(train_names + test_names, image_dir, cache_path=args.out_dir / "hash_index.csv")
    index = index.drop_duplicates("filename").set_index("filename")
    train_index = index.loc[train_names].reset_index()
    test_index = index.loc[test_names].reset_index()

    test_dino = train_dino = None
    if args.use_dino:
        device_emb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            embeddings = extract_dinov2_embeddings(
                train_names + test_names,
                image_dir,
                device_emb,
                model_name=args.dino_model,
                batch_size=max(args.batch_size, 16),
                num_workers=args.num_workers,
            )
            train_dino = embeddings[: len(train_names)]
            test_dino = embeddings[len(train_names) :]
        except Exception as exc:
            print(f"dino failed ({exc}); matching with md5+phash only")

    assignments = assign_matches(
        test_index,
        train_index,
        train_df["appearance"].to_numpy(),
        phash_threshold=args.phash_threshold,
        dino_threshold=args.dino_threshold if train_dino is not None else None,
        test_dino=test_dino,
        train_dino=train_dino,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_probs = predict_frame_probs(
        args.checkpoints,
        test_df,
        image_dir,
        device,
        weights=args.weights,
        tta=args.tta,
        tta_multiscale=args.tta_multiscale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    decision = load_decision_config(args.decision_json)
    model_probs, model_preds = apply_decision(model_probs, decision)
    model_confidence = model_probs.max(dim=1).values.numpy()

    preds = []
    sources = []
    for i, assignment in enumerate(assignments):
        if assignment.source != "none" and assignment.label is not None:
            preds.append(int(assignment.label))
            sources.append(assignment.source)
        else:
            preds.append(int(model_preds[i]))
            sources.append("model")

    submission = pd.DataFrame({"filename": test_df["filename"], "appearance": preds})
    submission.to_csv(args.out, index=False)

    diagnostics = pd.DataFrame(
        {
            "filename": test_df["filename"],
            "appearance": preds,
            "match_source": sources,
            "nearest_train": [a.train_filename for a in assignments],
            "phash_dist": [a.phash_dist for a in assignments],
            "dino_cosine": [a.dino_cosine for a in assignments],
            "n_within": [a.n_within for a in assignments],
            "labels_agree": [a.labels_agree for a in assignments],
            "model_pred": model_preds,
            "model_confidence": model_confidence,
        }
    )
    diagnostics["changed"] = diagnostics["appearance"] != diagnostics["model_pred"]
    diagnostics.to_csv(args.diagnostics_out, index=False)

    counts = diagnostics["match_source"].value_counts()
    print("match_source counts:")
    print(counts.to_string())
    changed = int(diagnostics["changed"].sum())
    print(f"model_predictions_overridden={changed}/{len(diagnostics)}")
    print(f"wrote {args.out} rows={len(submission)}")
    print(f"wrote {args.diagnostics_out}")


if __name__ == "__main__":
    main()
