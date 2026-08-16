from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from common import find_image_dir
from inference import apply_decision, load_decision_config, predict_frame_probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/submission_multi_ensemble.csv"))
    parser.add_argument("--decision-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-multiscale", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)

    probs = predict_frame_probs(
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
    probs, preds = apply_decision(probs, load_decision_config(args.decision_json))
    submission = pd.DataFrame({"filename": test_df["filename"], "appearance": preds})
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(submission)} checkpoints={len(args.checkpoints)}")


if __name__ == "__main__":
    main()
