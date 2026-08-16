from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from common import find_image_dir
from predict_multi_ensemble import load_model, predict_batch_probs
from torch.utils.data import DataLoader
from common import TomJerryDataset, make_eval_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=None)
    args = parser.parse_args()

    if not args.data_dir.exists():
        # Fallback to local
        args.data_dir = Path("data")

    # Find all .pt files
    if args.checkpoints is None or len(args.checkpoints) == 0:
        checkpoints = list(Path.cwd().rglob("*.pt")) + list(Path("/kaggle/working").rglob("*.pt"))
        checkpoints = list(set(checkpoints))
    else:
        checkpoints = args.checkpoints

    print(f"[*] Found {len(checkpoints)} checkpoint(s):\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)

    for cp in sorted(checkpoints, key=lambda x: str(x)):
        try:
            model, meta = load_model(cp, device)
            arch = meta.get("arch", "unknown")
            fold = meta.get("fold", "?")
            size = meta.get("image_size", "?")
            f1 = meta.get("best_f1", "?")
            print(f"--> File: {cp}")
            print(f"    Arch: {arch} | Fold: {fold} | Size: {size} | Best Val F1: {f1}")

            # Quick prediction check
            ds = TomJerryDataset(test_df.head(200), image_dir, make_eval_transform(size if isinstance(size, int) else 300), labeled=False)
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            all_preds = []
            with torch.no_grad():
                for images, _ in loader:
                    probs = predict_batch_probs(model, images, meta, device)
                    all_preds.extend(probs.argmax(dim=1).cpu().numpy().tolist())
            dist = pd.Series(all_preds).value_counts().to_dict()
            print(f"    Sample Pred Distribution (First 200 images): {dist}\n")
            del model
        except Exception as e:
            print(f"    [!] Error loading {cp}: {e}\n")


if __name__ == "__main__":
    main()
