from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from common import (
    CharacterPresenceModel,
    TomJerryDataset,
    class_logits_from_character_logits,
    find_image_dir,
    make_eval_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def predict_logits(model, images, device, blend_aux, hflip=False):
    images = images.to(device, non_blocking=True)
    if hflip:
        images = torch.flip(images, dims=[3])
    class_logits, character_logits = model(images)
    return (1.0 - blend_aux) * class_logits + blend_aux * class_logits_from_character_logits(character_logits)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CharacterPresenceModel(checkpoint["arch"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)
    ds = TomJerryDataset(test_df, image_dir, make_eval_transform(checkpoint["image_size"]), labeled=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    filenames, preds = [], []
    with torch.no_grad():
        for images, batch_filenames in tqdm(loader, desc="predict"):
            logits = predict_logits(model, images, device, checkpoint.get("blend_aux", 0.25), hflip=False)
            if args.tta:
                logits = (logits + predict_logits(model, images, device, checkpoint.get("blend_aux", 0.25), hflip=True)) / 2
            batch_preds = logits.argmax(dim=1).cpu().numpy().tolist()
            filenames.extend(batch_filenames)
            preds.extend(batch_preds)

    submission = pd.DataFrame({"filename": filenames, "appearance": preds})
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(submission)}")


if __name__ == "__main__":
    main()
