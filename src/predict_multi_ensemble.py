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
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/submission_multi_ensemble.csv"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CharacterPresenceModel(checkpoint["arch"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_batch_probs(model, images, checkpoint, device, hflip=False):
    images = images.to(device, non_blocking=True)
    if hflip:
        images = torch.flip(images, dims=[3])
    class_logits, character_logits = model(images)
    blend_aux = checkpoint.get("blend_aux", 0.25)
    logits = (1.0 - blend_aux) * class_logits + blend_aux * class_logits_from_character_logits(character_logits)
    return torch.softmax(logits, dim=1)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.weights is None or len(args.weights) == 0:
        weights = [1.0] * len(args.checkpoints)
    elif len(args.weights) == len(args.checkpoints):
        weights = args.weights
    else:
        raise ValueError("--weights must be omitted or have the same length as --checkpoints")

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("--weights must sum to a positive value")
    weights = [weight / weight_sum for weight in weights]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)
    prob_sum = torch.zeros((len(test_df), 4), dtype=torch.float32)

    for checkpoint_path, weight in zip(args.checkpoints, weights):
        model, checkpoint = load_model(checkpoint_path, device)
        image_size = checkpoint["image_size"]
        ds = TomJerryDataset(test_df, image_dir, make_eval_transform(image_size), labeled=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        offset = 0
        desc = f"{checkpoint['arch']}_fold{checkpoint.get('fold', '?')}_{image_size}"
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=desc):
                probs = predict_batch_probs(model, images, checkpoint, device, hflip=False)
                if args.tta:
                    probs = (probs + predict_batch_probs(model, images, checkpoint, device, hflip=True)) / 2

                batch_size = probs.size(0)
                prob_sum[offset : offset + batch_size] += probs.cpu() * weight
                offset += batch_size

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preds = prob_sum.argmax(dim=1).numpy().tolist()
    submission = pd.DataFrame({"filename": test_df["filename"], "appearance": preds})
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(submission)} checkpoints={len(args.checkpoints)}")


if __name__ == "__main__":
    main()
