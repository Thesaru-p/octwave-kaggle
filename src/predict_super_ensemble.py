from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    CharacterPresenceModel,
    TomJerryDataset,
    class_logits_from_character_logits,
    find_image_dir,
    make_eval_transform,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--power", type=float, default=1.5, help="Power-averaging exponent")
    parser.add_argument("--out", type=Path, default=Path("outputs/submission_super_ensemble.csv"))
    parser.add_argument("--probs-out", type=Path, default=Path("outputs/test_probs_super_ensemble.csv"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--calibrated-json", type=Path, default=None, help="Path to calibrated thresholds JSON")
    parser.add_argument("--multipliers", type=float, nargs=4, default=None, help="Direct class multipliers [w0, w1, w2, w3]")
    parser.add_argument("--char-thresholds", type=float, nargs=2, default=None, help="Binary thresholds [tau_tom, tau_jerry]")
    return parser.parse_args()


def normalize_weights(weights, n):
    if weights is None or len(weights) == 0:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError("--weights must have the same length as --checkpoints")
    total = sum(weights)
    if total <= 0:
        raise ValueError("--weights must sum to a positive value")
    return [w / total for w in weights]


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("arch", "efficientnet_b3")
    model = CharacterPresenceModel(arch, pretrained=False).to(device)
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
    probs = torch.softmax(logits, dim=1)
    
    # Also compute direct character probabilities
    tom_prob = torch.sigmoid(character_logits[:, 0])
    jerry_prob = torch.sigmoid(character_logits[:, 1])
    return probs, tom_prob, jerry_prob


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.probs_out is not None:
        args.probs_out.parent.mkdir(parents=True, exist_ok=True)

    weights = normalize_weights(args.weights, len(args.checkpoints))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)

    power = args.power
    powered_prob_sum = torch.zeros((len(test_df), 4), dtype=torch.float32)
    tom_prob_sum = torch.zeros(len(test_df), dtype=torch.float32)
    jerry_prob_sum = torch.zeros(len(test_df), dtype=torch.float32)

    print(f"[*] Starting Super Ensemble with {len(args.checkpoints)} checkpoints on {device}...")

    for checkpoint_path, weight in zip(args.checkpoints, weights):
        model, checkpoint = load_model(checkpoint_path, device)
        image_size = checkpoint.get("image_size", 300)
        arch = checkpoint.get("arch", "unknown")
        fold = checkpoint.get("fold", "?")
        desc = f"{arch}_f{fold}_{image_size}px (w={weight:.2f})"

        ds = TomJerryDataset(test_df, image_dir, make_eval_transform(image_size), labeled=False)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        offset = 0
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=desc):
                probs, p_tom, p_jerry = predict_batch_probs(model, images, checkpoint, device, hflip=False)
                if args.tta:
                    flip_probs, flip_tom, flip_jerry = predict_batch_probs(model, images, checkpoint, device, hflip=True)
                    probs = (probs + flip_probs) / 2.0
                    p_tom = (p_tom + flip_tom) / 2.0
                    p_jerry = (p_jerry + flip_jerry) / 2.0

                bs = probs.size(0)
                # Power weighted summation
                powered_prob_sum[offset : offset + bs] += (probs.cpu() ** power) * weight
                tom_prob_sum[offset : offset + bs] += p_tom.cpu() * weight
                jerry_prob_sum[offset : offset + bs] += p_jerry.cpu() * weight
                offset += bs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Reconstruct final ensembled probabilities
    final_probs = (powered_prob_sum) ** (1.0 / power)
    final_probs = final_probs / final_probs.sum(dim=1, keepdim=True)
    final_probs_np = final_probs.numpy()

    # Determine calibration method & parameters
    multipliers = None
    char_thresholds = None

    if args.calibrated_json is not None and args.calibrated_json.exists():
        with open(args.calibrated_json, "r") as f:
            calib = json.load(f)
        if "multipliers" in calib:
            multipliers = np.array(calib["multipliers"])
        if "binary_thresholds" in calib:
            char_thresholds = (calib["binary_thresholds"]["tom"], calib["binary_thresholds"]["jerry"])
        print(f"[+] Loaded calibration from {args.calibrated_json}")
    elif args.multipliers is not None:
        multipliers = np.array(args.multipliers)
    elif args.char_thresholds is not None:
        char_thresholds = tuple(args.char_thresholds)
    else:
        # High-performance default multipliers based on extensive OOF optimization
        multipliers = np.array([0.22, 0.28, 0.27, 0.23])
        print("[+] Applied optimal OOF class prior multipliers: [0.22, 0.28, 0.27, 0.23]")

    # Compute final predictions
    if char_thresholds is not None:
        t_tom, t_jerry = char_thresholds
        p_t = tom_prob_sum.numpy()
        p_j = jerry_prob_sum.numpy()
        is_tom = (p_t >= t_tom).astype(int)
        is_jerry = (p_j >= t_jerry).astype(int)
        preds = 2 * is_jerry + is_tom
        print(f"[+] Predicted using Binary Thresholds: Tom >= {t_tom:.3f}, Jerry >= {t_jerry:.3f}")
    else:
        scaled_probs = final_probs_np * multipliers
        preds = np.argmax(scaled_probs, axis=1)
        print(f"[+] Predicted using Class Multipliers: {multipliers.round(4).tolist()}")

    # Save submission
    submission = pd.DataFrame({"filename": test_df["filename"], "appearance": preds})
    submission.to_csv(args.out, index=False)
    print(f"\n[✓] Generated Submission: {args.out} ({len(submission)} rows)")
    print("\nPredicted Class Distribution:")
    print(submission["appearance"].value_counts().sort_index().to_string())

    # Save probability diagnostics
    if args.probs_out is not None:
        confidences = np.max(final_probs_np, axis=1)
        top2 = np.partition(final_probs_np, -2, axis=1)[:, -2:]
        margins = top2[:, 1] - top2[:, 0]

        probs_df = pd.DataFrame(
            {
                "filename": test_df["filename"],
                "appearance": preds,
                "confidence": confidences,
                "margin": margins,
                "prob_0": final_probs_np[:, 0],
                "prob_1": final_probs_np[:, 1],
                "prob_2": final_probs_np[:, 2],
                "prob_3": final_probs_np[:, 3],
                "prob_tom": tom_prob_sum.numpy(),
                "prob_jerry": jerry_prob_sum.numpy(),
            }
        )
        probs_df.to_csv(args.probs_out, index=False)
        print(f"[✓] Saved Probability Diagnostics: {args.probs_out}")


if __name__ == "__main__":
    main()
