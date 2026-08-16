from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
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
    parser.add_argument("--out", type=Path, default=Path("outputs/submission_knn.csv"))
    parser.add_argument("--diagnostics-out", type=Path, default=Path("outputs/knn_diagnostics.csv"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--model-prob-weight", type=float, default=0.25)
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CharacterPresenceModel(checkpoint["arch"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def forward_features_and_probs(model, images, checkpoint, device, hflip=False):
    images = images.to(device, non_blocking=True)
    if hflip:
        images = torch.flip(images, dims=[3])

    features = F.normalize(model.backbone(images), dim=1)
    class_logits, character_logits = model(images)
    blend_aux = checkpoint.get("blend_aux", 0.25)
    logits = (1.0 - blend_aux) * class_logits + blend_aux * class_logits_from_character_logits(character_logits)
    probs = torch.softmax(logits, dim=1)
    return features, probs


def extract_features_and_probs(model, checkpoint, frame, image_dir, device, args, labeled):
    ds = TomJerryDataset(frame, image_dir, make_eval_transform(checkpoint["image_size"]), labeled=labeled)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_features = []
    all_probs = []
    all_filenames = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"embed_{checkpoint['arch']}_fold{checkpoint.get('fold', '?')}"):
            if labeled:
                images = batch[0]
                filenames = None
            else:
                images, filenames = batch

            features, probs = forward_features_and_probs(model, images, checkpoint, device, hflip=False)
            if args.tta:
                flip_features, flip_probs = forward_features_and_probs(model, images, checkpoint, device, hflip=True)
                features = F.normalize(features + flip_features, dim=1)
                probs = (probs + flip_probs) / 2

            all_features.append(features.cpu())
            all_probs.append(probs.cpu())
            if filenames is not None:
                all_filenames.extend(filenames)

    features = torch.cat(all_features, dim=0)
    probs = torch.cat(all_probs, dim=0)
    return features, probs, all_filenames


def knn_probs(train_features, train_labels, test_features, k, temperature, chunk_size=512):
    k = min(k, train_features.size(0))
    train_labels = train_labels.long()
    all_probs = []
    all_top_indices = []
    all_top_sims = []

    for start in tqdm(range(0, test_features.size(0), chunk_size), desc="knn"):
        end = min(start + chunk_size, test_features.size(0))
        sims = test_features[start:end] @ train_features.T
        top_sims, top_indices = sims.topk(k=k, dim=1)
        top_labels = train_labels[top_indices]
        weights = torch.softmax(top_sims / temperature, dim=1)

        probs = torch.zeros((end - start, 4), dtype=torch.float32)
        probs.scatter_add_(1, top_labels, weights)
        all_probs.append(probs)
        all_top_indices.append(top_indices[:, 0].cpu())
        all_top_sims.append(top_sims[:, 0].cpu())

    return torch.cat(all_probs), torch.cat(all_top_indices), torch.cat(all_top_sims)


def normalize_weights(weights, n):
    if weights is None or len(weights) == 0:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError("--weights must be omitted or have the same length as --checkpoints")

    total = sum(weights)
    if total <= 0:
        raise ValueError("--weights must sum to a positive value")
    return [weight / total for weight in weights]


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.model_prob_weight <= 1:
        raise ValueError("--model-prob-weight must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    image_dir = find_image_dir(args.data_dir)
    train_labels = torch.tensor(train_df["appearance"].to_numpy(), dtype=torch.long)

    checkpoint_weights = normalize_weights(args.weights, len(args.checkpoints))
    final_probs = torch.zeros((len(test_df), 4), dtype=torch.float32)
    diagnostics = pd.DataFrame({"filename": test_df["filename"]})

    for model_index, (checkpoint_path, checkpoint_weight) in enumerate(zip(args.checkpoints, checkpoint_weights)):
        model, checkpoint = load_model(checkpoint_path, device)
        train_features, _, _ = extract_features_and_probs(model, checkpoint, train_df, image_dir, device, args, labeled=True)
        test_features, test_model_probs, _ = extract_features_and_probs(model, checkpoint, test_df, image_dir, device, args, labeled=False)

        train_features = F.normalize(train_features, dim=1)
        test_features = F.normalize(test_features, dim=1)
        neighbor_probs, top_indices, top_sims = knn_probs(
            train_features,
            train_labels,
            test_features,
            k=args.k,
            temperature=args.temperature,
        )

        blended_probs = (1.0 - args.model_prob_weight) * neighbor_probs + args.model_prob_weight * test_model_probs
        final_probs += blended_probs * checkpoint_weight

        prefix = f"model{model_index}_{checkpoint['arch']}_fold{checkpoint.get('fold', '?')}"
        diagnostics[f"{prefix}_nearest_train"] = train_df.iloc[top_indices.numpy()]["filename"].to_numpy()
        diagnostics[f"{prefix}_nearest_label"] = train_labels[top_indices].numpy()
        diagnostics[f"{prefix}_nearest_similarity"] = top_sims.numpy()
        diagnostics[f"{prefix}_knn_pred"] = neighbor_probs.argmax(dim=1).numpy()
        diagnostics[f"{prefix}_blend_pred"] = blended_probs.argmax(dim=1).numpy()
        diagnostics[f"{prefix}_blend_confidence"] = blended_probs.max(dim=1).values.numpy()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    preds = final_probs.argmax(dim=1).numpy().tolist()
    submission = pd.DataFrame({"filename": test_df["filename"], "appearance": preds})
    submission.to_csv(args.out, index=False)
    diagnostics["final_pred"] = preds
    diagnostics["final_confidence"] = final_probs.max(dim=1).values.numpy()
    diagnostics.to_csv(args.diagnostics_out, index=False)

    print(f"wrote {args.out} rows={len(submission)}")
    print(f"wrote {args.diagnostics_out} rows={len(diagnostics)}")


if __name__ == "__main__":
    main()
