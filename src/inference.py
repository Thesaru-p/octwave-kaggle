from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from common import (
    CharacterPresenceModel,
    TomJerryDataset,
    class_logits_from_character_logits,
    make_eval_transform,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def normalize_weights(weights, n: int) -> list[float]:
    if weights is None or len(weights) == 0:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError("--weights must be omitted or have the same length as --checkpoints")
    total = sum(weights)
    if total <= 0:
        raise ValueError("--weights must sum to a positive value")
    return [weight / total for weight in weights]


def load_model(checkpoint_path: Path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CharacterPresenceModel(checkpoint["arch"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def blended_logits(model, images, checkpoint, device, hflip: bool = False):
    images = images.to(device, non_blocking=True)
    if hflip:
        images = torch.flip(images, dims=[3])
    class_logits, character_logits = model(images)
    blend_aux = checkpoint.get("blend_aux", 0.25)
    return (1.0 - blend_aux) * class_logits + blend_aux * class_logits_from_character_logits(character_logits)


def predict_batch_probs(model, images, checkpoint, device, hflip: bool = False):
    return torch.softmax(blended_logits(model, images, checkpoint, device, hflip=hflip), dim=1)


def tta_sizes(image_size: int, tta_multiscale: bool) -> list[int]:
    sizes = [image_size]
    if tta_multiscale:
        extra = int(round(image_size * 1.14))
        if extra != image_size:
            sizes.append(extra)
    return sizes


def load_decision_config(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def apply_decision(probs: torch.Tensor, config: dict | None) -> tuple[torch.Tensor, np.ndarray]:
    if config is None:
        return probs, probs.argmax(dim=1).cpu().numpy()

    method = config.get("method", "argmax")
    if method == "bias":
        bias = torch.tensor(config["bias"], dtype=probs.dtype, device=probs.device)
        logits = torch.log(probs.clamp_min(1e-8)) + bias
        probs = torch.softmax(logits, dim=1)
        return probs, probs.argmax(dim=1).cpu().numpy()
    if method == "multipliers":
        multipliers = torch.tensor(config["multipliers"], dtype=probs.dtype, device=probs.device)
        scaled = probs * multipliers
        probs = scaled / scaled.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return probs, probs.argmax(dim=1).cpu().numpy()
    if method == "binary_thresholds":
        thresholds = config["binary_thresholds"]
        p_tom = probs[:, 1] + probs[:, 3]
        p_jerry = probs[:, 2] + probs[:, 3]
        is_tom = (p_tom >= float(thresholds["tom"])).to(torch.long)
        is_jerry = (p_jerry >= float(thresholds["jerry"])).to(torch.long)
        preds = (2 * is_jerry + is_tom).cpu().numpy()
        return probs, preds
    return probs, probs.argmax(dim=1).cpu().numpy()


def predict_frame_probs(
    checkpoints: list[Path],
    frame: pd.DataFrame,
    image_dir: Path,
    device,
    weights=None,
    tta: bool = False,
    tta_multiscale: bool = False,
    batch_size: int = 24,
    num_workers: int = 2,
) -> torch.Tensor:
    checkpoint_weights = normalize_weights(weights, len(checkpoints))
    prob_sum = torch.zeros((len(frame), 4), dtype=torch.float32)

    for checkpoint_path, weight in zip(checkpoints, checkpoint_weights):
        model, checkpoint = load_model(checkpoint_path, device)
        sizes = tta_sizes(int(checkpoint["image_size"]), tta_multiscale)
        acc = torch.zeros((len(frame), 4), dtype=torch.float32)
        for image_size in sizes:
            dataset = TomJerryDataset(frame, image_dir, make_eval_transform(image_size), labeled=False)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            offset = 0
            desc = f"{checkpoint['arch']}_fold{checkpoint.get('fold', '?')}_{image_size}"
            with torch.no_grad():
                for images, _ in tqdm(loader, desc=desc):
                    probs = predict_batch_probs(model, images, checkpoint, device, hflip=False)
                    if tta:
                        probs = (probs + predict_batch_probs(model, images, checkpoint, device, hflip=True)) / 2
                    batch = probs.size(0)
                    acc[offset : offset + batch] += probs.cpu()
                    offset += batch
        acc = acc / len(sizes)
        prob_sum += acc * weight
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return prob_sum
