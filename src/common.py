from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


SUPPORTED_ARCHES = [
    "convnext_tiny",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_v2_s",
]


LABEL_TO_CHARACTERS = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (0.0, 1.0),
    3: (1.0, 1.0),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_image_dir(data_dir: Path) -> Path:
    candidates = [data_dir / "images", data_dir / "images" / "images"]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.jpg")):
            return candidate
    raise FileNotFoundError(f"Could not find jpg images under {data_dir}")


def make_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.22,
                        contrast=0.22,
                        saturation=0.18,
                        hue=0.035,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomRotation(degrees=8),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.18, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )


def make_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class TomJerryDataset(Dataset):
    def __init__(self, frame, image_dir: Path, transform=None, labeled: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        filename = row["filename"]
        image = Image.open(self.image_dir / filename).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if not self.labeled:
            return image, filename

        label = int(row["appearance"])
        character_targets = torch.tensor(LABEL_TO_CHARACTERS[label], dtype=torch.float32)
        return image, torch.tensor(label, dtype=torch.long), character_targets


class CharacterPresenceModel(nn.Module):
    def __init__(self, arch: str, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        self.arch = arch
        weights = "DEFAULT" if pretrained else None

        if arch == "convnext_tiny":
            base = models.convnext_tiny(weights=weights)
            feature_dim = base.classifier[2].in_features
            self.backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten(1))
        elif arch in {
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_v2_s",
        }:
            base = getattr(models, arch)(weights=weights)
            feature_dim = base.classifier[1].in_features
            self.backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten(1))
        else:
            raise ValueError(f"Unsupported arch: {arch}. Choose one of: {SUPPORTED_ARCHES}")

        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(feature_dim, 4)
        self.character_head = nn.Linear(feature_dim, 2)

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return self.class_head(features), self.character_head(features)


def class_logits_from_character_logits(character_logits: torch.Tensor) -> torch.Tensor:
    tom = torch.sigmoid(character_logits[:, 0])
    jerry = torch.sigmoid(character_logits[:, 1])
    probs = torch.stack(
        [
            (1.0 - tom) * (1.0 - jerry),
            tom * (1.0 - jerry),
            (1.0 - tom) * jerry,
            tom * jerry,
        ],
        dim=1,
    )
    return torch.log(probs.clamp_min(1e-6))
