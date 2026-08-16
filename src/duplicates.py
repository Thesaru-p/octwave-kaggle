from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


try:
    import imagehash
except ImportError as exc:  # pragma: no cover
    raise ImportError("Phase 1 matching requires ImageHash. Install it with: pip install ImageHash") from exc


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int32)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_phash_hex(path: Path, hash_size: int = 8) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return str(imagehash.phash(image, hash_size=hash_size))


def hex_to_uint64(hex_hash: str) -> np.uint64:
    return np.uint64(int(hex_hash, 16))


def popcount_uint64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    packed = np.ascontiguousarray(values).view(np.uint8).reshape(*values.shape, 8)
    return np.unpackbits(packed, axis=-1).sum(axis=-1).astype(np.int16)


def hamming_block(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(query[:, None], gallery[None, :])
    return popcount_uint64(xor)


def index_images(
    filenames: list[str],
    image_dir: Path,
    cache_path: Path | None = None,
    hash_size: int = 8,
) -> pd.DataFrame:
    cached = None
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path).drop_duplicates("filename").set_index("filename")

    rows = []
    for filename in tqdm(filenames, desc="hash"):
        if cached is not None and filename in cached.index:
            row = cached.loc[filename]
            rows.append(
                {
                    "filename": filename,
                    "md5": row["md5"],
                    "phash": row["phash"],
                }
            )
            continue
        path = image_dir / filename
        rows.append(
            {
                "filename": filename,
                "md5": file_md5(path),
                "phash": file_phash_hex(path, hash_size=hash_size),
            }
        )

    index = pd.DataFrame(rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(cache_path, index=False)
    return index


def cluster_phash(phash_uint64: np.ndarray, threshold: int) -> np.ndarray:
    n = len(phash_uint64)
    uf = UnionFind(n)
    chunk = 256
    for start in tqdm(range(0, n, chunk), desc="cluster"):
        end = min(start + chunk, n)
        dist = hamming_block(phash_uint64[start:end], phash_uint64)
        for local_i, i in enumerate(range(start, end)):
            neighbors = np.flatnonzero(dist[local_i] <= threshold)
            for j in neighbors:
                if j > i:
                    uf.union(i, int(j))
    roots = uf.components()
    _, cluster_ids = np.unique(roots, return_inverse=True)
    return cluster_ids.astype(np.int32)


def nearest_other(phash_uint64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(phash_uint64)
    nearest = np.empty(n, dtype=np.int32)
    dist = np.empty(n, dtype=np.int16)
    chunk = 256
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = hamming_block(phash_uint64[start:end], phash_uint64)
        for local_i, i in enumerate(range(start, end)):
            block[local_i, i] = 999
            j = int(block[local_i].argmin())
            nearest[i] = j
            dist[i] = block[local_i, j]
    return nearest, dist


def agreement_by_threshold(distances: np.ndarray, agree: np.ndarray, thresholds: list[int]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        mask = distances <= threshold
        n = int(mask.sum())
        rate = float(agree[mask].mean()) if n else float("nan")
        rows.append({"threshold": threshold, "pairs": n, "label_agreement": rate})
    return pd.DataFrame(rows)


def extract_dinov2_embeddings(
    filenames: list[str],
    image_dir: Path,
    device,
    model_name: str = "dinov2_vitb14",
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
):
    import torch
    import torch.nn.functional as F
    from common import TomJerryDataset, make_eval_transform
    from torch.utils.data import DataLoader

    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True, trust_repo=True)
    except TypeError:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model.to(device).eval()

    frame = pd.DataFrame({"filename": filenames})
    dataset = TomJerryDataset(frame, image_dir, make_eval_transform(image_size), labeled=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"dino_{model_name}"):
            features = model(images.to(device, non_blocking=True))
            chunks.append(F.normalize(features.float(), dim=1).cpu())
    return torch.cat(chunks, dim=0).numpy()


@dataclass
class MatchAssignment:
    source: str
    train_filename: str | None
    label: int | None
    phash_dist: int
    dino_cosine: float
    n_within: int
    labels_agree: bool


def assign_matches(
    test_index: pd.DataFrame,
    train_index: pd.DataFrame,
    train_labels: np.ndarray,
    phash_threshold: int,
    dino_threshold: float | None = None,
    test_dino: np.ndarray | None = None,
    train_dino: np.ndarray | None = None,
) -> list[MatchAssignment]:
    train_md5 = defaultdict(list)
    for i, md5 in enumerate(train_index["md5"].tolist()):
        train_md5[md5].append(i)

    train_u = np.array([hex_to_uint64(h) for h in train_index["phash"]], dtype=np.uint64)
    test_u = np.array([hex_to_uint64(h) for h in test_index["phash"]], dtype=np.uint64)
    train_names = train_index["filename"].to_numpy()
    train_labels = np.asarray(train_labels)

    n_test = len(test_index)
    assignments = [
        MatchAssignment("none", None, None, 999, -1.0, 0, False) for _ in range(n_test)
    ]

    chunk = 256
    for start in tqdm(range(0, n_test, chunk), desc="match_phash"):
        end = min(start + chunk, n_test)
        dist = hamming_block(test_u[start:end], train_u)
        for local_i, i in enumerate(range(start, end)):
            md5_hits = train_md5.get(test_index.iloc[i]["md5"], [])
            if md5_hits:
                labels = train_labels[md5_hits]
                unique = np.unique(labels)
                nearest = md5_hits[0]
                if len(unique) == 1:
                    assignments[i] = MatchAssignment(
                        "md5",
                        str(train_names[nearest]),
                        int(unique[0]),
                        0,
                        -1.0,
                        len(md5_hits),
                        True,
                    )
                    continue
            nearest = int(dist[local_i].argmin())
            min_dist = int(dist[local_i, nearest])
            within = np.flatnonzero(dist[local_i] <= phash_threshold)
            assignments[i].phash_dist = min_dist
            assignments[i].n_within = int(len(within))
            if len(within) == 0:
                continue
            labels = train_labels[within]
            unique = np.unique(labels)
            agree = len(unique) == 1
            assignments[i].labels_agree = bool(agree)
            if agree:
                best = int(within[np.argmin(dist[local_i, within])])
                assignments[i] = MatchAssignment(
                    "phash",
                    str(train_names[best]),
                    int(train_labels[best]),
                    int(dist[local_i, best]),
                    -1.0,
                    int(len(within)),
                    True,
                )

    if test_dino is None or train_dino is None or dino_threshold is None:
        return assignments

    sims = test_dino @ train_dino.T
    for i, current in enumerate(assignments):
        if current.source != "none":
            current.dino_cosine = float(sims[i].max())
            continue
        nearest = int(sims[i].argmax())
        max_sim = float(sims[i, nearest])
        within = np.flatnonzero(sims[i] >= dino_threshold)
        current.dino_cosine = max_sim
        current.n_within = int(len(within))
        if len(within) == 0:
            continue
        labels = train_labels[within]
        unique = np.unique(labels)
        current.labels_agree = bool(len(unique) == 1)
        if len(unique) == 1:
            best = int(within[np.argmax(sims[i, within])])
            assignments[i] = MatchAssignment(
                "dino",
                str(train_names[best]),
                int(train_labels[best]),
                current.phash_dist,
                float(sims[i, best]),
                int(len(within)),
                True,
            )
        else:
            current.dino_cosine = max_sim
    return assignments
