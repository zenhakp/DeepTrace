"""PyTorch dataset over the preprocessed face crops."""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from .manifest import SPLITS
from .preprocess import read_faces_csv

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def loading_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    l = cfg.get("loading")
    if not isinstance(l, dict):
        raise ValueError("Invalid configuration: missing 'loading' section")
    d = l.get("degrade")
    if not isinstance(d, dict):
        raise ValueError("Invalid configuration: missing 'loading.degrade' section")
    for key in ("p_blur", "p_resize", "p_jpeg"):
        v = d.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise ValueError(f"Invalid configuration: loading.degrade.{key} must be in [0, 1], got {v!r}")
    for key in ("blur_sigma", "resize_scale", "jpeg_quality"):
        v = d.get(key)
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and v[0] <= v[1]):
            raise ValueError(f"Invalid configuration: loading.degrade.{key} must be [low, high], got {v!r}")
    if not 0 < d["resize_scale"][0] <= d["resize_scale"][1] <= 1:
        raise ValueError("Invalid configuration: loading.degrade.resize_scale must lie in (0, 1]")
    if not 1 <= d["jpeg_quality"][0] and d["jpeg_quality"][1] <= 100:
        raise ValueError("Invalid configuration: loading.degrade.jpeg_quality must lie in [1, 100]")
    for key in ("image_size", "batch_size"):
        v = l.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"Invalid configuration: loading.{key} must be a positive integer, got {v!r}")
    nw = l.get("num_workers", 0)
    if isinstance(nw, bool) or not isinstance(nw, int) or nw < 0:
        raise ValueError(f"Invalid configuration: loading.num_workers must be >= 0, got {nw!r}")
    return {"image_size": l["image_size"], "batch_size": l["batch_size"], "num_workers": nw,
            "hflip": bool(l.get("hflip", True)), "degrade": d}


def degrade(img: Image.Image, p: Dict[str, Any], rng) -> Image.Image:
    """Class-independent quality degradations (blur, down/up-resize, JPEG). The label is not an input."""
    if rng.random() < p["p_blur"]:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*p["blur_sigma"])))
    if rng.random() < p["p_resize"]:
        s = rng.uniform(*p["resize_scale"])
        w, h = img.size
        small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.Resampling.BILINEAR)
        img = small.resize((w, h), Image.Resampling.BICUBIC)
    if rng.random() < p["p_jpeg"]:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(*[int(q) for q in p["jpeg_quality"]]))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def load_split_rows(faces_csv, split: str) -> List[Dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}'. Expected one of {SPLITS}")
    if not Path(faces_csv).is_file():
        raise FileNotFoundError(f"Faces manifest not found: {faces_csv}. Run scripts/preprocess_faces.py first.")
    rows = [r for r in read_faces_csv(faces_csv) if r["kept"] == 1 and r["split"] == split]
    if not rows:
        raise ValueError(f"No kept rows for split '{split}' in {faces_csv}")
    return rows


def class_counts(rows: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    return {0: sum(r["label"] == 0 for r in rows), 1: sum(r["label"] == 1 for r in rows)}


class FaceCropDataset(Dataset):
    """Returns (normalized CHW float tensor, label 0=REAL / 1=MANIPULATED, image_id)."""

    def __init__(self, rows: Sequence[Dict[str, Any]], crops_dir, image_size: int = 380,
                 train: bool = False, degrade_cfg: Optional[Dict[str, Any]] = None, hflip: bool = True):
        self.rows = list(rows)
        self.crops_dir = Path(crops_dir)
        self.image_size, self.train, self.degrade_cfg, self.hflip = image_size, train, degrade_cfg, hflip
        missing = [r["out_path"] for r in self.rows if not (self.crops_dir / r["out_path"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Face crop not found: {len(missing)} of {len(self.rows)} files are missing in "
                f"{self.crops_dir}, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        r = self.rows[i]
        path = self.crops_dir / r["out_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Unable to read image: {path}") from exc
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if self.train:
            if self.hflip and random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.degrade_cfg:
                img = degrade(img, self.degrade_cfg, random)   # DataLoader seeds `random` per worker
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()), int(r["label"]), r["image_id"]


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 42) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=g, persistent_workers=num_workers > 0)
