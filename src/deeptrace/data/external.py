"""Face-crop preprocessing for EXTERNAL (cross-dataset) images, using the same pipeline as training.

By necessity two things differ from training: external images are unpaired, so there is no group colour
matching or group filtering, and a minimum native crop size drops tiny faces. Every image becomes one
manifest row with split='test'. These images must never be used for training.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np
from PIL import Image, ImageOps

from .faces import mean_chroma, q_lum_mean, sharpness, square_crop_box
from .inspection import IMG_EXT
from .preprocess import FACE_FIELDS, shortcut_audit


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "unknown"


def folder_items(root, label: int, method: str) -> Iterator[Dict[str, Any]]:
    root = Path(root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    for p in paths:
        yield {"key": f"{method}:{p.relative_to(root).as_posix()}", "label": label, "method": method,
               "origin": str(p), "read": p.read_bytes}


def parquet_items(paths: Iterable, image_col: str = "image", label_col: str = "label",
                  model_col: str = "model", real_value: str = "real", batch_size: int = 64) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in paths:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        missing = [c for c in (image_col, label_col, model_col) if c not in names]
        if missing:
            raise ValueError(f"Parquet file {path} lacks columns {missing}; available: {names}")
        idx = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=[image_col, label_col, model_col]):
            for im, lab, mod in zip(batch.column(image_col).to_pylist(), batch.column(label_col).to_pylist(),
                                    batch.column(model_col).to_pylist()):
                raw = im["bytes"] if isinstance(im, dict) else im
                is_real = str(lab).lower() == real_value
                yield {"key": f"{Path(path).name}:{idx}", "label": 0 if is_real else 1,
                       "method": "real" if is_real else slug(mod), "origin": f"{Path(path).name}#{idx}",
                       "read": (lambda b=raw: b)}
                idx += 1


def _selected(key: str, rate: float, seed: int) -> bool:
    if rate >= 1:
        return True
    return int(hashlib.sha256(f"ext:{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < rate


def process_one(raw: bytes, item: Dict[str, Any], image_id: str, detector: Callable, params: Dict[str, Any],
                out_dir, min_crop_side: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: None for k in FACE_FIELDS}
    row.update({"image_id": image_id, "group_id": image_id, "method": item["method"], "label": item["label"],
                "split": "test", "zip": "", "member": item["origin"], "decode_ok": 0, "face_found": 0,
                "kept": 0, "gray_matched": 0, "src_bytes": len(raw)})
    try:
        img0 = Image.open(io.BytesIO(raw))
        mode, qlum = img0.mode, q_lum_mean(img0)
        img = ImageOps.exif_transpose(img0).convert("RGB")
    except Exception:
        return row
    W, H = img.size
    row.update({"src_width": W, "src_height": H, "src_mode": mode, "decode_ok": 1, "q_lum_mean": qlum,
                "bytes_per_px": len(raw) / (W * H)})
    face = detector(np.asarray(img))
    if face is None:
        return row
    x, y, w, h, score = face
    box = square_crop_box(face, W, H, params["crop_margin"])
    side = box[2] - box[0]
    row.update({"face_found": 1, "face_x": x, "face_y": y, "face_w": w, "face_h": h, "face_score": score,
                "crop_l": box[0], "crop_t": box[1], "crop_r": box[2], "crop_b": box[3],
                "native_crop_side": side, "face_area_ratio": (w * h) / (W * H)})
    if side < min_crop_side:
        return row
    S = params["output_size"]
    crop = img.crop(box).resize((S, S), Image.Resampling.BICUBIC)
    rel = f"{image_id}.jpg"
    path = Path(out_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path, "JPEG", quality=params["jpeg_quality"])
    arr = np.asarray(crop)
    row.update({"kept": 1, "out_path": rel, "mean_chroma": mean_chroma(arr), "sharpness": sharpness(arr)})
    return row


def run_external(items: Iterable[Dict[str, Any]], detector: Callable, params: Dict[str, Any], out_dir, manifest_csv,
                 sample_rate: float = 1.0, seed: int = 42, limit: Optional[int] = None,
                 min_crop_side: int = 96, progress_every: int = 500) -> int:
    """Resumable. Returns the number of images processed in this call."""
    if not 0 < sample_rate <= 1:
        raise ValueError(f"Invalid configuration: sample_rate must be in (0, 1], got {sample_rate}")
    manifest_csv = Path(manifest_csv)
    done = set()
    if manifest_csv.is_file():
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            done = {r["image_id"] for r in csv.DictReader(f)}
    new_file = not manifest_csv.is_file()
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with open(manifest_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FACE_FIELDS)
        if new_file:
            w.writeheader()
        for item in items:
            if not _selected(item["key"], sample_rate, seed):
                continue
            image_id = f"{item['method']}/{hashlib.sha256(item['key'].encode()).hexdigest()[:16]}"
            if image_id in done:
                continue
            try:
                raw = item["read"]()
            except Exception:
                raw = b""                       # recorded as a decode failure, never silently skipped
            row = process_one(raw, item, image_id, detector, params, out_dir, min_crop_side)
            w.writerow({k: "" if row.get(k) is None else row[k] for k in FACE_FIELDS})
            f.flush()
            processed += 1
            if progress_every and processed % progress_every == 0:
                print(f"  processed {processed} images")
            if limit and processed >= limit:
                break
    return processed


def summarize_external(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "decoded": 0, "face_found": 0, "kept": 0})
    side: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        d = by[r["method"]]
        d["n"] += 1
        d["decoded"] += r["decode_ok"]
        d["face_found"] += r["face_found"]
        d["kept"] += r["kept"]
        if r["kept"]:
            side["real" if r["label"] == 0 else "manipulated"].append(r["native_crop_side"])
    return {
        "by_method": dict(by),
        "kept_by_class": {"real": len(side["real"]), "manipulated": len(side["manipulated"])},
        "median_native_crop_side": {k: statistics.median(v) for k, v in side.items() if v},
        "shortcut_audit": shortcut_audit(rows),
    }
