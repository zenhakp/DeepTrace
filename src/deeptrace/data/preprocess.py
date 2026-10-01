"""Face-crop preprocessing for the DeepFakeFace manifest, plus a metadata shortcut audit.

Design (see docs/dataset_selection.md): raw images are separable by size alone, so every image is
turned into a face crop with identical geometry, colour-mode matching per group, the same resize
filter and the same JPEG re-encoding. A group is kept only if a face is found in all of its images.
"""
from __future__ import annotations

import csv
import hashlib
import io
import statistics
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

from .faces import is_gray_source, mean_chroma, q_lum_mean, sharpness, square_crop_box
from .manifest import SPLITS

FACE_FIELDS = [
    "image_id", "group_id", "method", "label", "split", "zip", "member",
    "src_width", "src_height", "src_mode", "src_bytes", "decode_ok",
    "face_found", "face_x", "face_y", "face_w", "face_h", "face_score",
    "crop_l", "crop_t", "crop_r", "crop_b", "native_crop_side", "face_area_ratio",
    "q_lum_mean", "bytes_per_px", "gray_matched", "mean_chroma", "sharpness", "kept", "out_path"]
STR_FIELDS = {"image_id", "group_id", "method", "split", "zip", "member", "src_mode", "out_path"}
INT_FIELDS = {"label", "src_width", "src_height", "src_bytes", "decode_ok", "face_found", "crop_l",
              "crop_t", "crop_r", "crop_b", "native_crop_side", "gray_matched", "kept"}
AUDIT_FEATURES = ["native_crop_side", "face_area_ratio", "q_lum_mean", "bytes_per_px", "mean_chroma", "sharpness"]


def params_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg.get("preprocess")
    if not isinstance(p, dict):
        raise ValueError("Invalid configuration: missing 'preprocess' section")

    def need(key, ok, msg):
        v = p.get(key)
        if isinstance(v, bool) or not ok(v):
            raise ValueError(f"Invalid configuration: preprocess.{key} {msg}, got {v!r}")
        return v

    num = lambda v: isinstance(v, (int, float))
    return {
        "face_model_url": need("face_model_url", lambda v: isinstance(v, str) and v.startswith("http"), "must be an http(s) URL"),
        "face_model_file": need("face_model_file", lambda v: isinstance(v, str) and v, "must be a file name"),
        "face_score_threshold": need("face_score_threshold", lambda v: num(v) and 0 < v < 1, "must be in (0, 1)"),
        "crop_margin": need("crop_margin", lambda v: num(v) and v >= 1.0, "must be >= 1.0"),
        "output_size": need("output_size", lambda v: isinstance(v, int) and v >= 64, "must be an integer >= 64"),
        "jpeg_quality": need("jpeg_quality", lambda v: isinstance(v, int) and 1 <= v <= 100, "must be an integer in 1..100"),
        "gray_chroma_eps": need("gray_chroma_eps", lambda v: num(v) and v >= 0, "must be >= 0"),
    }


def load_manifest(path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}. Run scripts/build_manifest.py first.")
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("label", "file_bytes", "header_ok"):
                r[k] = int(r[k])
            for k in ("width", "height"):
                r[k] = int(r[k]) if r[k] else None
            rows.append(r)
    return rows


def subset_groups(group_ids: Sequence[str], n: int, seed: int) -> List[str]:
    """Deterministic random subset. Uses a different hash salt than the split, so all splits appear."""
    return sorted(set(group_ids),
                  key=lambda g: hashlib.sha256(f"subset:{seed}:{g}".encode("utf-8")).hexdigest())[:n]


def process_group(records: Sequence[Dict[str, Any]], zips: Dict[str, zipfile.ZipFile],
                  detector: Callable, params: Dict[str, Any], out_dir) -> List[Dict[str, Any]]:
    S, quality = params["output_size"], params["jpeg_quality"]
    items = []
    for r in sorted(records, key=lambda r: (r["method"] != "real", r["method"])):   # real photo first
        it = {"rec": r, "img": None, "rgb": None, "mode": "", "qlum": None, "face": None, "ok": 0}
        try:
            with zips[r["zip"]].open(r["member"]) as fh:
                raw = fh.read()
            img = Image.open(io.BytesIO(raw))
            it["mode"], it["qlum"] = img.mode, q_lum_mean(img)
            img = ImageOps.exif_transpose(img).convert("RGB")
            it["img"], it["rgb"], it["ok"] = img, np.asarray(img), 1
        except Exception:
            it["ok"] = 0
        if it["ok"]:
            it["face"] = detector(it["rgb"])          # detector errors are bugs: let them propagate
        items.append(it)

    first = items[0]
    gray_group = bool(first["rec"]["method"] == "real" and first["ok"]
                      and is_gray_source(first["mode"], first["rgb"], params["gray_chroma_eps"]))
    kept = all(it["ok"] and it["face"] is not None for it in items)

    rows = []
    for it in items:
        r = it["rec"]
        row: Dict[str, Any] = {
            "image_id": r["image_id"], "group_id": r["group_id"], "method": r["method"], "label": r["label"],
            "split": r["split"], "zip": r["zip"], "member": r["member"],
            "src_width": r["width"], "src_height": r["height"], "src_mode": it["mode"],
            "src_bytes": r["file_bytes"], "decode_ok": it["ok"], "face_found": int(it["face"] is not None),
            "q_lum_mean": it["qlum"], "kept": int(kept), "gray_matched": int(gray_group),
            "bytes_per_px": (r["file_bytes"] / (r["width"] * r["height"])) if r["width"] and r["height"] else None,
        }
        if it["face"] is not None:
            x, y, w, h, score = it["face"]
            W, H = it["img"].size
            box = square_crop_box(it["face"], W, H, params["crop_margin"])
            row.update({"face_x": x, "face_y": y, "face_w": w, "face_h": h, "face_score": score,
                        "crop_l": box[0], "crop_t": box[1], "crop_r": box[2], "crop_b": box[3],
                        "native_crop_side": box[2] - box[0], "face_area_ratio": (w * h) / (W * H)})
            if kept:
                crop = it["img"].crop(box)
                if gray_group:
                    crop = crop.convert("L").convert("RGB")
                crop = crop.resize((S, S), Image.Resampling.BICUBIC)
                rel = f"{r['method']}/{r['group_id']}.jpg"
                path = Path(out_dir) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(path, "JPEG", quality=quality)
                arr = np.asarray(crop)
                row.update({"mean_chroma": mean_chroma(arr), "sharpness": sharpness(arr), "out_path": rel})
        rows.append(row)
    return rows


def run_preprocess(rows: Sequence[Dict[str, Any]], folder, out_dir, faces_csv, detector: Callable,
                   params: Dict[str, Any], limit_groups: Optional[int] = None, seed: int = 42,
                   progress_every: int = 500) -> int:
    """Process groups not yet in faces_csv (resumable). Returns the number of groups processed now."""
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)
    gids = subset_groups(list(by_group), limit_groups, seed) if limit_groups else sorted(by_group)

    faces_csv = Path(faces_csv)
    done = set()
    if faces_csv.is_file():
        with open(faces_csv, newline="", encoding="utf-8") as f:
            done = {row["group_id"] for row in csv.DictReader(f)}
    todo = [g for g in gids if g not in done]

    zips = {n: zipfile.ZipFile(Path(folder) / n) for n in {r["zip"] for g in todo for r in by_group[g]}}
    new_file = not faces_csv.is_file()
    faces_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(faces_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FACE_FIELDS)
            if new_file:
                w.writeheader()
            t0 = time.time()
            for n, g in enumerate(todo, 1):
                for row in process_group(by_group[g], zips, detector, params, out_dir):
                    w.writerow({k: "" if row.get(k) is None else row[k] for k in FACE_FIELDS})
                f.flush()
                if progress_every and n % progress_every == 0:
                    print(f"  {n}/{len(todo)} groups, {time.time() - t0:.0f}s elapsed")
    finally:
        for z in zips.values():
            z.close()
    return len(todo)


def read_faces_csv(path) -> List[Dict[str, Any]]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in FACE_FIELDS:
                v = r.get(k, "")
                if k in STR_FIELDS:
                    continue
                r[k] = None if v == "" else (int(float(v)) if k in INT_FIELDS else float(v))
            out.append(r)
    return out


# ---------- reporting and shortcut audit ----------
def auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """P(score of a manipulated image > score of a real one); ties count half. 0.5 = uninformative."""
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(ranks[i] for i, l in enumerate(labels) if l == 1)
    return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)


def best_threshold(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, int, float]:
    """Single-feature rule maximising balanced accuracy: (threshold, direction, balanced accuracy)."""
    n1 = sum(labels)
    n0 = len(labels) - n1
    pairs = sorted(zip(scores, labels))
    best = (pairs[0][0], 1, 0.0)
    c0 = c1 = 0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            c1 += pairs[j][1] == 1
            c0 += pairs[j][1] == 0
            j += 1
        ba = ((n1 - c1) / n1 + c0 / n0) / 2          # predict MANIPULATED if score > threshold
        for value, direction in ((ba, 1), (1 - ba, -1)):
            if value > best[2]:
                best = (pairs[i][0], direction, value)
        i = j
    return best


def eval_threshold(scores, labels, thr: float, direction: int) -> Optional[float]:
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    pred = [(s > thr) if direction == 1 else (s <= thr) for s in scores]
    tpr = sum(p and l == 1 for p, l in zip(pred, labels)) / n1
    tnr = sum((not p) and l == 0 for p, l in zip(pred, labels)) / n0
    return (tpr + tnr) / 2


def shortcut_audit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Can a single cheap metadata feature separate the classes? strength = max(AUC, 1-AUC); 0.5 is ideal."""
    kept = [r for r in rows if r["kept"] == 1]
    out: Dict[str, Any] = {}
    for feat in AUDIT_FEATURES:
        sel = [r for r in kept if r.get(feat) is not None]
        if not sel:
            out[feat] = {"n": 0}
            continue
        a = auc([r["label"] for r in sel], [r[feat] for r in sel])
        entry: Dict[str, Any] = {"n": len(sel), "auc_all_splits": None if a is None else round(a, 4),
                                 "strength": None if a is None else round(max(a, 1 - a), 4)}
        tr = [r for r in sel if r["split"] == "train"]
        te = [r for r in sel if r["split"] == "test"]
        if tr and te and 0 < sum(r["label"] for r in tr) < len(tr):
            thr, d, ba_tr = best_threshold([r[feat] for r in tr], [r["label"] for r in tr])
            ba_te = eval_threshold([r[feat] for r in te], [r["label"] for r in te], thr, d)
            entry.update({"balanced_acc_train": round(ba_tr, 4),
                          "balanced_acc_test_at_train_threshold": None if ba_te is None else round(ba_te, 4)})
        out[feat] = entry
    return out


def summarize_faces(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    decoded, found = defaultdict(int), defaultdict(int)
    groups = {s: set() for s in SPLITS}
    kept_groups = {s: set() for s in SPLITS}
    kept_images = defaultdict(int)
    side = defaultdict(list)
    for r in rows:
        if r["decode_ok"]:
            decoded[r["method"]] += 1
            found[r["method"]] += r["face_found"]
        groups[r["split"]].add(r["group_id"])
        if r["kept"]:
            kept_groups[r["split"]].add(r["group_id"])
            kept_images["real" if r["label"] == 0 else "manipulated"] += 1
            side["real" if r["label"] == 0 else "manipulated"].append(r["native_crop_side"])
    return {
        "face_detection_rate_by_method": {m: round(found[m] / decoded[m], 4) for m in decoded},
        "decode_failures": sum(1 for r in rows if not r["decode_ok"]),
        "groups_processed_by_split": {s: len(groups[s]) for s in SPLITS},
        "groups_kept_by_split": {s: len(kept_groups[s]) for s in SPLITS},
        "kept_images_by_class": dict(kept_images),
        "median_native_crop_side": {k: statistics.median(v) for k, v in side.items()},
    }
