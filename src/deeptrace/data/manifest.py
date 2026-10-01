"""Manifest and leakage-safe split for paired datasets such as DeepFakeFace.

Measured structure (Step 4): the real zip holds one photo per name, and each fake zip holds
exactly one image per real photo with the same relative path. A 'group' is one real photo plus
every fake derived from it. A whole group goes to one split, so no source photo is shared
between train, validation and test.
"""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from .inspection import IMG_EXT

SPLITS = ("train", "val", "test")
REAL_METHOD = "real"
FIELDS = ["image_id", "group_id", "method", "label", "zip", "member", "split",
          "width", "height", "mode", "file_bytes", "header_ok"]


def group_id_of(member: str) -> str:
    """'insight/00/123_1980-01-01_2010.jpg' -> '00/123_1980-01-01_2010' (top folder and extension removed)."""
    parts = PurePosixPath(member).parts
    rel = PurePosixPath(*parts[1:]) if len(parts) > 1 else PurePosixPath(member)
    return str(rel.with_suffix(""))


def scan_zip(zip_path, method: str, label: int, progress_every: int = 10000) -> List[Dict[str, Any]]:
    """One record per image. header_ok only means the image header parsed, not that it fully decodes."""
    p = Path(zip_path)
    if not p.is_file():
        raise FileNotFoundError(f"Zip file not found: {p}")
    records: List[Dict[str, Any]] = []
    with zipfile.ZipFile(p) as zf:
        infos = sorted(
            (i for i in zf.infolist()
             if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in IMG_EXT
             and "__MACOSX" not in i.filename),
            key=lambda i: i.filename)
        for n, info in enumerate(infos, 1):
            width = height = None
            mode, ok = "", 1
            try:
                with zf.open(info) as fh:
                    img = Image.open(io.BytesIO(fh.read()))
                    width, height, mode = img.width, img.height, img.mode
            except Exception:
                ok = 0
            gid = group_id_of(info.filename)
            records.append({
                "image_id": f"{method}/{gid}", "group_id": gid, "method": method, "label": label,
                "zip": p.name, "member": info.filename, "split": "",
                "width": width, "height": height, "mode": mode,
                "file_bytes": info.file_size, "header_ok": ok,
            })
            if progress_every and n % progress_every == 0:
                print(f"  {p.name}: {n}/{len(infos)}")
    return records


def check_groups(records: Sequence[Dict[str, Any]], methods: Sequence[str]) -> int:
    """Every group must contain exactly one image per method. Fails loudly otherwise."""
    by_group: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        by_group[r["group_id"]][r["method"]] += 1
    expected = {m: 1 for m in methods}
    bad = sorted(g for g, c in by_group.items() if dict(c) != expected)
    if bad:
        raise ValueError(
            f"Invalid group structure: {len(bad)} of {len(by_group)} groups do not contain exactly one "
            f"image per method {sorted(methods)}. Examples: {bad[:5]}")
    return len(by_group)


def split_settings(cfg: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
    s = cfg.get("split")
    if not isinstance(s, dict):
        raise ValueError("Invalid configuration: missing 'split' section")
    seed, fr = s.get("seed"), s.get("fractions")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"Invalid configuration: split.seed must be an integer, got {seed!r}")
    if not isinstance(fr, dict) or set(fr) != set(SPLITS):
        raise ValueError("Invalid configuration: split.fractions must have exactly the keys train, val, test")
    vals = [fr[k] for k in SPLITS]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in vals) \
            or abs(sum(vals) - 1.0) > 1e-9:
        raise ValueError(f"Invalid configuration: split.fractions must be positive and sum to 1, got {fr}")
    return seed, {k: float(fr[k]) for k in SPLITS}


def assign_splits(group_ids: Sequence[str], fractions: Dict[str, float], seed: int) -> Dict[str, str]:
    """Deterministic, platform-independent: order groups by sha256(seed:group), then slice."""
    ids = sorted(set(group_ids),
                 key=lambda g: hashlib.sha256(f"{seed}:{g}".encode("utf-8")).hexdigest())
    n = len(ids)
    if n < 3:
        raise ValueError(f"Need at least 3 groups to split, got {n}")
    n_train = round(n * fractions["train"])
    n_val = round(n * fractions["val"])
    out: Dict[str, str] = {}
    for i, g in enumerate(ids):
        out[g] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    if len(set(out.values())) != 3:
        raise ValueError("Split produced an empty partition; check split.fractions")
    return out


def verify_no_leakage(records: Sequence[Dict[str, Any]]) -> None:
    seen: Dict[str, set] = defaultdict(set)
    for r in records:
        if r["split"] not in SPLITS:
            raise ValueError(f"Record without a valid split: {r['image_id']}")
        seen[r["group_id"]].add(r["split"])
    leaked = sorted(g for g, s in seen.items() if len(s) > 1)
    if leaked:
        raise ValueError(f"Leakage detected: {len(leaked)} groups appear in several splits, e.g. {leaked[:5]}")


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    imgs: Dict[str, Dict[str, int]] = {s: {} for s in SPLITS}
    groups: Dict[str, set] = {s: set() for s in SPLITS}
    for r in records:
        d = imgs[r["split"]]
        d[r["method"]] = d.get(r["method"], 0) + 1
        groups[r["split"]].add(r["group_id"])
    return {s: {"groups": len(groups[s]), "images": sum(imgs[s].values()), "by_method": imgs[s]} for s in SPLITS}


def shortcut_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Design check, NOT a detector result: how far does image size alone separate the classes?"""
    ok = [r for r in records if r["header_ok"]]

    def is512(r):
        return r["width"] == 512 and r["height"] == 512

    def profile(rs):
        n = len(rs)
        if not n:
            return {"n": 0}
        return {"n": n,
                "share_512x512": round(sum(is512(r) for r in rs) / n, 4),
                "share_non_square": round(sum(r["width"] != r["height"] for r in rs) / n, 4),
                "share_grayscale": round(sum(r["mode"] == "L" for r in rs) / n, 4)}

    out: Dict[str, Any] = {
        "header_failures": len(records) - len(ok),
        "rule": "predict MANIPULATED iff the image is exactly 512x512",
        "class_profile": {"real": profile([r for r in ok if r["label"] == 0]),
                          "manipulated": profile([r for r in ok if r["label"] == 1])},
        "rule_on_split": {},
    }
    for sp in SPLITS:
        rs = [r for r in ok if r["split"] == sp]
        tp = sum(r["label"] == 1 and is512(r) for r in rs)
        fp = sum(r["label"] == 0 and is512(r) for r in rs)
        fn = sum(r["label"] == 1 and not is512(r) for r in rs)
        tn = sum(r["label"] == 0 and not is512(r) for r in rs)
        n = len(rs)
        out["rule_on_split"][sp] = {
            "n": n,
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
    return out


def write_manifest(records: Sequence[Dict[str, Any]], path) -> str:
    """Write the CSV and return its sha256 so the exact split can be verified later."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") if r.get(k) is not None else "" for k in FIELDS})
    return hashlib.sha256(p.read_bytes()).hexdigest()
