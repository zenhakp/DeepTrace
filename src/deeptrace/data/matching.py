"""Matched evaluation subset: equalise the distribution of one nuisance feature across the classes.

Within each bin of the feature, both classes contribute the same number of images (chosen by a seeded
hash). After matching, the feature cannot separate the classes, so metrics on the subset are free of
that particular shortcut. The subset is for EVALUATION only; it is smaller than the full split.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

from .preprocess import shortcut_audit


def _key(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"match:{seed}:{image_id}".encode("utf-8")).hexdigest()


def matched_image_ids(rows: Sequence[Dict[str, Any]], feature: str, n_bins: int = 20, seed: int = 42) -> List[str]:
    sel = [r for r in rows if r.get("kept") == 1 and r.get(feature) is not None]
    real = [r for r in sel if r["label"] == 0]
    fake = [r for r in sel if r["label"] == 1]
    if not real or not fake:
        raise ValueError(f"Cannot match on {feature}: need images of both classes with this feature")
    lo = max(min(r[feature] for r in real), min(r[feature] for r in fake))
    hi = min(max(r[feature] for r in real), max(r[feature] for r in fake))
    if lo > hi:
        rr = (min(r[feature] for r in real), max(r[feature] for r in real))
        fr = (min(r[feature] for r in fake), max(r[feature] for r in fake))
        raise ValueError(f"Cannot match on {feature}: the classes have no overlapping range "
                         f"(real {rr}, manipulated {fr})")
    width = (hi - lo) / n_bins

    def bin_of(v: float) -> int:
        return 0 if width == 0 else min(int((v - lo) / width), n_bins - 1)

    bins: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    for r in sel:
        if lo <= r[feature] <= hi:
            bins.setdefault(bin_of(r[feature]), {0: [], 1: []})[r["label"]].append(r)
    chosen: List[str] = []
    for b in bins.values():
        n = min(len(b[0]), len(b[1]))
        for label in (0, 1):
            ordered = sorted(b[label], key=lambda r: _key(seed, r["image_id"]))
            chosen += [r["image_id"] for r in ordered[:n]]
    if not chosen:
        raise ValueError(f"Matching on {feature} produced an empty subset")
    return sorted(chosen)


def matching_report(rows: Sequence[Dict[str, Any]], ids: Sequence[str], feature: str) -> Dict[str, Any]:
    idset = set(ids)
    sub = [r for r in rows if r["image_id"] in idset]
    kept = [r for r in rows if r.get("kept") == 1]
    n_real_all = sum(r["label"] == 0 for r in kept)
    n_fake_all = sum(r["label"] == 1 for r in kept)
    n_real = sum(r["label"] == 0 for r in sub)
    n_fake = sum(r["label"] == 1 for r in sub)
    return {
        "feature": feature,
        "n_images": len(sub), "n_real": n_real, "n_manipulated": n_fake,
        "share_of_kept_real": round(n_real / n_real_all, 4) if n_real_all else None,
        "share_of_kept_manipulated": round(n_fake / n_fake_all, 4) if n_fake_all else None,
        "shortcut_audit_before": shortcut_audit(kept),
        "shortcut_audit_after": shortcut_audit(sub),
    }
