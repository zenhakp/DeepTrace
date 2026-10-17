"""Within-class analysis: does the model's output depend on a nuisance feature inside one class?"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Sequence


def binned_rates(rows_by_id: Dict[str, Dict[str, Any]], preds: Sequence[Dict[str, Any]], feature: str,
                 label: int, n_bins: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
    missing = sum(1 for p in preds if p["image_id"] not in rows_by_id)
    if missing:
        raise ValueError(f"{missing} prediction ids were not found in the manifest (wrong dataset?)")
    items = sorted(((rows_by_id[p["image_id"]][feature], float(p["fake_prob"])) for p in preds
                    if int(p["label"]) == label and rows_by_id[p["image_id"]].get(feature) is not None),
                   key=lambda t: t[0])
    if len(items) < n_bins:
        raise ValueError(f"Too few images ({len(items)}) with feature '{feature}' for {n_bins} bins")
    size, out = len(items) / n_bins, []
    for b in range(n_bins):
        chunk = items[round(b * size): round((b + 1) * size)]
        vals, probs = [v for v, _ in chunk], [p for _, p in chunk]
        out.append({"bin": b + 1, "n": len(chunk), "feature_min": min(vals), "feature_max": max(vals),
                    "mean_fake_prob": round(statistics.mean(probs), 4),
                    "share_predicted_manipulated": round(sum(p >= threshold for p in probs) / len(chunk), 4)})
    return out
