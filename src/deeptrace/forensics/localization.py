"""Turn an attribution map into model-indicated suspicious regions, labelled by how much they mean.

Every map is normalised to a maximum of 1, so a bright area exists even when the detector leaned REAL. The
evidence_strength label (rule and threshold in configs/part2_forensics.yaml, our own design choice) says
whether the regions support a MANIPULATED verdict or are shown for transparency only. Regions are not
ground-truth manipulated pixels; localization accuracy is not measured (no manipulation masks).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from deeptrace.forensics.attribution import Explanation

STRENGTHS = ("model_indicated", "weak", "none", "not_applicable")


def format_probability(p: float) -> str:
    """Two decimals, without rounding a small probability to 0.00 or a large one to 1.00."""
    if p < 0.01:
        return "below 0.01"
    if p > 0.99:
        return "above 0.99"
    return f"{p:.2f}"


def crop_to_original(box: Sequence[float], crop_box: Sequence[int], map_size: int) -> List[float]:
    """[l, t, r, b] in prepared-crop pixels -> original-image pixels (the crop is square)."""
    scale = (crop_box[2] - crop_box[0]) / map_size
    return [round(crop_box[0] + box[0] * scale, 1), round(crop_box[1] + box[1] * scale, 1),
            round(crop_box[0] + box[2] * scale, 1), round(crop_box[1] + box[3] * scale, 1)]


def extract_regions(cam: np.ndarray, crop_box: Sequence[int], threshold: float = 0.5,
                    min_area: float = 0.01, max_regions: int = 3) -> List[Dict[str, Any]]:
    """Connected areas where the map is >= threshold (fraction of its max), largest share of the map first."""
    total = float(cam.sum())
    if total <= 0:
        return []
    mask = (cam >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    size = cam.shape[0]
    regions = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        if area / cam.size < min_area:
            continue
        inside = labels == k
        box = [x, y, x + w, y + h]
        regions.append({"bbox_crop": box, "bbox_original": crop_to_original(box, crop_box, size),
                        "area_fraction": round(area / cam.size, 4),
                        "peak": round(float(cam[inside].max()), 4),
                        "share_of_map": round(float(cam[inside].sum()) / total, 4)})
    regions.sort(key=lambda r: r["share_of_map"], reverse=True)
    for i, r in enumerate(regions[:max_regions], start=1):
        r["rank"] = i
    return regions[:max_regions]


def evidence_strength(fake_probability: Optional[float], attribution: Optional[Dict[str, Any]],
                      min_fake_probability: float = 0.5) -> Tuple[str, str]:
    if attribution is None or fake_probability is None:
        return "not_applicable", "No face was analysed, so there is no prediction and no localization."
    if attribution["empty_map"]:
        return "none", ("No part of the face crop pushed the decision towards MANIPULATED; no regions are "
                        "indicated.")
    if fake_probability < min_fake_probability:
        return "weak", (f"Weak MANIPULATED evidence: the detector assigned a fake-class probability of "
                        f"{format_probability(fake_probability)}. The highlighted areas pushed towards MANIPULATED but were "
                        f"outweighed by the rest of the face; regions shown for transparency only.")
    return "model_indicated", (f"Model-indicated suspicious regions (the detector assigned a fake-class "
                               f"probability of {format_probability(fake_probability)}); not ground-truth manipulated pixels.")


def localize(explanation: Explanation, region_threshold: float = 0.5, min_region_area: float = 0.01,
             max_regions: int = 3, min_fake_probability: float = 0.5) -> Dict[str, Any]:
    """JSON-serialisable localization summary for one image."""
    result, a = explanation.result, explanation.attribution
    strength, message = evidence_strength(result.get("fake_probability"), a, min_fake_probability)
    if a is None:
        return {"evidence_strength": strength, "message": message, "regions": [], "positive_fraction": None,
                "attribution": None}
    regions = [] if a["empty_map"] else extract_regions(explanation.cam, a["crop_box"], region_threshold,
                                                        min_region_area, max_regions)
    evidence = a["positive_mass"] - a["negative_mass"]
    return {"evidence_strength": strength, "message": message, "regions": regions,
            "positive_fraction": round(a["positive_mass"] / evidence, 4) if evidence > 0 else None,
            "attribution": a,
            "rule": {"region_threshold": region_threshold, "min_region_area": min_region_area,
                     "max_regions": max_regions, "min_fake_probability": min_fake_probability}}
