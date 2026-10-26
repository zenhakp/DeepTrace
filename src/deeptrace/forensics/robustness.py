"""Controlled perturbations: how prediction, confidence and localization change.

Two modes (levels in configs/part2_forensics.yaml):
- pipeline: perturb the original image and rerun face detection, crop and detector. Realistic: the crop may move or
  the face may be lost. Maps are compared in original-image pixels over the overlap of the two crops.
- fixed_crop: perturb the baseline prepared crop and feed it straight to the model. Controlled: maps are compared
  pixel by pixel in crop space, which separates detector sensitivity from face-crop jitter.
A prediction flip or a NO_FACE is a finding, not an error. A handful of perturbations does not cover real-world
distortion.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from deeptrace.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from deeptrace.forensics.attribution import GradCAM, explain_image, to_original
from deeptrace.forensics.localization import evidence_strength, format_probability
from deeptrace.models.inference import DeepTraceDetector

MODES = ("pipeline", "fixed_crop")
COMPARE_MAX_SIDE = 512      # pipeline maps are compared on a copy of the image no larger than this (memory)
NOTE = ("A small set of controlled perturbations; it does not cover all real-world distortion, and results on a "
        "few images do not generalise.")


def jpeg(img: Image.Image, quality, rng=None) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def downscale(img: Image.Image, scale, rng=None) -> Image.Image:
    """Downscale and resize back, so detail is lost but pixel coordinates stay the same."""
    w, h = img.size
    small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BICUBIC)


def gaussian_blur(img: Image.Image, sigma, rng=None) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(sigma))       # Pillow's radius is the standard deviation


def brightness(img: Image.Image, factor, rng=None) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast(img: Image.Image, factor, rng=None) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def gaussian_noise(img: Image.Image, sigma, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) + rng.normal(0.0, sigma, (img.size[1], img.size[0], 3))
    return Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))


PERTURBATIONS = {   # config key -> (function, readable label)
    "jpeg_quality": (jpeg, "JPEG quality {}"),
    "downscale": (downscale, "downscale x{} and back"),
    "gaussian_blur_sigma": (gaussian_blur, "Gaussian blur sigma {}"),
    "brightness": (brightness, "brightness x{}"),
    "contrast": (contrast, "contrast x{}"),
    "gaussian_noise_sigma": (gaussian_noise, "Gaussian noise sigma {} (0-255 scale)"),
}


def perturbation_list(levels: Dict[str, Sequence]) -> List[Dict[str, Any]]:
    out = []
    for family, values in levels.items():
        if family not in PERTURBATIONS:
            raise ValueError(f"Unknown perturbation {family!r}; expected one of {sorted(PERTURBATIONS)}")
        for v in values:
            out.append({"family": family, "level": v, "name": f"{family}={v}",
                        "label": PERTURBATIONS[family][1].format(v)})
    return out


def apply(family: str, img: Image.Image, level, seed: int) -> Image.Image:
    """A fresh generator per call, so each result is reproducible regardless of order."""
    return PERTURBATIONS[family][0](img, level, np.random.default_rng(seed))


def crop_to_tensor(crop: Image.Image) -> torch.Tensor:
    """The same normalisation as DeepTraceDetector.prepare, for a crop that is already 380x380."""
    arr = (np.asarray(crop.convert("RGB"), dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _original_map(cam: np.ndarray, crop_box: Sequence[int], input_size: Sequence[int]) -> np.ndarray:
    w, h = input_size
    s = min(1.0, COMPARE_MAX_SIDE / max(w, h))
    box = [int(round(v * s)) for v in crop_box]
    return to_original(cam, box, (max(1, round(w * s)), max(1, round(h * s))))


def map_similarity(base: np.ndarray, other: np.ndarray, threshold: float = 0.5,
                   min_overlap: float = 0.5) -> Dict[str, Optional[float]]:
    """Pearson correlation and region-mask IoU over pixels both maps cover (NaN = not analysed)."""
    covered = ~np.isnan(base)
    valid = covered & ~np.isnan(other)
    overlap = float(valid.sum() / covered.sum()) if covered.any() else 0.0
    out: Dict[str, Optional[float]] = {"overlap_fraction": round(overlap, 4), "map_correlation": None,
                                       "region_iou": None}
    if overlap < min_overlap:
        return out
    a, b = base[valid], other[valid]
    if a.std() > 0 and b.std() > 0:
        out["map_correlation"] = round(float(np.corrcoef(a, b)[0, 1]), 4)
    ma, mb = a >= threshold, b >= threshold
    union = int((ma | mb).sum())
    if union:
        out["region_iou"] = round(float((ma & mb).sum() / union), 4)
    return out


def _summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: result.get(k) for k in ("status", "prediction", "fake_probability", "confidence", "crop_box",
                                       "native_crop_side")} | {
        "warnings": [w["code"] for w in result.get("warnings", [])]}


def _observation(p: Dict[str, Any], mode: str, row: Dict[str, Any], base: Dict[str, Any]) -> Optional[Dict]:
    where = f"{p['label']} ({mode})"
    if row["no_face"]:
        return {"perturbation": p["name"], "mode": mode, "kind": "no_face",
                "message": f"{where}: no face was detected after the perturbation, so no prediction was made."}
    if row["prediction_flipped"]:
        return {"perturbation": p["name"], "mode": mode, "kind": "prediction_flip",
                "message": f"{where}: the prediction changed from {base['prediction']} to {row['prediction']} "
                           f"(fake-class probability {format_probability(base['fake_probability'])} -> "
                           f"{format_probability(row['fake_probability'])})."}
    return None


def run_robustness(det, image, attribution: Dict[str, Any], localization: Dict[str, Any],
                   perturbations: Dict[str, Sequence], modes: Sequence[str] = MODES, seed: int = 42,
                   min_overlap_fraction: float = 0.5) -> Dict[str, Any]:
    """Baseline plus every perturbation in every mode. JSON-serialisable."""
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"Unknown robustness mode {mode!r}; expected one of {MODES}")
    plist = perturbation_list(perturbations)
    settings = {"modes": list(modes), "perturbations": {k: list(v) for k, v in perturbations.items()},
                "seed": seed, "min_overlap_fraction": min_overlap_fraction,
                "region_threshold": localization["region_threshold"], "compare_max_side": COMPARE_MAX_SIDE}
    img = DeepTraceDetector._load(image)          # same loading (EXIF, RGB, explicit errors) as the detector
    base = explain_image(det, img, **attribution)
    b = _summary(base.result)
    b["evidence_strength"] = evidence_strength(b["fake_probability"], base.attribution,
                                               localization["min_fake_probability"])[0]
    if base.result["status"] != "OK":
        return {"status": "NOT_APPLICABLE", "reason": "No face was detected in the original image, so there is "
                "no prediction to test.", "baseline": b, "perturbations": [], "observations": [],
                "settings": settings, "note": NOTE}
    thr = localization["region_threshold"]
    base_orig = _original_map(base.cam, b["crop_box"], base.result["input_size"])
    gradcam = GradCAM(det.detector.model, attribution["target_layer"], attribution["method"])
    rows, observations = [], []
    for mode in modes:
        for p in plist:
            if mode == "pipeline":
                e = explain_image(det, apply(p["family"], img, p["level"], seed), **attribution)
                res, attr = e.result, e.attribution
                sim = (map_similarity(base_orig, _original_map(e.cam, res["crop_box"], res["input_size"]), thr,
                                      min_overlap_fraction) if e.cam is not None else None)
                crop_iou = round(box_iou(b["crop_box"], res["crop_box"]), 4) if res.get("crop_box") else None
            else:
                x = crop_to_tensor(apply(p["family"], base.crop, p["level"], seed))
                res = {"status": "OK", **det.detector.predict(x)[0], "crop_box": b["crop_box"],
                       "native_crop_side": b["native_crop_side"], "warnings": []}
                out = gradcam(x, attribution["target_class"], attribution["target_score"])
                diff = max(abs(u - v) for u, v in zip(out["logits"], res["logits"]))
                if diff > attribution["consistency_atol"]:
                    raise RuntimeError(f"Attribution forward pass disagrees with the detector ({diff:.2e})")
                attr = {"empty_map": out["raw_max"] <= 0}
                sim, crop_iou = map_similarity(base.cam, out["cam"], thr, 0.0), None     # same crop by design
            fp = res.get("fake_probability")
            row = {"mode": mode, **p, "status": res["status"], "prediction": res.get("prediction"),
                   "fake_probability": fp, "confidence": res.get("confidence"),
                   "delta_fake_probability": None if fp is None else round(fp - b["fake_probability"], 6),
                   "no_face": res["status"] == "NO_FACE",
                   "prediction_flipped": None if fp is None else res["prediction"] != b["prediction"],
                   "evidence_strength": evidence_strength(fp, attr, localization["min_fake_probability"])[0],
                   "crop_iou": crop_iou, "native_crop_side": res.get("native_crop_side"),
                   "warnings": [w["code"] for w in res.get("warnings", [])], "localization": sim}
            rows.append(row)
            obs = _observation(p, mode, row, b)
            if obs:
                observations.append(obs)
    return {"status": "OK", "baseline": b, "perturbations": rows, "observations": observations,
            "settings": settings, "note": NOTE}


def summarise(reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (mode, perturbation) across images: counts and medians. Images whose baseline had no face are skipped."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for rep in reports:
        for row in rep["perturbations"]:
            groups.setdefault((row["mode"], row["name"]), []).append(row)

    def med(values):
        values = [v for v in values if v is not None]
        return round(float(np.median(values)), 4) if values else None

    table = []
    for (mode, name), rows in groups.items():
        loc = [r["localization"] or {} for r in rows]
        table.append({"mode": mode, "perturbation": name, "label": rows[0]["label"], "n": len(rows),
                      "flips": sum(bool(r["prediction_flipped"]) for r in rows),
                      "no_face": sum(r["no_face"] for r in rows),
                      "median_abs_delta_fake_probability": med([None if r["delta_fake_probability"] is None
                                                                else abs(r["delta_fake_probability"]) for r in rows]),
                      "max_abs_delta_fake_probability": max([abs(r["delta_fake_probability"]) for r in rows
                                                             if r["delta_fake_probability"] is not None], default=None),
                      "median_map_correlation": med([x.get("map_correlation") for x in loc]),
                      "median_region_iou": med([x.get("region_iou") for x in loc]),
                      "median_crop_iou": med([r["crop_iou"] for r in rows])})
    return table
