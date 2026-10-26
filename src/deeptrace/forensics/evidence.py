"""Structured evidence: one JSON document per analysed image or video (the Part 2 -> Part 3 contract).

The detector is the source of truth; this document only reports what it and the forensic checks measured.
Schema: docs/part2_evidence_schema.json, described in docs/part2_evidence_schema.md. Images and videos share
one shape: fields that do not apply are null or [] (for example temporal_stability is null for images).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from deeptrace.forensics import DEFAULT_CONFIG
from deeptrace.forensics.attribution import Explanation, explain_image
from deeptrace.forensics.localization import localize
from deeptrace.forensics.robustness import NOTE as ROBUSTNESS_NOTE, run_robustness
from deeptrace.forensics.stability import NOTE as ESS_NOTE, stability
from deeptrace.forensics.video import analyze_video
from deeptrace.models.detector import MODEL_NAME

SCHEMA_VERSION = "1.0"
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "docs" / "part2_evidence_schema.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
SCORE_SEMANTICS = ("fake_probability is the detector's softmax probability for the MANIPULATED class (for videos: "
                   "its mean over frames with a usable face). confidence is the probability of the predicted class. "
                   "Neither is calibrated, neither is forensic certainty, and neither is the probability that the "
                   "media is actually manipulated. Suggested wording: 'The detector assigned a fake-class "
                   "probability of 0.91.'")
PART2_LIMITATIONS = [
    "Suspicious regions are model-indicated regions from Grad-CAM on a 12x12 feature grid (about 32 px per cell of "
    "the 380 px face crop); they are not ground-truth manipulated pixels. Localization accuracy is not yet measured "
    "(no manipulation masks are available).",
    ROBUSTNESS_NOTE,
    ESS_NOTE,
]
SMALL_FACE_LIMITATION = ("The analysed face crop is small ({side} px native, under {limit} px); the detector was less "
                         "reliable on small faces in Part 1 testing.")
NO_FACE_LIMITATION = "No face was detected, so the detector made no prediction and no forensic checks were run."


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def media_type(path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported file type {ext!r}: expected an image ({sorted(IMAGE_EXTENSIONS)}) or a video "
                     f"({sorted(VIDEO_EXTENSIONS)})")


def _base(path, kind: str, model: Dict[str, Any], cfg: Dict[str, Any], config_path) -> Dict[str, Any]:
    p = Path(path)
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG
    return {
        "schema_version": SCHEMA_VERSION, "media_type": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"file_name": p.name, "sha256": file_sha256(p), "size_bytes": p.stat().st_size},
        "model": {"name": MODEL_NAME, **model},
        "score_semantics": SCORE_SEMANTICS,
        "settings": {"config_file": cfg_file.name, "config_sha256": file_sha256(cfg_file), "seed": cfg["run"]["seed"],
                     **{k: cfg[k] for k in ("attribution", "localization", "robustness", "stability", "evidence")},
                     **({"video": cfg["video"]} if kind == "video" else {})},
        "artifacts": {},
    }


def _ess(e: Optional[Dict[str, Any]], settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if e is None:
        return None
    return {**e, "weights": settings["weights"], "bands": settings["bands"], "mode": settings["mode"], "note": ESS_NOTE}


def _robustness_results(rep: Dict[str, Any]) -> List[Dict[str, Any]]:
    keep = ("mode", "name", "label", "prediction", "fake_probability", "delta_fake_probability", "prediction_flipped",
            "no_face", "crop_iou")
    return [{**{k: r[k] for k in keep}, "map_correlation": (r["localization"] or {}).get("map_correlation")}
            for r in rep["perturbations"]]


def _small_face(side: Optional[int], limit: int) -> Tuple[List[Dict[str, str]], List[str]]:
    if side is None or side >= limit:
        return [], []
    msg = SMALL_FACE_LIMITATION.format(side=side, limit=limit)
    return [{"code": "small_face_part2", "message": msg}], [msg]


def image_evidence(det, path, cfg: Dict[str, Any], config_path=None) -> Tuple[Dict[str, Any], Explanation, Dict]:
    """(evidence, baseline Explanation, localization) for one image. Robustness uses cfg['evidence'] modes."""
    ev_cfg, st_cfg, rob = cfg["evidence"], cfg["stability"], cfg["robustness"]
    e = explain_image(det, path, **cfg["attribution"])
    loc = localize(e, **cfg["localization"])
    r = e.result
    ev = _base(path, "image", r["model"], cfg, config_path)
    ev["input"].update({"width": r["input_size"][0], "height": r["input_size"][1]})
    warnings = list(r["warnings"])
    limitations = list(r["limitations"]) + PART2_LIMITATIONS
    face = {"found": r["face_found"], "face_box": r.get("face_box"), "face_score": r.get("face_score"),
            "crop_box": r.get("crop_box"), "native_crop_side": r.get("native_crop_side")}
    common = {"status": r["status"], "prediction": r["prediction"], "confidence": r["confidence"],
              "fake_probability": r["fake_probability"], "real_probability": r["real_probability"], "face": face,
              "aggregation": None, "frames": None, "temporal_stability": None, "suspicious_intervals": [],
              "isolated_suspicious_frames": [], "observations": []}
    if r["status"] != "OK":
        ev.update({**common, "localization": None, "suspicious_regions": [], "prediction_stability": None,
                   "confidence_stability": None, "localization_stability": None, "evidence_stability": None,
                   "robustness": None, "robustness_observations": [], "warnings": warnings,
                   "limitations": limitations + [NO_FACE_LIMITATION]})
        return ev, e, loc
    w, lim = _small_face(r["native_crop_side"], ev_cfg["small_face_min_side"])
    rep = run_robustness(det, path, cfg["attribution"], cfg["localization"], rob["perturbations"],
                         ev_cfg["robustness_modes"], cfg["run"]["seed"], rob["min_overlap_fraction"])
    st = stability(rep, st_cfg["mode"], st_cfg["weights"], st_cfg["bands"], st_cfg["margin_floor"], st_cfg["cap_band"])
    ev.update({**common,
               "localization": {k: loc[k] for k in ("evidence_strength", "message", "positive_fraction")}
                               | {"method": loc["attribution"]["method"], "target": "logit[MANIPULATED] - logit[REAL]"
                                  if loc["attribution"]["target_score"] == "logit_diff" else "logit[MANIPULATED]",
                                  "target_layer": loc["attribution"]["target_layer"]},
               "suspicious_regions": loc["regions"],
               "prediction_stability": st["prediction_stability"], "confidence_stability": st["confidence_stability"],
               "localization_stability": st["localization_stability"],
               "evidence_stability": _ess(st["evidence_stability"], st["settings"]),
               "robustness": {"modes": list(ev_cfg["robustness_modes"]), "runs": len(rep["perturbations"]),
                              "results": _robustness_results(rep), "note": ROBUSTNESS_NOTE},
               "robustness_observations": rep["observations"],
               "warnings": warnings + w, "limitations": limitations + lim})
    return ev, e, loc


def _median_component(keys: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    ok = [k for k in keys if k["stability"]["status"] == "OK"]
    if not ok:
        return None
    return {"value": round(float(np.median([k["stability"][name]["value"] for k in ok])), 4),
            "method": "median over key frames",
            "per_key_frame": [{"frame": k["frame"], "time_s": k["time_s"], **k["stability"][name]} for k in ok]}


def video_evidence(det, path, cfg: Dict[str, Any], config_path=None, on_frame=None) -> Dict[str, Any]:
    rep = analyze_video(det, path, cfg, on_frame=on_frame)
    ev = _base(path, "video", rep["model"], cfg, config_path)
    ev["input"].update({k: rep["video"][k] for k in ("width", "height", "fps", "frame_count", "duration_s")})
    face_frames = [f for f in rep["frames"] if f["native_crop_side"] is not None]
    by_frame = {f["frame"]: f for f in rep["frames"]}
    keys = rep["key_frames"]
    regions = [{"frame": k["frame"], "time_s": k["time_s"], **r} for k in keys for r in by_frame[k["frame"]]["regions"]]
    limitations = list(det.limitations) + PART2_LIMITATIONS + list(rep["limitations"])
    small = [f["native_crop_side"] for f in face_frames if f["native_crop_side"] < cfg["evidence"]["small_face_min_side"]]
    if small:
        limitations.append(f"{len(small)} of {len(face_frames)} analysed frame(s) had a small face crop (under "
                           f"{cfg['evidence']['small_face_min_side']} px native); the detector was less reliable on "
                           f"small faces in Part 1 testing.")
    if rep["status"] != "OK":
        limitations.append(NO_FACE_LIMITATION)
    ess = rep["evidence_stability"]
    ev.update({
        "status": rep["status"], "prediction": rep["prediction"], "confidence": rep["confidence"],
        "fake_probability": rep["fake_probability"],
        "real_probability": None if rep["fake_probability"] is None else round(1 - rep["fake_probability"], 6),
        "face": {"usable_face_frames": rep["usable_face_frames"], "usable_face_fraction": rep["usable_face_fraction"],
                 "sampled_frames": rep["sampling"]["sampled_frames"]},
        "aggregation": rep["aggregation"],
        "frames": [{k: f[k] for k in ("frame", "time_s", "status", "prediction", "fake_probability",
                                      "evidence_strength")} for f in rep["frames"]],
        "localization": None if not keys else {"method": cfg["attribution"]["method"],
                                                "note": "Regions are reported for the key frames only."},
        "suspicious_regions": regions,
        "prediction_stability": _median_component(keys, "prediction_stability"),
        "confidence_stability": _median_component(keys, "confidence_stability"),
        "localization_stability": _median_component(keys, "localization_stability"),
        "evidence_stability": None if ess is None else {**ess, "weights": cfg["stability"]["weights"],
                                                        "bands": cfg["stability"]["bands"],
                                                        "mode": cfg["stability"]["mode"], "note": ESS_NOTE},
        "temporal_stability": rep["temporal_stability"],
        "suspicious_intervals": rep["suspicious_intervals"],
        "isolated_suspicious_frames": rep["isolated_suspicious_frames"],
        "robustness": None if not keys else {"modes": list(cfg["video"]["robustness_modes"]),
                                             "key_frames": [k["frame"] for k in keys], "note": ROBUSTNESS_NOTE},
        "robustness_observations": [{"frame": k["frame"], "time_s": k["time_s"], **o}
                                    for k in keys for o in k["robustness_observations"]],
        "observations": rep["observations"], "warnings": rep["warnings"], "limitations": limitations,
    })
    return ev


def analyze_file(det, path, cfg: Dict[str, Any], config_path=None) -> Dict[str, Any]:
    """Evidence for an image or a video, chosen by file extension."""
    if media_type(path) == "image":
        return image_evidence(det, path, cfg, config_path)[0]
    return video_evidence(det, path, cfg, config_path)


# Minimal JSON Schema subset (type, enum, minimum, maximum, required, properties, items), so no extra dependency.
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(value, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[t])


def validate(instance, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Errors (empty if valid) for the subset of JSON Schema that docs/part2_evidence_schema.json uses."""
    errors: List[str] = []
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        if not any(_is(instance, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if _is(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    if isinstance(instance, dict):
        errors += [f"{path}: missing required key {k!r}" for k in schema.get("required", []) if k not in instance]
        for k, sub in schema.get("properties", {}).items():
            if k in instance:
                errors += validate(instance[k], sub, f"{path}.{k}")
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors
