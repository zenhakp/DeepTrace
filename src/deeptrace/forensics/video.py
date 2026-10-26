"""Video analysis by reusing the image detector on sampled frames (no temporal model).

frames sampled at a fixed rate -> per frame: face crop + detector + Grad-CAM (deeptrace.forensics.attribution)
-> video-level aggregation (mean fake-class probability over frames with a usable face), temporal stability,
suspicious intervals, and robustness + ESS on a few key frames only (the full perturbation set per frame is too
slow on CPU). Frames without a face are findings, never silently dropped. Settings: configs/part2_forensics.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from deeptrace.forensics.attribution import explain_image
from deeptrace.forensics.localization import format_probability, localize
from deeptrace.forensics.robustness import box_iou, run_robustness
from deeptrace.forensics.stability import BAND_ORDER, COMPONENTS, band, stability

VIDEO_LIMITATIONS = [
    "Frames are analysed independently by the image detector; there is no temporal model.",
    "Only the largest detected face in each frame is analysed; in videos with several people the analysed face can "
    "change between frames (reported as face jumps).",
    "Only sampled frames are analysed, so short events between sampled frames can be missed.",
    "Robustness and the Evidence Stability Score are measured on a few key frames, chosen as the most suspicious "
    "frames plus the median frame; this deliberately biases the video score downward (conservative).",
    "Video compression, low source resolution and blur can shift frame predictions: on still photos, mild blur and "
    "recompression moved REAL predictions towards MANIPULATED. One real phone clip stayed REAL and one AI-generated "
    "clip was MANIPULATED, but that is one clip of each with different source quality, so compression or source "
    "quality as a cause is not ruled out. Video detection accuracy is not yet measured.",
]


def _open(path):
    import cv2
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Video file not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to read video: {p}")
    return cap


def video_info(path) -> Dict[str, Any]:
    import cv2
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info = {"fps": fps, "frame_count": count, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_s": round(count / fps, 3) if fps > 0 else None}
    finally:
        cap.release()
    if fps <= 0:
        raise RuntimeError(f"Unable to read the frame rate of video: {path}")
    return info


def iter_frames(path, sample_fps: float, max_frames: int,
                only: Optional[Sequence[int]] = None) -> Iterator[Tuple[int, float, Image.Image]]:
    """Sequential decoding (seeking is unreliable for some codecs). Yields (frame index, time in s, RGB frame)
    for frames on the sampling grid, or only the given frame indices."""
    fps = video_info(path)["fps"]
    step = max(1.0, fps / sample_fps)
    wanted = set(only) if only is not None else None
    cap = _open(path)
    try:
        idx, k, yielded = 0, 0, 0
        while yielded < (len(wanted) if wanted is not None else max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            take = idx in wanted if wanted is not None else idx == round(k * step)
            if take:
                k += 1
                yielded += 1
                yield idx, round(idx / fps, 3), Image.fromarray(frame[:, :, ::-1].copy())
            idx += 1
    finally:
        cap.release()


def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """[start, end] positions (inclusive) of consecutive True values."""
    runs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def suspicious_intervals(frames: Sequence[Dict[str, Any]], threshold: float = 0.5, min_frames: int = 2,
                         merge_gap: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Runs of sampled frames with fake-class probability >= threshold. Gaps of up to merge_gap sampled frames
    (below threshold or no face) are bridged. Runs shorter than min_frames are returned as isolated frames."""
    hot = [f["fake_probability"] is not None and f["fake_probability"] >= threshold for f in frames]
    merged: List[List[int]] = []
    for s, e in _runs(hot):
        if merged and s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    intervals, isolated = [], []
    for s, e in merged:
        inside = [f for f, h in zip(frames[s:e + 1], hot[s:e + 1]) if h]
        ps = [f["fake_probability"] for f in inside]
        peak = max(inside, key=lambda f: f["fake_probability"])
        item = {"start_s": frames[s]["time_s"], "end_s": frames[e]["time_s"], "start_frame": frames[s]["frame"],
                "end_frame": frames[e]["frame"], "sampled_frames": e - s + 1, "suspicious_frames": len(inside),
                "mean_fake_probability": round(float(np.mean(ps)), 4), "max_fake_probability": round(max(ps), 4),
                "peak_time_s": peak["time_s"]}
        (intervals if len(inside) >= min_frames else isolated).append(item)
    return intervals, isolated


def temporal_stability(frames: Sequence[Dict[str, Any]], video_label: Optional[str], max_jump: float = 0.2,
                       face_jump_iou: float = 0.3) -> Optional[Dict[str, Any]]:
    """Fraction of consecutive face-frame pairs that are steady (same label, |change| < max_jump)."""
    face = [f for f in frames if f["fake_probability"] is not None]
    if len(face) < 2:
        return None
    pairs = list(zip(face[:-1], face[1:]))
    steady = sum(a["prediction"] == b["prediction"] and abs(a["fake_probability"] - b["fake_probability"]) < max_jump
                 for a, b in pairs)
    jumps = [{"from_frame": a["frame"], "to_frame": b["frame"], "time_s": b["time_s"],
              "crop_iou": round(box_iou(a["crop_box"], b["crop_box"]), 4)}
             for a, b in pairs if box_iou(a["crop_box"], b["crop_box"]) < face_jump_iou]
    ps = [f["fake_probability"] for f in face]
    return {"value": round(steady / len(pairs), 4), "method": "fraction_of_steady_consecutive_pairs",
            "pairs": len(pairs), "steady_pairs": steady,
            "label_flips": sum(a["prediction"] != b["prediction"] for a, b in pairs),
            "label_agreement": round(sum(f["prediction"] == video_label for f in face) / len(face), 4),
            "fake_probability_std": round(float(np.std(ps)), 4),
            "mean_abs_change": round(float(np.mean([abs(a["fake_probability"] - b["fake_probability"])
                                                    for a, b in pairs])), 4),
            "face_jumps": jumps, "max_jump": max_jump, "face_jump_iou": face_jump_iou}


def key_frames(frames: Sequence[Dict[str, Any]], top_suspicious: int = 2, include_median: bool = True) -> List[int]:
    """Positions (in frames) of the most suspicious face frames plus the median one, in time order."""
    face = [i for i, f in enumerate(frames) if f["fake_probability"] is not None]
    ranked = sorted(face, key=lambda i: frames[i]["fake_probability"], reverse=True)
    chosen = ranked[:top_suspicious]
    if include_median and ranked:
        median = ranked[len(ranked) // 2]
        if median not in chosen:
            chosen.append(median)
    return sorted(chosen)


def _video_ess(per_frame: List[Dict[str, Any]], weights, bands, cap_band) -> Optional[Dict[str, Any]]:
    ok = [k for k in per_frame if k["stability"]["status"] == "OK"]
    if not ok:
        return None
    score = float(np.median([k["stability"]["evidence_stability"]["score"] for k in ok]))
    uncapped = band(score, bands)
    lost = sum(k["stability"]["prediction_stability"]["flips"] + k["stability"]["prediction_stability"]["no_face"]
               for k in ok)
    capped = cap_band is not None and lost > 0 and BAND_ORDER.index(uncapped) > BAND_ORDER.index(cap_band)
    label = cap_band if capped else uncapped
    return {"score": round(score, 4), "band": label, "band_uncapped": uncapped, "band_capped": capped,
            "cap_reason": (f"{lost} prediction flip(s) or lost face(s) across the key frames; any flip or NO_FACE caps "
                           f"the band at {cap_band} (project rule)." if capped else None),
            "method": "median of key-frame Evidence Stability Scores",
            "components": {c: round(float(np.median([k["stability"][f"{c}_stability"]["value"] for k in ok])), 4)
                           for c in COMPONENTS},
            "key_frames": [{"frame": k["frame"], "time_s": k["time_s"],
                            "score": k["stability"]["evidence_stability"]["score"],
                            "band": k["stability"]["evidence_stability"]["band"],
                            "driver": k["stability"]["evidence_stability"]["driver"]} for k in ok],
            "bias_note": "Key frames are the most suspicious frames plus the median frame, which biases this score "
                         "downward (conservative).",
            "summary": (f"Video Evidence Stability Score {score:.2f} ({label}"
                        f"{f', capped from {uncapped}' if capped else ''}; median of {len(ok)} key frames; "
                        f"project-defined, not a probability).")}


def analyze_video(det, path, cfg: Dict[str, Any], on_frame=None) -> Dict[str, Any]:
    """cfg: the whole Part 2 config. Returns a JSON-serialisable report; key-frame Explanations are not included."""
    v, attr, loc_cfg = cfg["video"], cfg["attribution"], cfg["localization"]
    st_cfg, rob = cfg["stability"], cfg["robustness"]
    info = video_info(path)
    frames: List[Dict[str, Any]] = []
    model = None
    for idx, t, img in iter_frames(path, v["sample_fps"], v["max_frames"]):
        e = explain_image(det, img, **attr)
        loc = localize(e, **loc_cfg)
        r = e.result
        model = model or r["model"]
        frames.append({"frame": idx, "time_s": t, "status": r["status"], "prediction": r["prediction"],
                       "fake_probability": r["fake_probability"], "confidence": r["confidence"],
                       "crop_box": r.get("crop_box"), "native_crop_side": r.get("native_crop_side"),
                       "warnings": [w["code"] for w in r["warnings"]], "evidence_strength": loc["evidence_strength"],
                       "top_region": loc["regions"][0]["bbox_original"] if loc["regions"] else None})
        if on_frame:
            on_frame(frames[-1])
    if not frames:
        raise RuntimeError(f"Unable to read any frame from video: {path}")
    face = [f for f in frames if f["fake_probability"] is not None]
    usable = len(face) / len(frames)
    observations = [{"kind": "no_face_frames", "start_s": frames[s]["time_s"], "end_s": frames[e]["time_s"],
                     "frames": e - s + 1,
                     "message": f"No face was detected in {e - s + 1} sampled frame(s) between {frames[s]['time_s']} s "
                                f"and {frames[e]['time_s']} s; these frames have no prediction."}
                    for s, e in _runs([f["fake_probability"] is None for f in frames])]
    warnings = []
    if usable < v["low_face_coverage"]:
        warnings.append({"code": "low_face_coverage",
                         "message": f"Only {usable:.0%} of sampled frames had a usable face; the video-level result "
                                    f"rests on {len(face)} frame(s)."})
    min_side = cfg["evidence"]["small_face_min_side"]
    small = sum(f["native_crop_side"] < min_side for f in face)
    if small:
        warnings.append({"code": "small_face",
                         "message": f"{small} of {len(face)} face frame(s) had a face crop under {min_side} px native; "
                                    f"accuracy was lower for small faces in Part 1 testing."})
    report: Dict[str, Any] = {"status": "OK" if face else "NO_FACE", "video": info, "model": model,
                              "sampling": {"sample_fps": v["sample_fps"], "max_frames": v["max_frames"],
                                           "sampled_frames": len(frames)},
                              "usable_face_frames": len(face), "usable_face_fraction": round(usable, 4),
                              "frames": frames, "observations": observations, "warnings": warnings,
                              "limitations": list(VIDEO_LIMITATIONS), "settings": {"video": v}}
    if not face:
        report.update({"prediction": None, "fake_probability": None, "confidence": None, "aggregation": None,
                       "temporal_stability": None, "suspicious_intervals": [], "isolated_suspicious_frames": [],
                       "key_frames": [], "evidence_stability": None})
        return report
    ps = [f["fake_probability"] for f in face]
    mean_p = float(np.mean(ps))
    label = "MANIPULATED" if mean_p >= 0.5 else "REAL"
    intervals, isolated = suspicious_intervals(frames, v["interval_threshold"], v["min_interval_frames"],
                                               v["merge_gap_frames"])
    per_key = []
    for pos in key_frames(frames, v["key_frames"]["top_suspicious"], v["key_frames"]["include_median"]):
        f = frames[pos]
        img = next(iter_frames(path, v["sample_fps"], 1, only=[f["frame"]]))[2]
        rep = run_robustness(det, img, attr, loc_cfg, rob["perturbations"], v["robustness_modes"],
                             cfg["run"]["seed"], rob["min_overlap_fraction"])
        st = stability(rep, st_cfg["mode"], st_cfg["weights"], st_cfg["bands"], st_cfg["margin_floor"],
                       st_cfg["cap_band"])
        per_key.append({"frame": f["frame"], "time_s": f["time_s"], "fake_probability": f["fake_probability"],
                        "robustness_observations": rep["observations"], "stability": st})
    report.update({
        "prediction": label, "fake_probability": round(mean_p, 6), "confidence": round(max(mean_p, 1 - mean_p), 6),
        "aggregation": {"method": "mean fake-class probability over frames with a usable face",
                        "mean": round(mean_p, 4), "median": round(float(np.median(ps)), 4), "max": round(max(ps), 4),
                        "min": round(min(ps), 4),
                        "manipulated_frame_fraction": round(sum(p >= 0.5 for p in ps) / len(ps), 4),
                        "summary": f"Mean fake-class probability {format_probability(mean_p)} over {len(face)} "
                                   f"frame(s) with a usable face; not a probability that the video is manipulated."},
        "temporal_stability": temporal_stability(frames, label, v["max_jump"], v["face_jump_iou"]),
        "suspicious_intervals": intervals, "isolated_suspicious_frames": isolated, "key_frames": per_key,
        "evidence_stability": _video_ess(per_key, st_cfg["weights"], st_cfg["bands"], st_cfg["cap_band"]),
    })
    ts = report["temporal_stability"]
    if ts and ts["face_jumps"]:
        observations.append({"kind": "face_jump", "frames": len(ts["face_jumps"]),
                             "message": f"The analysed face crop jumped between {len(ts['face_jumps'])} pair(s) of "
                                        f"consecutive frames (crop overlap below {v['face_jump_iou']}); a different "
                                        f"face may have been analysed."})
    return report
