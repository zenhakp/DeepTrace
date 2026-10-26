"""Prediction, confidence and localization stability, and the Evidence Stability Score (ESS).

Computed from a robustness report (deeptrace.forensics.robustness), so no model is needed. The ESS is a
project-defined summary of how well the evidence held up under the configured perturbations. It is NOT a
probability that the image is manipulated, and its weights and HIGH / MEDIUM / LOW bands are our own design
choices (configs/part2_forensics.yaml), not calibrated against anything.

  prediction_stability   = runs whose label equals the baseline / runs   (NO_FACE counts as not equal)
  confidence_stability   = mean over runs of max(0, 1 - |p - p0| / margin), margin = max(|p0 - 0.5|, floor)
                           (margin retention: the same change costs more near the 0.5 decision boundary)
  localization_stability = mean over runs of max(0, map correlation)       (no comparable map counts as 0)
  ESS = w_prediction * prediction + w_confidence * confidence + w_localization * localization
Band cap (project rule): if any run flipped the prediction or lost the face, the band is at most cap_band (MEDIUM by
default); the score itself is unchanged and band_capped / cap_reason record it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

COMPONENTS = ("prediction", "confidence", "localization")
BAND_ORDER = ("LOW", "MEDIUM", "HIGH")
NOTE = ("The Evidence Stability Score is a project-defined summary of how the evidence held up under a small set "
        "of controlled perturbations. It is not a probability that the image is manipulated; the weights and "
        "bands are design choices, not calibrated.")


def validate(weights: Dict[str, float], bands: Dict[str, float], cap_band: Optional[str] = None) -> None:
    if set(weights) != set(COMPONENTS):
        raise ValueError(f"ESS weights must be given for exactly {COMPONENTS}, got {sorted(weights)}")
    if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"ESS weights must be non-negative and sum to 1, got {weights}")
    if not 0 <= bands["medium"] <= bands["high"] <= 1:
        raise ValueError(f"ESS bands must satisfy 0 <= medium <= high <= 1, got {bands}")
    if cap_band is not None and cap_band not in BAND_ORDER:
        raise ValueError(f"cap_band must be one of {BAND_ORDER} or null, got {cap_band!r}")


def band(score: float, bands: Dict[str, float]) -> str:
    return "HIGH" if score >= bands["high"] else "MEDIUM" if score >= bands["medium"] else "LOW"


def components(rows: Sequence[Dict[str, Any]], baseline_p: float, margin_floor: float = 0.01) -> Optional[Dict]:
    """The three stability components over one mode's perturbation rows."""
    n = len(rows)
    if n == 0:
        return None
    margin = max(abs(baseline_p - 0.5), margin_floor)
    agree = sum(r["prediction_flipped"] is False for r in rows)
    kept = [0.0 if r["fake_probability"] is None else max(0.0, 1.0 - abs(r["fake_probability"] - baseline_p) / margin)
            for r in rows]
    corr = [(r["localization"] or {}).get("map_correlation") for r in rows]
    loc = [0.0 if c is None else max(0.0, c) for c in corr]
    deltas = [abs(r["fake_probability"] - baseline_p) for r in rows if r["fake_probability"] is not None]
    return {
        "prediction_stability": {"value": round(agree / n, 4), "agreeing_runs": agree, "runs": n,
                                 "flips": sum(bool(r["prediction_flipped"]) for r in rows),
                                 "no_face": sum(r["no_face"] for r in rows)},
        "confidence_stability": {"value": round(float(np.mean(kept)), 4), "method": "margin_retention",
                                 "baseline_fake_probability": baseline_p, "baseline_margin": round(margin, 4),
                                 "mean_abs_delta_fake_probability": round(float(np.mean(deltas)), 4) if deltas else None,
                                 "max_abs_delta_fake_probability": round(float(max(deltas)), 4) if deltas else None},
        "localization_stability": {"value": round(float(np.mean(loc)), 4), "method": "mean_clipped_map_correlation",
                                   "compared_runs": sum(c is not None for c in corr),
                                   "median_map_correlation": round(float(np.median([c for c in corr if c is not None])), 4)
                                   if any(c is not None for c in corr) else None},
    }


def evidence_stability(comp: Dict[str, Any], weights: Dict[str, float], bands: Dict[str, float],
                       cap_band: Optional[str] = "MEDIUM") -> Dict[str, Any]:
    """comp: a components() result (or a stability() result, which contains the same keys)."""
    values = {k: comp[f"{k}_stability"]["value"] for k in COMPONENTS}
    score = sum(weights[k] * values[k] for k in COMPONENTS)
    lost = {k: round(weights[k] * (1 - values[k]), 4) for k in COMPONENTS}
    driver = max(COMPONENTS, key=lambda k: lost[k]) if max(lost.values()) > 0 else None
    uncapped = band(score, bands)
    ps = comp["prediction_stability"]
    capped = (cap_band is not None and ps["flips"] + ps["no_face"] > 0
              and BAND_ORDER.index(uncapped) > BAND_ORDER.index(cap_band))
    label = cap_band if capped else uncapped
    reason = (f"{ps['flips']} prediction flip(s) and {ps['no_face']} lost face(s) in {ps['runs']} runs; any flip or "
              f"NO_FACE caps the band at {cap_band} (project rule). The score is unchanged." if capped else None)
    summary = (f"Evidence Stability Score {score:.2f} ({label}{f', capped from {uncapped}' if capped else ''}; "
               f"project-defined band, not a probability)."
               + (f" Largest cost: {driver} stability {values[driver]:.2f} (costs {lost[driver]:.2f} of 1.00 points)."
                  if driver else ""))
    return {"score": round(score, 4), "band": label, "band_uncapped": uncapped, "band_capped": capped,
            "cap_reason": reason, "components": values, "points_lost": lost, "driver": driver, "summary": summary}


def stability(report: Dict[str, Any], mode: str = "pipeline", weights: Optional[Dict[str, float]] = None,
              bands: Optional[Dict[str, float]] = None, margin_floor: float = 0.01,
              cap_band: Optional[str] = "MEDIUM") -> Dict[str, Any]:
    """Stability components and ESS for one robustness report. The ESS uses `mode`; other modes are diagnostics."""
    weights = weights or {"prediction": 0.4, "confidence": 0.3, "localization": 0.3}
    bands = bands or {"high": 0.8, "medium": 0.6}
    validate(weights, bands, cap_band)
    settings = {"mode": mode, "weights": weights, "bands": bands, "margin_floor": margin_floor, "cap_band": cap_band}
    if report["status"] != "OK":
        return {"status": "NOT_APPLICABLE", "reason": report.get("reason", "No prediction to test."),
                "prediction_stability": None, "confidence_stability": None, "localization_stability": None,
                "evidence_stability": None, "diagnostics": {}, "settings": settings, "note": NOTE}
    p0 = report["baseline"]["fake_probability"]
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in report["perturbations"]:
        by_mode.setdefault(row["mode"], []).append(row)
    if mode not in by_mode:
        raise ValueError(f"The robustness report has no {mode!r} runs (modes: {sorted(by_mode)})")
    main = components(by_mode[mode], p0, margin_floor)
    diagnostics = {}
    for other, rows in by_mode.items():
        if other != mode:
            c = components(rows, p0, margin_floor)
            diagnostics[other] = {**c, "evidence_stability": evidence_stability(c, weights, bands, cap_band)}
    return {"status": "OK", **main, "evidence_stability": evidence_stability(main, weights, bands, cap_band),
            "diagnostics": diagnostics, "settings": settings, "note": NOTE}


def weight_sensitivity(results: Sequence[Dict[str, Any]], alternatives: Dict[str, Dict[str, float]],
                       bands: Dict[str, float], cap_band: Optional[str] = "MEDIUM") -> List[Dict[str, Any]]:
    """Recompute the band of each stability() result under alternative weights (no model needed).

    results: dicts with "image" plus a stability() result. Returns one row per alternative weighting.
    """
    ok = [r for r in results if r["status"] == "OK"]
    rows = []
    for name, w in alternatives.items():
        validate(w, bands, cap_band)
        changed = []
        for r in ok:
            alt = evidence_stability(r, w, bands, cap_band)
            if alt["band"] != r["evidence_stability"]["band"]:
                changed.append({"image": r["image"], "band": r["evidence_stability"]["band"], "alt_band": alt["band"],
                                "score": r["evidence_stability"]["score"], "alt_score": alt["score"]})
        rows.append({"name": name, "weights": w, "images": len(ok), "changed": len(changed), "changes": changed})
    return rows
