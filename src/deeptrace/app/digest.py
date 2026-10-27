"""Evidence digest: the evidence JSON as short, ordered fact sentences, grouped into report sections.

Each section has two fact lists:
  facts      shown to the user and used as the deterministic (template) text. Part 2's own sentences (summary,
             message, note, cap_reason, observations, warnings, limitations) are quoted unchanged here.
  llm_facts  what the LLM sees. Same content, but phrased so it cannot be misread; empty = the section is not sent
             to the LLM and is always shown verbatim (limitations and warnings, schema rule 6).
Two Part 2 sentences are deliberately not given to the LLM, after it misread them on example_image (step 3 run):
  - localization.message puts the face crop's fake-class probability next to the word "regions"; the LLM wrote that
    the probability was "in the suspicious regions". The LLM gets regions by rank and location only.
  - evidence_stability.summary writes a component value and its cost in one phrase; the LLM called the value the
    cost. The LLM gets values and points lost as separate sentences.
Every fact comes from a field of the evidence document; nothing is added from outside it. Numbers are rounded to
2 decimals, and probabilities never round to look certain (0.9994 -> "above 0.99").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from deeptrace.app.evidence import Evidence

TITLES = {"prediction": "Detector prediction", "evidence": "Supporting evidence: model-indicated suspicious regions",
          "stability": "Evidence stability", "robustness": "Robustness observations",
          "temporal": "Over time (video)", "limitations": "Limitations and warnings"}
NO_FACE_FACT = "No face was found, so the detector made no prediction: there is no verdict for this {media}."
STRENGTH_FACT = {
    "model_indicated": "Evidence strength: model-indicated (the detector leaned MANIPULATED).",
    "weak": "Evidence strength: weak (the detector leaned REAL), so these regions are shown for transparency only.",
    "none": "Evidence strength: none (no part of the face crop pushed the detector towards MANIPULATED).",
}
NOT_PIXELS = ("The regions show where the detector's attribution map was strongest; they are not ground-truth "
              "manipulated pixels.")


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    facts: Tuple[str, ...]
    llm_facts: Tuple[str, ...]


@dataclass(frozen=True)
class Digest:
    media_type: str
    status: str
    score_semantics: str
    sections: Tuple[Section, ...]
    prediction: Optional[str] = None      # the detector's label, None for NO_FACE
    points_lost: Tuple[str, ...] = ()     # stability points lost per component, as written in the facts

    def section(self, key: str) -> Section:
        return next(s for s in self.sections if s.key == key)

    def llm_sections(self) -> List[Section]:
        return [s for s in self.sections if s.llm_facts]

    def all_facts(self) -> List[str]:
        return [f for s in self.sections for f in s.facts]


def _p(x: float) -> str:
    return f"{x:.2f}"


def _prob(x: float) -> str:
    """A detector probability. 0.9994 must not read as "1.00" (certainty), nor 0.003 as "0.00"."""
    return "above 0.99" if x > 0.99 and x < 1 else "below 0.01" if 0 < x < 0.01 else _p(x)


def _t(seconds: float) -> str:
    return f"{seconds:.1f} s"


def _where(bbox_crop, side: int) -> str:
    """Coarse position of a region's centre in the face crop, as seen in the image (left = viewer's left)."""
    cx, cy = (bbox_crop[0] + bbox_crop[2]) / 2 / side, (bbox_crop[1] + bbox_crop[3]) / 2 / side
    v = "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"
    h = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
    return "centre" if (v, h) == ("middle", "centre") else f"{v}-{h}"


def _regions(regions, side: int) -> List[str]:
    return [f"Region {r['rank']} (rank {r['rank']}): {_where(r['bbox_crop'], side)} of the face crop."
            for r in sorted(regions, key=lambda r: r["rank"])]


def _prediction(d: Dict[str, Any]) -> List[str]:
    m = d["model"]
    facts = [f"This is {'a video' if d['media_type'] == 'video' else 'an image'}, analysed by the {m['name']} "
             f"detector ({m['version']})."]
    if d["media_type"] == "video":
        f = d["face"]
        facts.append(f"{f['sampled_frames']} frames were sampled over {_t(d['input']['duration_s'])}; "
                     f"{f['usable_face_frames']} had a usable face.")
    facts.append(f"The detector's prediction is {d['prediction']}.")
    if d["media_type"] == "video":
        facts.append(d["aggregation"]["summary"])
        face = [f for f in d["frames"] if f["status"] == "OK"]
        facts.append(f"{sum(f['prediction'] == 'MANIPULATED' for f in face)} of the {len(face)} frames with a usable "
                     f"face were predicted MANIPULATED.")
    else:
        facts.append(f"The detector assigned a fake-class probability of {_prob(d['fake_probability'])}.")
    facts.append(f"Confidence (the probability of the predicted class) is {_prob(d['confidence'])}; it is "
                 f"uncalibrated and is not forensic certainty.")
    return facts


def _evidence(d: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    side, regions = d["model"]["image_size"], d["suspicious_regions"]
    if d["media_type"] == "image":
        loc = d["localization"]
        llm = [STRENGTH_FACT.get(loc["evidence_strength"], loc["message"]),
               f"Number of model-indicated suspicious regions: {len(regions)}."] + _regions(regions, side)
        return [loc["message"]] + llm[1:], llm + ([NOT_PIXELS] if regions else [])
    if d["localization"] is None:
        fact = ["No key frames were analysed, so no regions are reported."]
        return fact, fact
    by_frame = {f["frame"]: f for f in d["frames"]}
    keys = sorted({r["frame"] for r in regions} | set(d["robustness"]["key_frames"]))
    shown = [d["localization"]["note"], f"Number of model-indicated suspicious regions over all key frames: "
                                        f"{len(regions)}."]
    llm = list(shown)
    for k in keys:
        f = by_frame.get(k, {})
        strength = (f.get("evidence_strength") or "not reported").replace("_", "-")
        where = ", ".join(_where(r["bbox_crop"], side) for r in sorted(regions, key=lambda r: r["rank"])
                          if r["frame"] == k) or "none"
        p = f.get("fake_probability")
        head = f"Key frame at {_t(f.get('time_s', 0.0))}: prediction {f.get('prediction')}"
        shown.append(f"{head}, fake-class probability {'n/a' if p is None else _prob(p)}, evidence strength "
                     f"{strength}; region position(s) in the face crop: {where}.")
        llm.append(f"{head}, evidence strength {strength}; region position(s) in the face crop: {where}.")
    return shown, llm + ([NOT_PIXELS] if regions else [])


def _stability(d: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    es = d["evidence_stability"]
    if es is None:
        fact = ["No Evidence Stability Score is available for this result."]
        return fact, fact
    c, lost = es["components"], es["points_lost"]
    names = ("prediction", "confidence", "localization")
    clear = [f"Component values (0 to 1, higher is more stable): " + ", ".join(f"{n} {_p(c[n])}" for n in names) + ".",
             f"Points lost by each component (out of 1.00): " + ", ".join(f"{n} {_p(lost[n])}" for n in names) + ".",
             f"The {es['driver']} component lost the most points ({_p(lost[es['driver']])}); its value is "
             f"{_p(c[es['driver']])}."]
    extra = ([es["cap_reason"]] if es["band_capped"] else []) + ([es["bias_note"]] if es.get("bias_note") else [])
    shown = [es["summary"]] + clear + extra + [es["note"]]
    llm = [f"Evidence Stability Score: {_p(es['score'])}, band {es['band']}. It is a project-defined 0 to 1 score "
           f"with HIGH / MEDIUM / LOW bands, not a probability of manipulation."] + clear + extra
    return shown, llm


def _robustness(d: Dict[str, Any]) -> List[str]:
    if d["robustness"] is None:
        return ["No robustness runs were made for this result."]
    facts = []
    ps = d["prediction_stability"]
    if ps and "runs" in ps:
        facts.append(f"{ps['agreeing_runs']} of {ps['runs']} perturbed runs kept the prediction ({ps['flips']} "
                     f"flip(s), {ps['no_face']} run(s) where the face was lost).")
    for k in (ps or {}).get("per_key_frame", []):
        facts.append(f"Key frame at {_t(k['time_s'])}: {k['agreeing_runs']} of {k['runs']} perturbed runs kept the "
                     f"prediction.")
    cs = d["confidence_stability"]
    if cs and "mean_abs_delta_fake_probability" in cs:
        facts.append(f"Under the perturbations the fake-class probability moved by {_p(cs['mean_abs_delta_fake_probability'])} "
                     f"on average (largest change {_p(cs['max_abs_delta_fake_probability'])}).")
    obs = d["robustness_observations"]
    facts += [(f"Key frame at {_t(o['time_s'])}: " if "time_s" in o else "") + o["message"] for o in obs]
    if not obs:
        facts.append("No prediction flips or lost faces were observed in these perturbation runs.")
    facts.append(d["robustness"]["note"])
    return facts


def _temporal(d: Dict[str, Any]) -> List[str]:
    facts = []
    ts = d["temporal_stability"]
    if ts is None:
        facts.append("Temporal stability is not available (fewer than 2 frames had a usable face).")
    else:
        facts.append(f"Temporal stability is {_p(ts['value'])}: {ts['steady_pairs']} of {ts['pairs']} consecutive "
                     f"frame pairs kept the same label with only a small change in fake-class probability; "
                     f"{ts['label_flips']} label flip(s).")
    for i in d["suspicious_intervals"]:
        facts.append(f"Suspicious interval from {_t(i['start_s'])} to {_t(i['end_s'])}: {i['suspicious_frames']} "
                     f"suspicious sampled frame(s), mean fake-class probability {_prob(i['mean_fake_probability'])}, "
                     f"maximum {_prob(i['max_fake_probability'])}.")
    if not d["suspicious_intervals"]:
        facts.append("No suspicious interval (2 or more sampled frames in a row leaning MANIPULATED) was found.")
    for i in d["isolated_suspicious_frames"]:
        facts.append(f"Isolated suspicious frame(s) at {_t(i.get('start_s', 0.0))} (too short to form an interval).")
    facts += [o["message"] for o in d["observations"]]
    return facts


def build_digest(ev: Evidence) -> Digest:
    d = ev.data
    limitations = [f"Warning: {w['message']}" for w in d["warnings"]] + list(d["limitations"])
    if not ev.has_prediction:  # no verdict: nothing is sent to the LLM
        parts = [("prediction", [NO_FACE_FACT.format(media=d["media_type"])], [])]
    else:
        pred, rob = _prediction(d), _robustness(d)
        parts = [("prediction", pred, pred), ("evidence", *_evidence(d)), ("stability", *_stability(d)),
                 ("robustness", rob, rob)]
        if ev.is_video:
            temporal = _temporal(d)
            parts.append(("temporal", temporal, temporal))
    parts.append(("limitations", limitations, []))
    es = d["evidence_stability"]
    return Digest(d["media_type"], d["status"], d["score_semantics"],
                  tuple(Section(k, TITLES[k], tuple(f for f in facts if f), tuple(f for f in llm if f))
                        for k, facts, llm in parts),
                  d["prediction"], tuple(_p(v) for v in es["points_lost"].values()) if es else ())
