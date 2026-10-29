"""Markdown report for one analysed file: evidence facts, explanation, notes and any Q&A answers.

The report is written next to the evidence JSON and overlay PNGs (one job folder), so overlay images are linked by
file name. Every section lists its evidence facts verbatim; a language-model text is shown above them and marked as
such, so a reader can always check the explanation against the evidence. Limitations and warnings are always listed
in full (schema rule 6).
"""
from __future__ import annotations

from typing import Sequence

from deeptrace.app.digest import prob_text
from deeptrace.app.evidence import Evidence
from deeptrace.app.explain import Explanation
from deeptrace.app.qa import Answer

LLM_SOURCE = "Written by the local language model from the evidence below; it passed the evidence checks."
DISCLAIMER = ("This report explains the output of the DeepTrace detector, which is the source of truth for the "
              "prediction. Scores are detector outputs, not probabilities that the media is manipulated, and the "
              "report is not proof either way. Read it together with the limitations above.")


def _result(ev: Evidence) -> str:
    d = ev.data
    if not ev.has_prediction:
        return "**No verdict:** no face was found, so the detector made no prediction."
    mean = "mean " if ev.is_video else ""  # a video's score is the mean over frames with a usable face
    return f"**Detector prediction:** {d['prediction']} ({mean}fake-class probability {prob_text(d['fake_probability'])})"


def render_markdown(ev: Evidence, explanation: Explanation, answers: Sequence[Answer] = ()) -> str:
    d = ev.data
    m, i = d["model"], d["input"]
    lines = [f"# DeepTrace report: {i['file_name']}", "",
             f"- Media: {d['media_type']}, {i['width']}x{i['height']} px"
             + (f", {i['duration_s']:.1f} s" if d["media_type"] == "video" else ""),
             f"- File sha256: `{i['sha256']}`",
             f"- Analysed: {d['generated_at']} (evidence schema {d['schema_version']})",
             f"- Detector: {m['name']} {m['version']}, checkpoint sha256 `{m['checkpoint_sha256'][:12]}…`",
             "", _result(ev), ""]
    overlays = ev.overlay_files()
    if overlays:
        lines += ["## Overlay images", ""]
        lines += [f"![{'overlay' if k == 'image' else f'key frame {k}'}]({name})" for k, name in overlays.items()]
        lines += ["", "Highlighted boxes are model-indicated suspicious regions, not ground-truth manipulated pixels.", ""]
    for s in explanation.sections:
        lines += [f"## {s.title}", ""]
        if s.source == "llm":
            lines += [s.text, "", f"*{LLM_SOURCE}*", "", "Evidence:", ""]
        lines += [f"- {f}" for f in s.facts] + [""]
        if s.problems:
            lines += [f"*The language-model text for this section was rejected: {'; '.join(s.problems)}.*", ""]
    if explanation.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in explanation.notes] + [""]
    if answers:
        lines += ["## Questions and answers", ""]
        for a in answers:
            lines += [f"**Q:** {a.question}", "", f"**A:** {a.text}", ""]
            if a.suggestions:
                lines += ["Questions this analysis can answer:", ""] + [f"- {s}" for s in a.suggestions] + [""]
            if a.facts:
                lines += ["Evidence used:", ""] + [f"- {f}" for f in a.facts] + [""]
    if explanation.model:
        lines += [f"*Explanation model: {explanation.model} (local), {explanation.seconds} s.*", ""]
    lines += ["---", "", DISCLAIMER, ""]
    return "\n".join(lines)
