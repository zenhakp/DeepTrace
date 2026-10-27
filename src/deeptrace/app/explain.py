"""Grounded explanation: evidence -> digest -> prompt -> local LLM -> one short text per report section.

Grounding: the LLM sees only the digest's llm_facts (taken from the evidence JSON) plus the rules, and must answer
with a JSON object holding one text per LLM section (Ollama structured outputs). Every section keeps its shown facts,
so the deterministic template text (the facts themselves) is always available:
  - limitations and warnings: never sent to the LLM, always shown verbatim (schema rule 6; also the longest part
    of the prompt and of the answer, so leaving it out roughly halves the time),
  - NO_FACE: no LLM call at all (there is no verdict to explain; a fixed text is safest),
  - LLM unavailable or unusable answer: every section uses the template, and `notes` says why,
  - a section missing from the answer: that section uses the template,
  - a section text that fails the deterministic checks (validate.py): that section uses the template, and its
    problems are kept in `problems` and named in `notes`.
The system prompt is the same for every file, so Ollama can reuse its processed prefix while the model is loaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from deeptrace.app.digest import Digest, build_digest
from deeptrace.app.evidence import Evidence
from deeptrace.app.llm import LLMSettings, LLMUnavailable, chat
from deeptrace.app.validate import check_section

MAX_WORDS = 40
# Rules 1-7 are shared with Q&A (qa.py), so both follow exactly the same wording rules.
RULES = """Rules:
1. Use only the facts given. Do not add any verdict, score, number, region, experiment or result that is not in the \
facts. Copy numbers exactly as they are written in the facts; never turn them into percentages.
2. Never override or question the detector's prediction, and never give your own opinion on whether the media is \
real or fake.
3. Scores are not certainty. Write "the detector assigned a fake-class probability of 0.65", never "65% chance", \
"65% confident" or "likely fake".
4. Regions are "model-indicated suspicious regions", never "manipulated pixels" or "the manipulated area". Describe \
regions only by their rank and position; a region has no probability of its own.
5. The Evidence Stability Score is a project-defined 0 to 1 score with HIGH / MEDIUM / LOW bands. It is not a \
probability of manipulation. A component's value and the points it lost are different numbers; never swap them.
6. Never write "definitely", "certainly", "proves", "proof", "guaranteed" or "ground truth", except inside the phrase \
"not ground-truth manipulated pixels".
7. Do not claim novelty or state-of-the-art performance."""
SYSTEM = f"""You explain the output of a deepfake detector to a non-expert reader. The detector's result is the \
source of truth. You are not a detector: you only explain the FACTS you are given.

{RULES}
8. For each section write at most 2 short sentences and at most {MAX_WORDS} words in plain English, using only the \
facts of that section.

Answer with a JSON object with one string per section, using exactly the section keys listed after the facts."""


@dataclass(frozen=True)
class ExplainedSection:
    key: str
    title: str
    facts: Tuple[str, ...]
    text: str
    source: str                   # "llm" or "template"
    problems: Tuple[str, ...] = ()  # why an LLM text was rejected (then source is "template")


@dataclass(frozen=True)
class Explanation:
    sections: Tuple[ExplainedSection, ...]
    notes: Tuple[str, ...]          # why template text was used where an LLM text was expected, shown to the user
    model: Optional[str] = None     # set when the LLM was called successfully
    seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None   # tokens Ollama processed (a reused prompt prefix is not counted)
    output_tokens: Optional[int] = None

    @property
    def llm_used(self) -> bool:
        return any(s.source == "llm" for s in self.sections)


def template_text(facts: Tuple[str, ...]) -> str:
    return " ".join(facts)


def output_schema(digest: Digest) -> Dict[str, Any]:
    keys = [s.key for s in digest.llm_sections()]
    return {"type": "object", "properties": {k: {"type": "string"} for k in keys}, "required": keys,
            "additionalProperties": False}


def build_messages(digest: Digest) -> List[Dict[str, str]]:
    sections = digest.llm_sections()
    facts = "\n\n".join(f"## {s.key} ({s.title})\n" + "\n".join(f"- {f}" for f in s.llm_facts) for s in sections)
    user = (f"Wording rule from the evidence (its number is only an example, not a result for this file):\n"
            f"{digest.score_semantics}\n\nFACTS for this {digest.media_type}:\n\n{facts}\n\n"
            f"Section keys: {', '.join(s.key for s in sections)}. Write the JSON object now.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _section(s, text=None, problems=()) -> ExplainedSection:
    return ExplainedSection(s.key, s.title, s.facts, text or template_text(s.facts), "llm" if text else "template",
                            tuple(problems))


def _parse(text: str) -> Dict[str, Any]:
    try:
        answer = json.loads(text)
    except ValueError:
        raise LLMUnavailable("bad_response", "the LLM answer is not JSON") from None
    if not isinstance(answer, dict):
        raise LLMUnavailable("bad_response", "the LLM answer is not a JSON object")
    return answer


NO_FACE_NOTE = "No face was found, so there is no verdict to explain; the LLM was not used."


def template_explanation(ev: Evidence) -> Explanation:
    """The deterministic report alone (the app shows it at once, before any LLM text exists)."""
    digest = build_digest(ev)
    return Explanation(tuple(_section(s) for s in digest.sections), () if ev.has_prediction else (NO_FACE_NOTE,))


def explain(ev: Evidence, settings: Optional[LLMSettings] = None, llm: Callable = chat) -> Explanation:
    """Never raises for LLM problems: they become template sections plus a note."""
    if not ev.has_prediction:
        return template_explanation(ev)
    digest = build_digest(ev)
    settings = settings or LLMSettings.from_config()
    try:
        reply = llm(build_messages(digest), settings, json_schema=output_schema(digest))
        answer = _parse(reply.text)
    except LLMUnavailable as e:
        return Explanation(tuple(_section(s) for s in digest.sections),
                           (f"The LLM explanation is unavailable ({e}); the evidence is shown as text instead.",))
    sections, notes = [], []
    for s in digest.sections:
        if not s.llm_facts:
            sections.append(_section(s))
            continue
        text = answer.get(s.key)
        if not (isinstance(text, str) and text.strip()):
            notes.append(f"The LLM gave no text for '{s.title}'; the evidence is shown as text instead.")
            sections.append(_section(s))
            continue
        problems = check_section(digest, s.key, text.strip())
        if problems:
            notes.append(f"The LLM text for '{s.title}' did not pass the evidence checks ({'; '.join(problems)}); "
                         f"the evidence is shown as text instead.")
            sections.append(_section(s, problems=problems))
        else:
            sections.append(_section(s, text.strip()))
    return Explanation(tuple(sections), tuple(notes), reply.model, round(reply.total_s, 1), reply.prompt_tokens,
                       reply.output_tokens)


def explanation_from_dict(d: Dict[str, Any]) -> Explanation:
    """The inverse of dataclasses.asdict(explanation) (used when job records are read back from disk)."""
    sections = tuple(ExplainedSection(s["key"], s["title"], tuple(s["facts"]), s["text"], s["source"],
                                      tuple(s.get("problems", ()))) for s in d["sections"])
    return Explanation(sections, tuple(d["notes"]), d.get("model"), d.get("seconds"), d.get("prompt_tokens"),
                       d.get("output_tokens"))
