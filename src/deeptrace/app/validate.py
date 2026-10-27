"""Deterministic checks of one LLM section text against the evidence digest. No second LLM.

check_section(digest, key, text) returns a list of problems; empty = the text may be shown. Any problem means the
section shows its template text (the evidence facts) instead. The checks are scoped to the section's own LLM facts,
because the real failures seen so far were facts moved to the wrong place (a face-level probability "in the
regions", a component value called its cost). They catch wording patterns, not meaning: accurate text can be
rejected (by design: the template is always correct), and wrong text that reuses the section's numbers can pass.

Rules (schema doc "Rules for Part 3" and the project brief):
  no_face          NO_FACE has no verdict: no label, no fake/real wording, no probability, no number
  banned_word      definitely, proves, guaranteed, state-of-the-art, ... ("not ground-truth manipulated pixels" is
                   the required disclaimer and allowed)
  percent_score    a score written as a percentage or as certainty ("65% confident", "65% chance")
  number_not_in_facts  every number must be one of the section's numbers (rounding allowed: 0.6518 -> 0.65)
  label_not_in_facts   REAL / MANIPULATED only where the section's facts use that label
  verdict_claim    the media stated as fake / real / manipulated ("the image is manipulated", "likely fake")
  probability_as_certainty  a probability may not be given as 1 or 0 ("probability of 1.00") unless the facts do
                   (the digest writes 0.9994 as "above 0.99")
  region_probability   the regions section has no probability (regions have none of their own)
  cost_not_points_lost a number in a clause about cost / points lost must be one of the points lost
Known gap: a wrong number that also occurs elsewhere in the same section passes (for example "the maximum was 1.00"
in a video's temporal section, which also says "temporal stability is 1.00"), unless a rule above names the clause.
"""
from __future__ import annotations

import re
from typing import List, Sequence

from deeptrace.app.digest import Digest

NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
BANNED = re.compile(r"\b(?:definitely|definitive(?:ly)?|certainly|with certainty|undoubtedly|without (?:a )?doubt|"
                    r"proves?|proven|proof|guarantee[sd]?|conclusive(?:ly)?|state[- ]of[- ]the[- ]art|novel|"
                    r"ground[- ]truth|manipulated pixels)\b", re.I)
DISCLAIMER = re.compile(r"\bnot\s+(?:\w+\s+){0,2}?(?:ground[- ]truth(?:\s+manipulated\s+pixels)?|manipulated\s+pixels)",
                        re.I)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
PERCENT_SCORE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent)\s*(?:chance|confident|confidence|likely|likelihood|"
                           r"probability|probable|sure|certain)", re.I)
LABEL = re.compile(r"\b(REAL|MANIPULATED)\b")
VERDICT = re.compile(r"\b(?:image|video|photo|picture|face|media|frame|clip|it)\s+(?:is|was|appears to be|seems to be|"
                     r"looks)\s+(?:\w+ly\s+)?(?:a\s+)?(?:fake|real|manipulated|authentic|genuine|deepfake|forged|"
                     r"altered)\b|"
                     r"\b(?:likely|probably)\s+(?:fake|real|manipulated|authentic|genuine|a deepfake)\b", re.I)
HEDGE = re.compile(r"probability that|likelihood that|whether|\bnot\b|\bno\b|\bnever\b", re.I)
NO_FACE_WORDS = re.compile(r"\b(?:real|fake|manipulated|authentic|genuine|deepfake|probability|confidence)\b", re.I)
COST = re.compile(r"\b(?:cost|costs|lost|loses|points)\b", re.I)
PROBABILITY = re.compile(r"\bprobabilit(?:y|ies)\b", re.I)
# the number that is the probability's value: "probability of 1.00", "probability was 1", "probability is at 0"
PROBABILITY_VALUE = re.compile(r"\bprobabilit(?:y|ies)(?:\s+(?:of|was|is|at|reached|equal to)){0,2}\s+"
                               r"(\d+(?:\.\d+)?)(?!\w|\.\d)", re.I)
CLAUSE = re.compile(r"[;,.]\s|\band\b")


def _matches(n: str, known: Sequence[str]) -> bool:
    """n is one of the known numbers, or one of them rounded to n's decimals (0.6518 -> 0.65, 1.0 -> 1)."""
    places = len(n.split(".")[1]) if "." in n else 0
    return any(round(float(k), places) == float(n) for k in known)


def check_section(digest: Digest, key: str, text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return ["empty: no text"]
    if digest.status == "NO_FACE":
        found = NO_FACE_WORDS.findall(text) + NUMBER.findall(text)
        return [f"no_face: NO_FACE has no verdict, but the text uses {sorted(set(found))}"] if found else []
    facts = " ".join(digest.section(key).llm_facts)
    known = NUMBER.findall(facts)
    problems = [f"banned_word: {m.group(0)!r}" for m in BANNED.finditer(DISCLAIMER.sub(" ", text))]
    problems += [f"percent_score: {m.group(0)!r}" for m in PERCENT_SCORE.finditer(text)]
    problems += [f"percent_score: {m.group(0)!r} is not in the facts" for m in PERCENT.finditer(text)
                 if m.group(0) not in facts and f"{m.group(1)}%" not in facts]
    problems += [f"number_not_in_facts: {n}" for n in NUMBER.findall(text) if not _matches(n, known)]
    problems += [f"label_not_in_facts: {m.group(1)}" for m in LABEL.finditer(text) if m.group(1) not in facts]
    problems += [f"verdict_claim: {m.group(0)!r}" for m in VERDICT.finditer(text)
                 if not HEDGE.search(text[max(0, m.start() - 50):m.start()])]
    problems += [f"probability_as_certainty: {m.group(0)!r}" for m in PROBABILITY_VALUE.finditer(text)
                 if float(m.group(1)) in (0.0, 1.0)
                 and not re.search(rf"probability (?:of )?{re.escape(m.group(1))}(?![\d.])", facts)]
    if key == "evidence" and PROBABILITY.search(text):
        problems.append("region_probability: the regions section mentions a probability; regions have none")
    if key == "stability" and digest.points_lost:
        for clause in CLAUSE.split(text):
            if COST.search(clause):
                problems += [f"cost_not_points_lost: {n} is not one of the points lost {list(digest.points_lost)}"
                             for n in NUMBER.findall(clause) if n not in ("1.00", "1") and
                             not _matches(n, digest.points_lost)]
    return list(dict.fromkeys(problems))  # keep order, drop repeats
