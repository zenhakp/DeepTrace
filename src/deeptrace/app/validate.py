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
  label_contradicts_detector  a claim such as "the prediction is REAL" that is not the detector's label (needed
                   because the About-the-method facts mention both labels; general rules ending in "when" pass)
  verdict_claim    the media stated as fake / real / manipulated ("the image is manipulated", "likely fake")
  probability_as_certainty  a probability may not be given as 1 or 0 ("probability of 1.00") unless the facts do
                   (the digest writes 0.9994 as "above 0.99")
  region_probability   the regions section has no probability (regions have none of their own)
  cap_misattributed when the band was capped: stability called low, or lost points given as the reason for the band
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
CLAIMED_LABEL = re.compile(r"\b(?:prediction(?:\s+for\s+this\s+\w+)?\s+(?:is|was)|predicted(?:\s+it)?(?:\s+as)?|"
                           r"classified(?:\s+it)?(?:\s+as)?|labell?ed(?:\s+it)?(?:\s+as)?)\s+(REAL|MANIPULATED)\b"
                           r"(?!\s+when\b)", re.I)
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
BAND_WORD = re.compile(r"\b(?:HIGH|MEDIUM|LOW)\b")
CAP_WORD = re.compile(r"\bcap(?:s|ped|ping)?\b", re.I)
LOW_STABILITY = re.compile(r"\b(?:overall\s+)?stability\s+(?:is|was|remains)\s+(?:not\s+high|low|weak|poor)\b|"
                           r"\b(?:unstable|not\s+(?:very\s+)?stable)\b", re.I)
FACE_LOST = re.compile(r"\blost\s+faces?\b", re.I)  # '0 lost face(s) in 12 runs' is a count of runs, not points lost
CLAUSE = re.compile(r"[;,.]\s|\band\b")


def _matches(n: str, known: Sequence[str]) -> bool:
    """n is one of the known numbers, or one of them rounded to n's decimals (0.6518 -> 0.65, 1.0 -> 1)."""
    places = len(n.split(".")[1]) if "." in n else 0
    return any(round(float(k), places) == float(n) for k in known)


def grounding_facts(digest: Digest, keys: Sequence[str]) -> List[str]:
    """The facts a text about these sections may use: the LLM facts, or the shown facts for a section that is never
    explained by the LLM (limitations and warnings, which Q&A may quote)."""
    return [f for k in keys for f in (digest.section(k).llm_facts or digest.section(k).facts)]


def check_section(digest: Digest, key: str, text: str) -> List[str]:
    """One explanation section, checked against that section's facts only."""
    return check_text(digest, text, [key])


COMPARE = re.compile(r"\b(below|under|above|over)\s+(\d\.\d+)", re.I)
SCORE_SUBJECT = re.compile(r"\b(?:score|stability|it|this|band)\b|\d\.\d+", re.I)


def _false_comparisons(digest: Digest, text: str) -> List[str]:
    """'The score is below 0.80' when it is 0.82. Only comparisons with a band threshold are checked, and only
    where the words before them in the same clause could be about the score (a band's definition, 'LOW below 0.60',
    is not a claim about this file)."""
    problems = []
    for clause in CLAUSE.split(text):
        for m in COMPARE.finditer(clause):
            n = float(m.group(2))
            before = clause[max(0, m.start() - 25):m.start()]
            if n not in digest.thresholds or re.search(r"\b(?:HIGH|MEDIUM|LOW)\W*$", before) or \
                    not SCORE_SUBJECT.search(before):
                continue
            below = m.group(1).lower() in ("below", "under")
            if (below and not digest.score < n) or (not below and not digest.score > n):
                problems.append(f"score_comparison: {m.group(0)!r}, but the score is {digest.score:.2f}")
    return problems


def check_text(digest: Digest, text: str, keys: Sequence[str]) -> List[str]:
    """A text grounded on the given sections (one for an explanation section; the cited ones for a Q&A answer)."""
    if not isinstance(text, str) or not text.strip():
        return ["empty: no text"]
    if digest.status == "NO_FACE":
        found = NO_FACE_WORDS.findall(text) + NUMBER.findall(text)
        return [f"no_face: NO_FACE has no verdict, but the text uses {sorted(set(found))}"] if found else []
    facts = " ".join(grounding_facts(digest, keys))
    known = NUMBER.findall(facts)
    problems = [f"banned_word: {m.group(0)!r}" for m in BANNED.finditer(DISCLAIMER.sub(" ", text))]
    problems += [f"percent_score: {m.group(0)!r}" for m in PERCENT_SCORE.finditer(text)]
    problems += [f"percent_score: {m.group(0)!r} is not in the facts" for m in PERCENT.finditer(text)
                 if m.group(0) not in facts and f"{m.group(1)}%" not in facts]
    problems += [f"number_not_in_facts: {n}" for n in NUMBER.findall(text) if not _matches(n, known)]
    problems += [f"label_not_in_facts: {m.group(1)}" for m in LABEL.finditer(text) if m.group(1) not in facts]
    problems += [f"label_contradicts_detector: {m.group(0)!r}, but the detector predicted {digest.prediction}"
                 for m in CLAIMED_LABEL.finditer(text) if digest.prediction and m.group(1).upper() != digest.prediction]
    problems += [f"verdict_claim: {m.group(0)!r}" for m in VERDICT.finditer(text)
                 if not HEDGE.search(text[max(0, m.start() - 50):m.start()])]
    problems += [f"probability_as_certainty: {m.group(0)!r}" for m in PROBABILITY_VALUE.finditer(text)
                 if float(m.group(1)) in (0.0, 1.0)
                 and not re.search(rf"probability (?:of )?{re.escape(m.group(1))}(?![\d.])", facts)]
    if list(keys) == ["evidence"] and PROBABILITY.search(text):
        problems.append("region_probability: the regions section mentions a probability; regions have none")
    if "stability" in keys and digest.points_lost:
        for clause in CLAUSE.split(text):
            if COST.search(clause) and not FACE_LOST.search(clause):
                problems += [f"cost_not_points_lost: {n} is not one of the points lost {list(digest.points_lost)}"
                             for n in NUMBER.findall(clause) if n not in ("1.00", "1") and
                             not _matches(n, digest.points_lost)]
    if "stability" in keys and digest.band_capped:
        # the band was lowered by the cap rule, not by a low score: do not give lost points as the reason, and do
        # not call the stability low (the shown score is above the band's threshold)
        problems += [f"cap_misattributed: {m.group(0)!r}, but the band was capped and the score is not low"
                     for m in LOW_STABILITY.finditer(text)]
        problems += [f"cap_misattributed: a clause gives lost points as the reason for the band: {c.strip()!r}"
                     for c in CLAUSE.split(text)
                     if BAND_WORD.search(c) and COST.search(c) and not CAP_WORD.search(c)]
    if "stability" in keys and digest.score is not None:
        problems += _false_comparisons(digest, text)
    return list(dict.fromkeys(problems))  # keep order, drop repeats
