import json
import re

import pytest

from deeptrace.app.digest import build_digest
from deeptrace.app.evidence import REPO_ROOT, check_evidence

EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"


def _doc(name):
    return json.loads((EXAMPLES / f"example_{name}.json").read_text(encoding="utf-8"))


def _digest(d):
    return build_digest(check_evidence(d))


@pytest.mark.parametrize("name,keys", [
    ("image", ["prediction", "evidence", "stability", "robustness", "limitations"]),
    ("video", ["prediction", "evidence", "stability", "robustness", "temporal", "limitations"]),
    ("no_face", ["prediction", "limitations"]),
])
def test_sections(name, keys):
    assert [s.key for s in _digest(_doc(name)).sections] == keys


def test_no_face_has_no_verdict_or_score():
    dg = _digest(_doc("no_face"))
    text = " ".join(dg.section("prediction").facts)
    assert "no verdict" in text and "image" in text
    assert not re.search(r"\bREAL\b|\bMANIPULATED\b|\d\.\d", text)


@pytest.mark.parametrize("name", ["image", "video", "no_face"])
def test_limitations_and_warnings_are_complete_and_verbatim(name):
    d = _doc(name)
    facts = _digest(d).section("limitations").facts
    assert [f"Warning: {w['message']}" for w in d["warnings"]] + d["limitations"] == list(facts)


def test_part2_sentences_are_quoted_unchanged():
    d = _doc("image")
    facts = _digest(d).all_facts()
    for sentence in [d["localization"]["message"], d["evidence_stability"]["summary"], d["evidence_stability"]["note"],
                     d["robustness"]["note"]] + [o["message"] for o in d["robustness_observations"]]:
        assert sentence in facts


def test_image_prediction_facts():
    facts = _digest(_doc("image")).section("prediction").facts
    assert "The detector's prediction is MANIPULATED." in facts
    assert "The detector assigned a fake-class probability of 0.65." in facts


def test_probability_near_one_never_reads_as_one():
    facts = " ".join(_digest(_doc("video")).all_facts())
    assert "probability 1.00" not in facts and "maximum 1.00" not in facts
    assert "fake-class probability above 0.99" in facts


def test_video_facts():
    dg = _digest(_doc("video"))
    assert "11 of the 11 frames with a usable face were predicted MANIPULATED." in dg.section("prediction").facts
    assert any(f.startswith("Suspicious interval from 0.0 s to 5.0 s") for f in dg.section("temporal").facts)
    assert sum(f.startswith("Key frame at") for f in dg.section("evidence").facts) == 3


def test_band_cap_reason_is_shown():
    d = _doc("image")
    d["evidence_stability"].update(band_capped=True, cap_reason="Capped at MEDIUM: 2 prediction flip(s).")
    assert "Capped at MEDIUM: 2 prediction flip(s)." in _digest(d).section("stability").facts


def test_weak_localization_message_is_passed_through():
    d = _doc("image")
    msg = "Weak MANIPULATED evidence: ... regions shown for transparency only."
    d.update(prediction="REAL", fake_probability=0.3, real_probability=0.7, confidence=0.7)
    d["localization"].update(evidence_strength="weak", message=msg)
    assert msg in _digest(d).section("evidence").facts


def test_no_robustness_flips_is_stated():
    d = _doc("image")
    d["robustness_observations"] = []
    assert "No prediction flips or lost faces were observed in these perturbation runs." in \
        _digest(d).section("robustness").facts


def test_video_without_temporal_stability_or_intervals():
    d = _doc("video")
    d["temporal_stability"], d["suspicious_intervals"] = None, []
    facts = " ".join(_digest(d).section("temporal").facts)
    assert "Temporal stability is not available" in facts and "No suspicious interval" in facts


def test_digest_is_deterministic():
    assert _digest(_doc("video")) == _digest(_doc("video"))


def test_limitations_are_never_sent_to_the_llm():
    for name in ("image", "video", "no_face"):
        dg = _digest(_doc(name))
        assert dg.section("limitations").llm_facts == ()
        assert "limitations" not in [s.key for s in dg.llm_sections()]
    assert _digest(_doc("no_face")).llm_sections() == []


# Known failure 1 (step 3 live run): the LLM put the face crop's fake-class probability "in the regions".
@pytest.mark.parametrize("name", ["image", "video"])
def test_region_facts_for_the_llm_carry_no_probability(name):
    facts = " ".join(_digest(_doc(name)).section("evidence").llm_facts)
    assert "probability" not in facts and not re.search(r"0\.\d", facts)
    assert "not ground-truth manipulated pixels" in facts


def test_regions_are_described_by_rank_and_position():
    facts = _digest(_doc("image")).section("evidence").llm_facts
    assert [f for f in facts if f.startswith("Region")] == [
        "Region 1 (rank 1): middle-left of the face crop.", "Region 2 (rank 2): middle-right of the face crop.",
        "Region 3 (rank 3): lower-centre of the face crop."]


def test_shown_region_facts_still_quote_the_localization_message():
    d = _doc("image")
    assert _digest(d).section("evidence").facts[0] == d["localization"]["message"]


@pytest.mark.parametrize("strength", ["weak", "none"])
def test_weak_and_none_strength_are_explained_for_the_llm(strength):
    d = _doc("image")
    d.update(prediction="REAL", fake_probability=0.3, real_probability=0.7, confidence=0.7)
    d["localization"]["evidence_strength"] = strength
    first = _digest(d).section("evidence").llm_facts[0]
    assert first.startswith(f"Evidence strength: {strength}")
    assert ("transparency only" in first) == (strength == "weak")


# Known failure 2 (step 3 live run): the LLM called the confidence component's value (0.18) its cost (0.25).
def test_stability_value_and_cost_are_separate_sentences():
    llm = _digest(_doc("image")).section("stability").llm_facts
    assert "Component values (0 to 1, higher is more stable): prediction 0.83, confidence 0.18, localization 0.84." in llm
    assert "Points lost by each component (out of 1.00): prediction 0.07, confidence 0.25, localization 0.05." in llm
    assert "The confidence component lost the most points (0.25); its value is 0.18." in llm
    assert not any("Largest cost" in f for f in llm)  # Part 2's combined phrase is shown to the user only


def test_stability_summary_is_still_shown_to_the_user():
    d = _doc("image")
    assert d["evidence_stability"]["summary"] in _digest(d).section("stability").facts
