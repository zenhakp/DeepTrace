import json

import pytest

from deeptrace.app.digest import build_digest
from deeptrace.app.evidence import REPO_ROOT, check_evidence
from deeptrace.app.explain import explain
from deeptrace.app.llm import LLMReply, LLMSettings
from deeptrace.app.validate import check_section

EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"


def _doc(name):
    return json.loads((EXAMPLES / f"example_{name}.json").read_text(encoding="utf-8"))


def _real_image():
    d = _doc("image")
    d.update(prediction="REAL", fake_probability=0.3, real_probability=0.7, confidence=0.7)
    d["localization"].update(evidence_strength="weak", message="Weak MANIPULATED evidence: ... transparency only.")
    return d


def _dg(d):
    return build_digest(check_evidence(d))


def _rules(problems):
    return {p.split(":")[0] for p in problems}


@pytest.mark.parametrize("make", [lambda: _doc("image"), lambda: _doc("video"), _real_image])
def test_repeating_the_sections_facts_always_passes(make):
    dg = _dg(make())
    for s in dg.llm_sections():
        assert check_section(dg, s.key, " ".join(s.llm_facts)) == [], s.key


def test_real_prediction_never_definitely_fake():
    problems = check_section(_dg(_real_image()), "prediction", "The image is definitely fake.")
    assert {"banned_word", "verdict_claim"} <= _rules(problems)


def test_real_prediction_cannot_be_called_manipulated():
    problems = check_section(_dg(_real_image()), "prediction", "The detector's prediction is MANIPULATED.")
    assert _rules(problems) == {"label_not_in_facts"}


@pytest.mark.parametrize("text", ["The image is REAL.", "The detector found it authentic.",
                                  "The fake-class probability is 0.10.", "No verdict: confidence was low.",
                                  "It has 3 regions."])
def test_no_face_allows_no_verdict_score_or_number(text):
    assert _rules(check_section(_dg(_doc("no_face")), "prediction", text)) == {"no_face"}


def test_no_face_plain_statement_passes():
    text = "No face was found in this image, so the detector made no prediction and there is no verdict."
    assert check_section(_dg(_doc("no_face")), "prediction", text) == []


@pytest.mark.parametrize("text", ["This proves the image was altered.", "The result is definitely reliable.",
                                  "A state-of-the-art detector.", "These are the manipulated pixels.",
                                  "The ground truth is MANIPULATED.", "It is certainly a deepfake."])
def test_banned_words(text):
    assert "banned_word" in _rules(check_section(_dg(_doc("image")), "prediction", text))


def test_required_disclaimer_is_allowed():
    text = "There are 3 model-indicated suspicious regions; they are not ground-truth manipulated pixels."
    assert check_section(_dg(_doc("image")), "evidence", text) == []


@pytest.mark.parametrize("text", ["There is a 65% chance it was altered.", "The model is 65 percent confident."])
def test_percent_scores(text):
    assert "percent_score" in _rules(check_section(_dg(_doc("image")), "prediction", text))


def test_numbers_must_come_from_the_section():
    dg = _dg(_doc("image"))
    assert check_section(dg, "prediction", "The detector assigned a fake-class probability of 0.65.") == []
    assert _rules(check_section(dg, "prediction", "The fake-class probability is 0.72.")) == {"number_not_in_facts"}
    # 0.84 is a fact of the stability section, not of the prediction section
    assert _rules(check_section(dg, "prediction", "Localization stability is 0.84.")) == {"number_not_in_facts"}


def test_rounding_of_a_fact_is_allowed_but_not_rounding_up_to_certainty():
    dg = _dg(_doc("video"))
    assert check_section(dg, "temporal", "The maximum fake-class probability was above 0.99.") == []
    problems = check_section(dg, "temporal", "The maximum fake-class probability was 1.00.")
    assert _rules(problems) == {"probability_as_certainty"}  # 1.00 is in this section (temporal stability 1.00)
    # Known gap (validate.py docstring): without the word "probability" the reused number is not caught.
    assert check_section(dg, "temporal", "The maximum was 1.00.") == []


@pytest.mark.parametrize("text,ok", [
    ("The image is manipulated.", False), ("The video looks fake.", False), ("This is likely fake.", False),
    ("The score is not a probability that the image is manipulated.", True),
])
def test_verdict_claims(text, ok):
    assert ("verdict_claim" in _rules(check_section(_dg(_doc("image")), "stability", text))) != ok


def test_region_section_has_no_probability():
    text = "The regions have a high probability of being edited."
    assert "region_probability" in _rules(check_section(_dg(_doc("image")), "evidence", text))


@pytest.mark.parametrize("text,ok", [
    ("The confidence component lost the most points (0.25); its value is 0.18.", True),
    ("Confidence is 0.18 and lost 0.25 points; localization is 0.84 and lost 0.05 points.", True),
    ("Points lost by each component (out of 1.00): prediction 0.07, confidence 0.25.", True),
    ("The confidence stability cost the most points (0.18).", False),
    ("The component with the highest cost is confidence stability at 0.18.", False),
])
def test_stability_cost_numbers(text, ok):
    assert ("cost_not_points_lost" in _rules(check_section(_dg(_doc("image")), "stability", text))) != ok


def test_empty_text():
    assert check_section(_dg(_doc("image")), "prediction", "  ") == ["empty: no text"]


def test_explain_replaces_a_rejected_section_with_the_template():
    d = _doc("image")
    dg = _dg(d)
    answer = {s.key: " ".join(s.llm_facts) for s in dg.llm_sections()}
    answer["evidence"] = "The detector assigned a fake-class probability of 0.65 in the suspicious regions."

    def llm(messages, settings, json_schema=None):
        return LLMReply(json.dumps(answer), "qwen3:4b-instruct", 900, 250, 0.0, 30.0, 40.0, 70.0)

    ex = explain(check_evidence(d), LLMSettings(), llm=llm)
    by_key = {s.key: s for s in ex.sections}
    assert by_key["evidence"].source == "template" and by_key["evidence"].text.startswith(d["localization"]["message"])
    assert _rules(by_key["evidence"].problems) == {"number_not_in_facts", "region_probability"}
    assert by_key["prediction"].source == "llm" and by_key["prediction"].problems == ()
    assert len(ex.notes) == 1 and "did not pass the evidence checks" in ex.notes[0]
