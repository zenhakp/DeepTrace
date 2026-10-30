"""explain() with a fake LLM function (no Ollama needed)."""
import json

import pytest

from deeptrace.app.digest import build_digest
from deeptrace.app.evidence import REPO_ROOT, load_evidence
from deeptrace.app.explain import build_messages, explain, output_schema, template_text
from deeptrace.app.llm import LLMReply, LLMSettings, LLMUnavailable

EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"
SETTINGS = LLMSettings()


def _ev(name):
    return load_evidence(EXAMPLES / f"example_{name}.json")


def fake_llm(answer=None, error=None, calls=None):
    def llm(messages, settings, json_schema=None):
        if calls is not None:
            calls.append((messages, json_schema))
        if error:
            raise error
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return LLMReply(text, "qwen3:4b-instruct", 900, 300, 0.0, 30.0, 40.0, 70.04)
    return llm


def _all_keys(name):
    return {s.key: f"Text for {s.key}." for s in build_digest(_ev(name)).llm_sections()}


def test_no_face_never_calls_the_llm():
    calls = []
    ex = explain(_ev("no_face"), SETTINGS, llm=fake_llm(_all_keys("no_face"), calls=calls))
    assert calls == [] and not ex.llm_used
    assert all(s.source == "template" for s in ex.sections)
    assert "no verdict" in ex.sections[0].text and "LLM was not used" in ex.notes[0]


def test_llm_answer_fills_every_section():
    ex = explain(_ev("image"), SETTINGS, llm=fake_llm(_all_keys("image")))
    assert ex.llm_used and ex.notes == () and ex.model == "qwen3:4b-instruct" and ex.seconds == 70.0
    assert [(s.key, s.text, s.source) for s in ex.sections][0] == ("prediction", "Text for prediction.", "llm")
    assert all(s.facts for s in ex.sections)  # facts are always kept for the UI
    assert {s.key: s.source for s in ex.sections}["limitations"] == "template"  # always verbatim, no note
    assert (ex.prompt_tokens, ex.output_tokens) == (900, 300)


@pytest.mark.parametrize("code", ["not_running", "model_missing", "timeout", "bad_response"])
def test_llm_unavailable_falls_back_to_template(code):
    ex = explain(_ev("video"), SETTINGS, llm=fake_llm(error=LLMUnavailable(code, "x")))
    assert not ex.llm_used and code in ex.notes[0]
    assert [s.text for s in ex.sections] == [template_text(s.facts) for s in build_digest(_ev("video")).sections]


@pytest.mark.parametrize("answer", ["not json at all", "[1, 2]", '"text"'])
def test_unusable_answer_falls_back_to_template(answer):
    ex = explain(_ev("image"), SETTINGS, llm=fake_llm(answer))
    assert not ex.llm_used and "bad_response" in ex.notes[0]


def test_missing_or_empty_section_uses_template_for_that_section():
    answer = _all_keys("image")
    del answer["stability"]
    answer["robustness"] = "   "
    ex = explain(_ev("image"), SETTINGS, llm=fake_llm({**answer, "extra_key": "ignored"}))
    by_key = {s.key: s for s in ex.sections}
    assert by_key["stability"].source == by_key["robustness"].source == "template"
    assert by_key["prediction"].source == "llm" and len(ex.notes) == 2
    assert "extra_key" not in by_key


def test_prompt_contains_rules_wording_rule_and_every_fact():
    ev = _ev("video")
    dg = build_digest(ev)
    system, user = build_messages(dg)
    assert system["role"] == "system" and user["role"] == "user"
    assert ev.data["score_semantics"] in user["content"] and "only an example" in user["content"]
    assert "never turn them into percentages" in system["content"] and "never swap them" in system["content"]
    assert "a region has no probability of its own" in system["content"] and "at most 40 words" in system["content"]
    assert "Section keys: prediction, evidence, stability, robustness, temporal." in user["content"]
    assert all(f in user["content"] for s in dg.llm_sections() for f in s.llm_facts)
    sent = {f for s in dg.llm_sections() for f in s.llm_facts}  # robustness.note is also a limitation: allowed
    only_limitations = [f for f in dg.section("limitations").facts if f not in sent]
    assert only_limitations and not any(f in user["content"] for f in only_limitations)


def test_system_prompt_is_identical_for_every_file():
    systems = {build_messages(build_digest(_ev(n)))[0]["content"] for n in ("image", "video")}
    assert len(systems) == 1


def test_output_schema_matches_sections():
    for name in ("image", "video"):
        dg = build_digest(_ev(name))
        schema = output_schema(dg)
        assert schema["required"] == [s.key for s in dg.llm_sections()] and schema["additionalProperties"] is False


def test_llm_receives_the_schema():
    calls = []
    explain(_ev("image"), SETTINGS, llm=fake_llm(_all_keys("image"), calls=calls))
    assert calls[0][1] == output_schema(build_digest(_ev("image")))
