import json

from deeptrace.app.evidence import REPO_ROOT, load_evidence
from deeptrace.app.explain import explain
from deeptrace.app.llm import LLMReply, LLMSettings, LLMUnavailable
from deeptrace.app.qa import ask
from deeptrace.app.report import DISCLAIMER, LLM_SOURCE, render_markdown

EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"


def _ev(name):
    return load_evidence(EXAMPLES / f"example_{name}.json")


def _offline(*_, **__):
    raise LLMUnavailable("not_running", "test")


def _report(name, llm=_offline, answers=()):
    ev = _ev(name)
    return ev, render_markdown(ev, explain(ev, LLMSettings(), llm=llm), answers)


def test_template_report_has_every_fact_limitation_and_the_disclaimer():
    ev, md = _report("image")
    d = ev.data
    assert md.startswith("# DeepTrace report: example_image.jpg")
    assert "**Detector prediction:** MANIPULATED (fake-class probability 0.65)" in md
    for text in [d["localization"]["message"], d["evidence_stability"]["summary"]] + d["limitations"] + \
            [o["message"] for o in d["robustness_observations"]]:
        assert f"- {text}" in md
    assert DISCLAIMER in md and "The LLM explanation is unavailable" in md and LLM_SOURCE not in md


def test_overlays_are_linked_by_file_name():
    _, md = _report("video")
    for frame in ("24", "48", "120"):
        assert f"![key frame {frame}](example_video_frame{frame}_overlay.png)" in md
    assert "not ground-truth manipulated pixels" in md


def test_video_probability_is_not_rounded_to_certainty():
    _, md = _report("video")
    assert "(mean fake-class probability 0.99)" in md and "probability 1.00" not in md


def test_no_face_report_has_no_verdict():
    _, md = _report("no_face")
    assert "**No verdict:** no face was found" in md
    assert "Detector prediction:**" not in md and "MANIPULATED" not in md.split("## Limitations")[0]


def test_llm_text_is_marked_and_keeps_its_evidence():
    ev = _ev("image")
    text = "The detector assigned a fake-class probability of 0.65, and its prediction is MANIPULATED."
    answer = {"prediction": text, "evidence": "There are 3 model-indicated suspicious regions.",
              "stability": "The Evidence Stability Score is 0.64, band MEDIUM.",
              "robustness": "10 of 12 perturbed runs kept the prediction."}

    def llm(messages, settings, json_schema=None):
        return LLMReply(json.dumps(answer), "qwen3:4b-instruct", 900, 250, 0.0, 30.0, 40.0, 70.0)

    md = render_markdown(ev, explain(ev, LLMSettings(), llm=llm))
    section = md.split("## Detector prediction")[1].split("## ")[0]
    assert text in section and LLM_SOURCE in section and "- The detector's prediction is MANIPULATED." in section
    assert "*Explanation model: qwen3:4b-instruct (local), 70.0 s.*" in md


def test_answers_are_included_with_their_evidence():
    ev = _ev("no_face")
    a = ask(ev, "Is it fake?", LLMSettings(), llm=_offline)
    _, md = _report("no_face", answers=[a])
    assert "## Questions and answers" in md and "**Q:** Is it fake?" in md and f"**A:** {a.text}" in md
