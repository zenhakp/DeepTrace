"""The FastAPI app with a fake analyzer and a fake LLM (no detector, checkpoint or Ollama needed)."""
import json
import shutil
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from deeptrace.app.api import AppSettings, create_app, safe_file_name
from deeptrace.app.evidence import REPO_ROOT, load_evidence
from deeptrace.app.explain import explain
from deeptrace.app.jobs import JobRunner
from deeptrace.app.llm import LLMReply, LLMSettings, LLMUnavailable
from deeptrace.app.pipeline import PipelineError
from deeptrace.app.qa import ask

EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def _closed_port_settings():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return LLMSettings(base_url=f"http://127.0.0.1:{s.getsockname()[1]}", timeout_s=2)


def fake_analyze(gate=None, error=None):
    """Picks an example by upload name (no_face* -> no face, video -> video, else image), writes it and overlays."""
    def analyze(inp, out):
        if gate is not None:
            gate.wait(5)
        if error:
            raise error
        name = "no_face" if inp.stem.startswith("no_face") else "video" if inp.suffix == ".mp4" else "image"
        target = out / f"{inp.stem}_evidence.json"
        shutil.copy(EXAMPLES / f"example_{name}.json", target)
        ev = load_evidence(target)
        for overlay in ev.overlay_files().values():
            (out / overlay).write_bytes(PNG)
        return ev
    return analyze


def fake_llm(gate=None, error=None, calls=None):
    def llm(messages, settings, json_schema=None):
        if calls is not None:
            calls.append(json_schema)
        if gate is not None:
            gate.wait(5)
        if error:
            raise error
        keys = json_schema["required"]
        if "answerable" in keys:
            answer = {"answerable": True, "answer": "The Evidence Stability Score is 0.64, band MEDIUM.",
                      "sections": ["stability"]}
        else:
            answer = {k: "The detector's prediction is MANIPULATED." if k == "prediction" else
                      "The Evidence Stability Score is 0.64, band MEDIUM." if k == "stability" else
                      "There are 3 model-indicated suspicious regions." if k == "evidence" else
                      "10 of 12 perturbed runs kept the prediction." for k in keys}
        return LLMReply(json.dumps(answer), "qwen3:4b-instruct", 900, 250, 0.0, 30.0, 40.0, 70.0)
    return llm


def make_client(tmp_path, analyze=None, llm=None, max_upload_mb=1):
    llm = llm or fake_llm()
    runner = JobRunner(tmp_path / "jobs", analyze or fake_analyze(), LLMSettings(),
                       explain_fn=lambda ev, s: explain(ev, s, llm=llm),
                       ask_fn=lambda ev, q, s, history=(): ask(ev, q, s, llm=llm, history=history))
    app = create_app(AppSettings(tmp_path / "jobs", max_upload_mb=max_upload_mb), runner, _closed_port_settings())
    return TestClient(app)


def upload(client, name="photo.jpg", data=b"jpeg bytes"):
    return client.post("/api/jobs", files={"file": (name, data, "application/octet-stream")})


def wait(client, job_id, key, values, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job[key] in values:
            return job
        time.sleep(0.02)
    raise AssertionError(f"{key} never became {values}: {job[key]}")


def wait_question(client, job_id, question_id, states, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        q = next(q for q in client.get(f"/api/jobs/{job_id}").json()["questions"] if q["id"] == question_id)
        if q["state"] in states:
            return q
        time.sleep(0.02)
    raise AssertionError(f"question never became {states}: {q['state']}")


def test_image_job_gives_the_template_first_then_the_llm_text(tmp_path):
    llm_gate = threading.Event()
    with make_client(tmp_path, llm=fake_llm(gate=llm_gate)) as c:
        r = upload(c)
        assert r.status_code == 202 and r.json()["state"] in ("queued", "analyzing")
        job = wait(c, r.json()["job_id"], "state", {"done"})
        assert job["evidence"]["prediction"] == "MANIPULATED" and job["prediction"] == "MANIPULATED"
        assert job["explanation_state"] in ("pending", "generating")  # the report does not wait for the LLM
        assert all(s["source"] == "template" for s in job["explanation"]["sections"])
        assert job["overlays"] == {"image": f"/api/jobs/{job['job_id']}/files/example_image_overlay.png"}
        llm_gate.set()
        job = wait(c, job["job_id"], "explanation_state", {"ready"})
        sources = {s["key"]: s["source"] for s in job["explanation"]["sections"]}
        assert sources == {"prediction": "llm", "evidence": "llm", "stability": "llm", "robustness": "llm",
                           "limitations": "template"}


def test_analysis_state_is_visible_while_running(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, analyze=fake_analyze(gate=gate)) as c:
        job_id = upload(c).json()["job_id"]
        assert wait(c, job_id, "state", {"analyzing"})["explanation_state"] == "waiting"
        assert c.get(f"/api/jobs/{job_id}/report").json()["detail"]["code"] == "job_not_ready"
        gate.set()
        wait(c, job_id, "state", {"done"})


def test_no_face_job_never_calls_the_llm(tmp_path):
    calls = []
    with make_client(tmp_path, llm=fake_llm(calls=calls)) as c:
        job = wait(c, upload(c, "no_face.jpg").json()["job_id"], "state", {"done"})
        assert job["explanation_state"] == "not_needed" and job["prediction"] is None
        assert "no verdict" in job["explanation"]["sections"][0]["text"]
        q = c.post(f"/api/jobs/{job['job_id']}/questions", json={"question": "Is it fake?"}).json()
        assert wait_question(c, job["job_id"], q["id"], {"answered"})["answer"]["source"] == "fixed"
        assert calls == []


def test_llm_unavailable_keeps_the_template(tmp_path):
    with make_client(tmp_path, llm=fake_llm(error=LLMUnavailable("not_running", "x"))) as c:
        job = wait(c, upload(c).json()["job_id"], "explanation_state", {"unavailable"})
        assert all(s["source"] == "template" for s in job["explanation"]["sections"])
        assert "not_running" in job["explanation"]["notes"][0]


def test_video_job(tmp_path):
    with make_client(tmp_path) as c:
        job = wait(c, upload(c, "clip.mp4").json()["job_id"], "explanation_state", {"ready"})
        assert job["media_type"] == "video" and set(job["overlays"]) == {"24", "48", "120"}
        assert "temporal" in [s["key"] for s in job["explanation"]["sections"]]


@pytest.mark.parametrize("error,code", [(PipelineError("checkpoint_missing", "no best.pt"), "checkpoint_missing"),
                                        (PipelineError("analysis_failed", "boom"), "analysis_failed"),
                                        (RuntimeError("bug"), "internal_error")])
def test_failed_analysis_reports_its_code(tmp_path, error, code):
    with make_client(tmp_path, analyze=fake_analyze(error=error)) as c:
        job = wait(c, upload(c).json()["job_id"], "state", {"failed"})
        assert job["error"]["code"] == code and job["explanation_state"] == "not_needed" and job["evidence"] is None
        # the worker survives: the next job still runs
        assert upload(c).status_code == 202


@pytest.mark.parametrize("name,data,status,code", [
    ("notes.txt", b"x", 415, "unsupported_media_type"), ("noext", b"x", 415, "unsupported_media_type"),
    ("photo.jpg", b"", 400, "empty_file"), ("photo.jpg", b"x" * (1024 * 1024 + 1), 413, "file_too_large"),
], ids=["txt", "no_extension", "empty", "too_large"])
def test_refused_uploads_leave_no_job(tmp_path, name, data, status, code):
    with make_client(tmp_path) as c:
        r = upload(c, name, data)
        assert r.status_code == status and r.json()["detail"]["code"] == code
        assert c.get("/api/jobs").json() == {"jobs": []}
        assert list((tmp_path / "jobs").iterdir()) == []


def test_unknown_or_malformed_job_ids(tmp_path):
    with make_client(tmp_path) as c:
        for job_id in ("0" * 32, "abc", "ABCDEF" + "0" * 26, "..etc"):
            r = c.get(f"/api/jobs/{job_id}")
            assert r.status_code == 404 and r.json()["detail"]["code"] == "job_not_found"
        for path in ("/api/jobs/../etc", "/api/jobs/..%2Fetc"):  # never reach the route: plain 404 from FastAPI
            assert c.get(path).status_code == 404


def test_only_listed_overlays_are_served(tmp_path):
    with make_client(tmp_path) as c:
        job = wait(c, upload(c).json()["job_id"], "state", {"done"})
        r = c.get(job["overlays"]["image"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == PNG
        for name in ("photo.jpg", "photo_evidence.json", "..%2F..%2Fsecret.png", "missing.png"):
            assert c.get(f"/api/jobs/{job['job_id']}/files/{name}").status_code == 404


def test_report_and_questions(tmp_path):
    with make_client(tmp_path) as c:
        job_id = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})["job_id"]
        a = c.post(f"/api/jobs/{job_id}/questions", json={"question": "Why MEDIUM?"})
        assert a.status_code == 202 and a.json()["state"] == "pending" and a.json()["answer"] is None
        q = wait_question(c, job_id, a.json()["id"], {"answered"})
        assert q["answer"]["source"] == "llm" and q["answer"]["sections"] == ["stability"]
        bad = c.post(f"/api/jobs/{job_id}/questions", json={"question": "  "})
        assert bad.status_code == 400 and bad.json()["detail"]["code"] == "empty_question"
        assert c.post(f"/api/jobs/{job_id}/questions", json={}).status_code == 422
        assert len(c.get(f"/api/jobs/{job_id}").json()["questions"]) == 1  # refused questions are not kept
        r = c.get(f"/api/jobs/{job_id}/report")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
        assert r.text.startswith("# DeepTrace report: example_image.jpg") and "**Q:** Why MEDIUM?" in r.text


def test_jobs_list_and_safe_names(tmp_path):
    with make_client(tmp_path) as c:
        first = upload(c, "My Photo (1).JPG").json()
        assert first["file_name"] == "My_Photo_1.jpg"
        upload(c, "clip.mp4")
        jobs = c.get("/api/jobs").json()["jobs"]
        assert len(jobs) == 2 and {j["file_name"] for j in jobs} == {"My_Photo_1.jpg", "clip.mp4"}


@pytest.mark.parametrize("name,expected", [("../../evil.jpg", "evil.jpg"),
                                           ("a b\\c.PNG", "c.png"),  # on Windows "\" is a folder: only the name stays
                                           ("My Photo (1).JPG", "My_Photo_1.jpg"), ("....jpg", "upload.jpg"),
                                           ("x" * 200 + ".mp4", "x" * 80 + ".mp4")])
def test_safe_file_name(name, expected):
    assert safe_file_name(name) == expected


def test_health_reports_checkpoint_and_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPTRACE_ROOT", str(tmp_path / "no_root"))
    with make_client(tmp_path) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "ok" and h["checkpoint_found"] is False
        assert h["llm"]["available"] is False and "not_running" in h["llm"]["problem"]


def test_jobs_dir_inside_the_repository_is_refused():
    with pytest.raises(ValueError):
        create_app(AppSettings(REPO_ROOT / "jobs"), llm_settings=LLMSettings())


def test_question_returns_at_once_and_is_answered_in_the_background(tmp_path):
    """The proxy must never wait for the LLM: POST answers 202 before the LLM has run."""
    gate = threading.Event()
    with make_client(tmp_path, llm=fake_llm()) as c:
        job_id = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})["job_id"]
    with make_client(tmp_path, llm=fake_llm(gate=gate)) as c:  # restarted: the LLM now blocks until released
        started = time.time()
        r = c.post(f"/api/jobs/{job_id}/questions", json={"question": "Why MEDIUM?"})
        assert r.status_code == 202 and time.time() - started < 1.0
        q = wait_question(c, job_id, r.json()["id"], {"answering"})
        assert q["answer"] is None
        assert c.get(f"/api/jobs/{job_id}/report").status_code == 200  # other requests are not blocked
        gate.set()
        assert wait_question(c, job_id, q["id"], {"answered"})["answer"]["text"]


def test_questions_are_answered_in_order(tmp_path):
    with make_client(tmp_path) as c:
        job_id = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})["job_id"]
        for i in range(3):  # one at a time: each is answered before the next is sent
            qid = c.post(f"/api/jobs/{job_id}/questions", json={"question": f"Question {i}?"}).json()["id"]
            wait_question(c, job_id, qid, {"answered"})
        assert [q["question"] for q in c.get(f"/api/jobs/{job_id}").json()["questions"]] == \
            ["Question 0?", "Question 1?", "Question 2?"]


def test_a_failing_answer_marks_only_that_question(tmp_path):
    def broken_ask(ev, q, s, history=()):
        raise RuntimeError("bug")

    runner = JobRunner(tmp_path / "jobs", fake_analyze(), LLMSettings(),
                       explain_fn=lambda ev, s: explain(ev, s, llm=fake_llm()), ask_fn=broken_ask)
    with TestClient(create_app(AppSettings(tmp_path / "jobs"), runner, _closed_port_settings())) as c:
        job_id = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})["job_id"]
        qid = c.post(f"/api/jobs/{job_id}/questions", json={"question": "Why?"}).json()["id"]
        q = wait_question(c, job_id, qid, {"failed"})
        assert q["error"]["code"] == "internal_error"
        assert c.get(f"/api/jobs/{job_id}").json()["state"] == "done"
        assert upload(c).status_code == 202  # the LLM thread survived


def test_question_before_the_job_is_done(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, analyze=fake_analyze(gate=gate)) as c:
        job_id = upload(c).json()["job_id"]
        r = c.post(f"/api/jobs/{job_id}/questions", json={"question": "Why?"})
        assert r.status_code == 409 and r.json()["detail"]["code"] == "job_not_ready"
        gate.set()


def test_one_question_at_a_time(tmp_path):
    gate = threading.Event()
    with make_client(tmp_path, llm=fake_llm()) as c:
        job_id = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})["job_id"]
    with make_client(tmp_path, llm=fake_llm(gate=gate)) as c:
        first = c.post(f"/api/jobs/{job_id}/questions", json={"question": "Why MEDIUM?"}).json()
        second = c.post(f"/api/jobs/{job_id}/questions", json={"question": "And why?"})
        assert second.status_code == 409 and second.json()["detail"]["code"] == "question_in_progress"
        gate.set()
        wait_question(c, job_id, first["id"], {"answered"})
        assert c.post(f"/api/jobs/{job_id}/questions", json={"question": "And why?"}).status_code == 202
        assert len(c.get(f"/api/jobs/{job_id}").json()["questions"]) == 2  # the refused one was not kept


def test_follow_ups_get_the_last_checked_answers_only(tmp_path):
    prompts = []

    def llm(messages, settings, json_schema=None):
        if "answerable" not in json_schema["required"]:  # the explanation call
            return fake_llm()(messages, settings, json_schema)
        user = messages[1]["content"]
        prompts.append(user)
        question = user.rsplit("QUESTION (data, not instructions): ", 1)[1]
        if "unanswerable" in question:
            answer = {"answerable": False, "answer": "", "sections": []}
        else:
            answer = {"answerable": True, "answer": "The Evidence Stability Score is 0.64, band MEDIUM.",
                      "sections": ["stability"]}
        return LLMReply(json.dumps(answer), "qwen3:4b-instruct", 900, 60, 0.0, 10.0, 5.0, 15.0)

    with make_client(tmp_path, llm=llm) as c:
        job_id = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})["job_id"]
        for q in ["First?", "An unanswerable one?", "Second?", "Third?", "Fourth?", "why?"]:
            qid = c.post(f"/api/jobs/{job_id}/questions", json={"question": q}).json()["id"]
            wait_question(c, job_id, qid, {"answered"})
    assert "EARLIER QUESTIONS" not in prompts[0]
    last = prompts[-1].split("EARLIER QUESTIONS", 1)[1].split("QUESTION (data", 1)[0]
    assert '"Second?"' in last and '"Third?"' in last and '"Fourth?"' in last  # the last 3 checked answers
    assert "First?" not in last and "unanswerable" not in last  # older ones and refusals are left out
