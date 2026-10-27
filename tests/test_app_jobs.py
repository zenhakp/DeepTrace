"""Job records survive a restart (step 7): a new JobRunner on the same folder rebuilds the list from job.json."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from deeptrace.app.api import AppSettings, create_app
from deeptrace.app.jobs import INTERRUPTED_NOTE, RECORD, JobRunner
from deeptrace.app.llm import LLMSettings

from test_app_api import _closed_port_settings, fake_analyze, fake_llm, make_client, upload, wait


def _restart(tmp_path):
    """A fresh runner and app on the same jobs folder, as after a server restart."""
    runner = JobRunner(tmp_path / "jobs", fake_analyze(), LLMSettings(), explain_fn=lambda ev, s: None)
    return TestClient(create_app(AppSettings(tmp_path / "jobs"), runner, _closed_port_settings()))


def _finished_job(tmp_path, name="photo.jpg", question=True):
    with make_client(tmp_path) as c:
        job = wait(c, upload(c, name).json()["job_id"], "explanation_state", {"ready", "not_needed"})
        if question:
            c.post(f"/api/jobs/{job['job_id']}/questions", json={"question": "Why MEDIUM?"})
        return c.get(f"/api/jobs/{job['job_id']}").json()


def _edit_record(tmp_path, job_id, **changes):
    path = tmp_path / "jobs" / job_id / RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(changes)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_finished_job_is_restored_unchanged(tmp_path):
    before = _finished_job(tmp_path)
    with _restart(tmp_path) as c:
        after = c.get(f"/api/jobs/{before['job_id']}").json()
        assert after == before  # state, evidence, LLM explanation, answers, overlays
        assert c.get(after["overlays"]["image"]).status_code == 200
        assert "**Q:** Why MEDIUM?" in c.get(f"/api/jobs/{before['job_id']}/report").text
        assert [j["job_id"] for j in c.get("/api/jobs").json()["jobs"]] == [before["job_id"]]


def test_no_face_and_failed_jobs_are_restored(tmp_path):
    no_face = _finished_job(tmp_path, "no_face.jpg", question=False)
    with _restart(tmp_path) as c:
        assert c.get(f"/api/jobs/{no_face['job_id']}").json() == no_face


def test_interrupted_analysis_becomes_failed(tmp_path):
    job = _finished_job(tmp_path, question=False)
    _edit_record(tmp_path, job["job_id"], state="analyzing", explanation_state="waiting", explanation=None,
                 evidence_file=None)
    with _restart(tmp_path) as c:
        after = c.get(f"/api/jobs/{job['job_id']}").json()
        assert after["state"] == "failed" and after["error"]["code"] == "interrupted"
        assert after["explanation_state"] == "not_needed"
    on_disk = json.loads((tmp_path / "jobs" / job["job_id"] / RECORD).read_text(encoding="utf-8"))
    assert on_disk["error"]["code"] == "interrupted"  # written back, so it is stable across restarts


@pytest.mark.parametrize("state", ["pending", "generating"])
def test_interrupted_explanation_keeps_the_template(tmp_path, state):
    job = _finished_job(tmp_path, question=False)
    sections = job["explanation"]["sections"]
    template = [{**s, "source": "template", "text": " ".join(s["facts"])} for s in sections]
    _edit_record(tmp_path, job["job_id"], explanation_state=state,
                 explanation={**job["explanation"], "sections": template, "notes": [], "model": None})
    with _restart(tmp_path) as c:
        after = c.get(f"/api/jobs/{job['job_id']}").json()
        assert after["state"] == "done" and after["explanation_state"] == "unavailable"
        assert after["explanation"]["notes"] == [INTERRUPTED_NOTE]
        assert all(s["source"] == "template" for s in after["explanation"]["sections"])


def test_evidence_is_checked_again_on_restore(tmp_path):
    job = _finished_job(tmp_path, question=False)
    (tmp_path / "jobs" / job["job_id"] / "photo_evidence.json").write_text('{"schema_version": "2.0"}')
    with _restart(tmp_path) as c:
        after = c.get(f"/api/jobs/{job['job_id']}").json()
        assert after["state"] == "failed" and after["error"]["code"] == "unsupported_schema_version"
        assert after["evidence"] is None and after["explanation"] is None


def test_deleted_evidence_file(tmp_path):
    job = _finished_job(tmp_path, question=False)
    (tmp_path / "jobs" / job["job_id"] / "photo_evidence.json").unlink()
    with _restart(tmp_path) as c:
        assert c.get(f"/api/jobs/{job['job_id']}").json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("damage", ["{not json", '{"record_version": 99}', "[]"])
def test_damaged_record_is_skipped_and_others_load(tmp_path, damage):
    good, bad = _finished_job(tmp_path, question=False), _finished_job(tmp_path, "clip.mp4", question=False)
    (tmp_path / "jobs" / bad["job_id"] / RECORD).write_text(damage)
    with _restart(tmp_path) as c:
        assert [j["job_id"] for j in c.get("/api/jobs").json()["jobs"]] == [good["job_id"]]
    assert (tmp_path / "jobs" / bad["job_id"]).is_dir()  # nothing is deleted


def test_record_for_another_folder_is_skipped(tmp_path):
    job = _finished_job(tmp_path, question=False)
    _edit_record(tmp_path, job["job_id"], id="f" * 32)
    with _restart(tmp_path) as c:
        assert c.get("/api/jobs").json() == {"jobs": []}


def test_folders_without_a_record_are_ignored(tmp_path):
    (tmp_path / "jobs" / ("a" * 32)).mkdir(parents=True)
    (tmp_path / "jobs" / "not-a-job").mkdir()
    with _restart(tmp_path) as c:
        assert c.get("/api/jobs").json() == {"jobs": []}


def test_record_is_written_when_the_job_is_submitted(tmp_path):
    with make_client(tmp_path) as c:
        job_id = upload(c).json()["job_id"]
        record = json.loads((tmp_path / "jobs" / job_id / RECORD).read_text(encoding="utf-8"))
        assert record["id"] == job_id and record["record_version"] == 1
        wait(c, job_id, "explanation_state", {"ready"})
    final = json.loads((tmp_path / "jobs" / job_id / RECORD).read_text(encoding="utf-8"))
    assert final["state"] == "done" and final["explanation_state"] == "ready"
    assert not (tmp_path / "jobs" / job_id / f"{RECORD}.tmp").exists()


def test_a_failing_save_never_stops_a_job(tmp_path, monkeypatch, caplog):
    """Windows refuses to replace a file another program has open; jobs must still finish."""
    import deeptrace.app.jobs as jobs_mod

    def locked(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(jobs_mod.os, "replace", locked)
    monkeypatch.setattr(jobs_mod.time, "sleep", lambda s: None)
    with make_client(tmp_path) as c:
        first = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})
        second = wait(c, upload(c, "clip.mp4").json()["job_id"], "explanation_state", {"ready"})  # worker alive
        assert first["state"] == second["state"] == "done"
    assert "could not save job.json" in caplog.text
    assert not list((tmp_path / "jobs").glob(f"*/{RECORD}.tmp"))


def test_a_briefly_locked_record_is_retried(tmp_path, monkeypatch):
    import deeptrace.app.jobs as jobs_mod
    real, calls = jobs_mod.os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) % 2:  # every first attempt fails, the retry succeeds
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(jobs_mod.os, "replace", flaky)
    with make_client(tmp_path) as c:
        job_id = wait(c, upload(c).json()["job_id"], "explanation_state", {"ready"})["job_id"]
    record = json.loads((tmp_path / "jobs" / job_id / RECORD).read_text(encoding="utf-8"))
    assert record["explanation_state"] == "ready"
