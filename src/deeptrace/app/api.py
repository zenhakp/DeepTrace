"""DeepTrace FastAPI backend (Part 3). Local only.

  uvicorn deeptrace.app.api:app --host 127.0.0.1 --port 8000

  GET  /api/health                      checkpoint found? LLM reachable and model installed?
  POST /api/jobs                        upload one image or video (multipart field "file") -> 202 + job
  GET  /api/jobs                        all jobs in the jobs folder, newest first (summaries)
  GET  /api/jobs/{id}                   state, error, evidence, explanation (template at once, LLM text later)
  GET  /api/jobs/{id}/files/{name}      an overlay PNG listed in the evidence artifacts (nothing else)
  GET  /api/jobs/{id}/report            the Markdown report (current explanation + answers)
  POST /api/jobs/{id}/questions         {"question": "..."} -> 202 + queued question (answered in the background)

Errors are {"detail": {"code", "message"}} with a fitting HTTP status. Poll GET /api/jobs/{id}: "state" says when
the report is ready, "explanation_state" when the LLM text is, and each question's "state" when it is answered
(see jobs.py). No request waits for the LLM. The detector is the source of truth;
this API only runs it (scripts/analyze.py) and explains its evidence.
"""
from __future__ import annotations

import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from deeptrace.app.evidence import REPO_ROOT
from deeptrace.app.jobs import JOB_ID, Job, JobRunner
from deeptrace.app.llm import DEFAULT_CONFIG, LLMSettings, LLMUnavailable, check
from deeptrace.app.pipeline import (IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, checkpoint_path, media_type_of,
                                    run_analysis)
from deeptrace.app.qa import QuestionError
from deeptrace.app.report import render_markdown
from deeptrace.common.config import load_config

CHUNK = 1024 * 1024


@dataclass(frozen=True)
class AppSettings:
    jobs_dir: Path
    max_upload_mb: float = 100
    analysis_timeout_s: float = 900
    run_name: str = "mixed"

    @classmethod
    def from_config(cls, path=None) -> "AppSettings":
        a = load_config(path or DEFAULT_CONFIG)["app"]
        return cls(Path(a["jobs_dir"]).expanduser(), a["max_upload_mb"], a["analysis_timeout_s"], a["run_name"])


class QuestionBody(BaseModel):
    question: str


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, detail={"code": code, "message": message})


def safe_file_name(name: str) -> str:
    """Keep letters, digits, '.', '_' and '-' of the stem (max 80), so the name is safe on disk and in outputs."""
    p = Path(name or "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("._") or "upload"
    return f"{stem[:80]}{p.suffix.lower()}"


def job_summary(job: Job) -> Dict[str, Any]:
    return {"job_id": job.id, "file_name": job.file_name, "media_type": job.media_type,
            "created_at": job.created_at, "state": job.state, "explanation_state": job.explanation_state,
            "error": job.error, "prediction": job.evidence.data["prediction"] if job.evidence else None}


def job_view(job: Job) -> Dict[str, Any]:
    ev = job.evidence
    return {**job_summary(job),
            "evidence": ev.data if ev else None,
            "explanation": asdict(job.explanation) if job.explanation else None,
            "overlays": {k: f"/api/jobs/{job.id}/files/{name}" for k, name in ev.overlay_files().items()} if ev else {},
            "questions": [asdict(q) for q in job.questions]}


def create_app(settings: Optional[AppSettings] = None, runner: Optional[JobRunner] = None,
               llm_settings: Optional[LLMSettings] = None) -> FastAPI:
    settings = settings or AppSettings.from_config()
    llm_settings = llm_settings or LLMSettings.from_config()
    jobs_dir = settings.jobs_dir.resolve()
    if jobs_dir == REPO_ROOT or REPO_ROOT in jobs_dir.parents:
        raise ValueError(f"app.jobs_dir is inside the repository ({jobs_dir}); outputs must not be committed")
    if runner is None:
        runner = JobRunner(jobs_dir, lambda inp, out: run_analysis(inp, out, settings.run_name,
                                                                   settings.analysis_timeout_s), llm_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runner.start()
        yield
        runner.stop()

    app = FastAPI(title="DeepTrace", description=__doc__, lifespan=lifespan)
    app.state.runner = runner

    def get_job(job_id: str) -> Job:
        job = runner.get(job_id) if JOB_ID.match(job_id) else None
        if job is None:
            raise _error(404, "job_not_found", f"no job {job_id!r}")
        return job

    def done_job(job_id: str) -> Job:
        job = get_job(job_id)
        if job.state != "done":
            raise _error(409, "job_not_ready", f"the job is {job.state}; wait until it is done")
        return job

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        try:
            check(llm_settings)
            llm = {"available": True, "model": llm_settings.model, "problem": None}
        except LLMUnavailable as e:
            llm = {"available": False, "model": llm_settings.model, "problem": str(e)}
        return {"status": "ok", "checkpoint_found": checkpoint_path(settings.run_name).is_file(), "llm": llm}

    @app.post("/api/jobs", status_code=202)
    def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
        name = safe_file_name(file.filename)
        media = media_type_of(name)
        if media is None:
            raise _error(415, "unsupported_media_type", f"{Path(file.filename or '').suffix or 'no extension'!r} "
                                                        f"is not supported; images {sorted(IMAGE_EXTENSIONS)}, "
                                                        f"videos {sorted(VIDEO_EXTENSIONS)}")
        limit = int(settings.max_upload_mb * 1024 * 1024)
        job, size = runner.new_job(name, media), 0
        try:
            with open(job.input_path, "wb") as out:
                while chunk := file.file.read(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise _error(413, "file_too_large", f"the upload is over {settings.max_upload_mb:g} MB")
                    out.write(chunk)
            if size == 0:
                raise _error(400, "empty_file", "the uploaded file is empty")
        except BaseException:
            runner.discard(job)
            raise
        runner.submit(job)
        return job_view(runner.get(job.id))

    @app.get("/api/jobs")
    def list_jobs() -> Dict[str, Any]:
        return {"jobs": [job_summary(j) for j in runner.list()]}

    @app.get("/api/jobs/{job_id}")
    def get(job_id: str) -> Dict[str, Any]:
        return job_view(get_job(job_id))

    @app.get("/api/jobs/{job_id}/files/{name}")
    def overlay(job_id: str, name: str) -> FileResponse:
        job = get_job(job_id)
        allowed = set(job.evidence.overlay_files().values()) if job.evidence else set()
        path = job.folder / name
        if name not in allowed or not path.is_file():
            raise _error(404, "file_not_found", f"{name!r} is not an overlay of this job")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/jobs/{job_id}/report", response_class=PlainTextResponse)
    def report(job_id: str) -> PlainTextResponse:
        job = done_job(job_id)
        return PlainTextResponse(render_markdown(job.evidence, job.explanation, job.answers),
                                 media_type="text/markdown; charset=utf-8")

    @app.post("/api/jobs/{job_id}/questions", status_code=202)
    def question(job_id: str, body: QuestionBody) -> Dict[str, Any]:
        job = done_job(job_id)
        try:
            return asdict(runner.ask(job.id, body.question))
        except QuestionError as e:  # a question already open is a conflict (409), a bad question a 400
            raise _error(409 if e.code == "question_in_progress" else 400, e.code, str(e)) from None

    return app


def __getattr__(name: str):
    """`uvicorn deeptrace.app.api:app` builds the app on first use, so importing this module reads no config."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
