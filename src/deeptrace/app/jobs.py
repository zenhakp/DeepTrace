"""Background job runner for the API: one folder and one record per upload, two worker threads.

  analysis thread     queued -> analyzing -> done (evidence + template report at once) | failed (error code)
  explanation thread  explanation_state: pending -> generating -> ready (the LLM answered; sections that failed
                      the checks stay template) | unavailable (no LLM answer; template stays, notes say why)
NO_FACE and failed jobs get explanation_state "not_needed". Each thread handles one job at a time (both are CPU
heavy), and a new upload can be analysed while an earlier one is explained. The LLM never delays the report.

Records are kept in memory; the files stay in the job folder (whether records must survive a restart is step 7).
"""
from __future__ import annotations

import logging
import queue
import shutil
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from deeptrace.app.evidence import Evidence, EvidenceError
from deeptrace.app.explain import Explanation, explain, template_explanation
from deeptrace.app.llm import LLMSettings
from deeptrace.app.pipeline import PipelineError
from deeptrace.app.qa import Answer, ask

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    file_name: str
    media_type: str
    folder: Path
    created_at: str
    state: str = "queued"                   # queued | analyzing | done | failed
    explanation_state: str = "waiting"      # waiting | pending | generating | ready | unavailable | not_needed
    error: Optional[Dict[str, str]] = None  # {"code", "message"} when failed
    evidence: Optional[Evidence] = None
    explanation: Optional[Explanation] = None
    answers: List[Answer] = field(default_factory=list)

    @property
    def input_path(self) -> Path:
        return self.folder / self.file_name


class JobRunner:
    def __init__(self, jobs_dir: Path, analyze: Callable[[Path, Path], Evidence],
                 llm_settings: Optional[LLMSettings] = None, explain_fn: Callable = explain, ask_fn: Callable = ask):
        self.jobs_dir, self.analyze = Path(jobs_dir), analyze
        self.llm_settings, self.explain_fn, self.ask_fn = llm_settings, explain_fn, ask_fn
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._analysis: "queue.Queue[Optional[Job]]" = queue.Queue()
        self._explanation: "queue.Queue[Optional[Job]]" = queue.Queue()
        self._threads: List[threading.Thread] = []

    def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        for target, name in ((self._analysis_loop, "analysis"), (self._explanation_loop, "explanation")):
            t = threading.Thread(target=target, name=f"deeptrace-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._analysis.put(None)
        self._explanation.put(None)
        for t in self._threads:
            t.join(timeout_s)
        self._threads = []

    def new_job(self, file_name: str, media_type: str) -> Job:
        """Create the job folder; the caller writes the upload to job.input_path, then calls submit()."""
        job_id = uuid.uuid4().hex
        folder = self.jobs_dir / job_id
        folder.mkdir(parents=True)
        return Job(job_id, file_name, media_type, folder, datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def discard(self, job: Job) -> None:
        """Remove the folder of a job whose upload was refused (it was never submitted)."""
        shutil.rmtree(job.folder, ignore_errors=True)

    def submit(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
        self._analysis.put(job)

    def get(self, job_id: str) -> Optional[Job]:
        """A consistent copy (never half-updated by a worker thread), or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else replace(job, answers=list(job.answers))

    def list(self) -> List[Job]:
        with self._lock:
            jobs = [replace(j, answers=list(j.answers)) for j in self._jobs.values()]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def ask(self, job_id: str, question: str) -> Answer:
        """Synchronous (the caller waits for the LLM). The job must be done. Raises QuestionError."""
        answer = self.ask_fn(self.get(job_id).evidence, question, self.llm_settings)
        with self._lock:
            self._jobs[job_id].answers.append(answer)
        return answer

    def _set(self, job: Job, **values) -> None:
        with self._lock:
            for k, v in values.items():
                setattr(job, k, v)

    def _analysis_loop(self) -> None:
        while (job := self._analysis.get()) is not None:
            self._set(job, state="analyzing")
            try:
                ev = self.analyze(job.input_path, job.folder)
            except (PipelineError, EvidenceError) as e:
                self._set(job, state="failed", explanation_state="not_needed", error={"code": e.code, "message": str(e)})
                continue
            except Exception as e:  # never kill the worker thread; the job reports it
                log.exception("analysis of job %s failed", job.id)
                self._set(job, state="failed", explanation_state="not_needed",
                          error={"code": "internal_error", "message": f"{type(e).__name__}: {e}"})
                continue
            self._set(job, evidence=ev, explanation=template_explanation(ev), state="done",
                      explanation_state="pending" if ev.has_prediction else "not_needed")
            if ev.has_prediction:
                self._explanation.put(job)

    def _explanation_loop(self) -> None:
        while (job := self._explanation.get()) is not None:
            self._set(job, explanation_state="generating")
            try:
                ex = self.explain_fn(job.evidence, self.llm_settings)
            except Exception:  # explain() never raises for LLM problems; this is a bug, keep the template
                log.exception("explanation of job %s failed", job.id)
                self._set(job, explanation_state="unavailable")
                continue
            self._set(job, explanation=ex, explanation_state="ready" if ex.model else "unavailable")
