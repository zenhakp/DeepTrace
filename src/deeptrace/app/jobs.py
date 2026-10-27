"""Background job runner for the API: one folder and one record per upload, two worker threads.

  analysis thread     queued -> analyzing -> done (evidence + template report at once) | failed (error code)
  explanation thread  explanation_state: pending -> generating -> ready (the LLM answered; sections that failed
                      the checks stay template) | unavailable (no LLM answer; template stays, notes say why)
NO_FACE and failed jobs get explanation_state "not_needed". Each thread handles one job at a time (both are CPU
heavy), and a new upload can be analysed while an earlier one is explained. The LLM never delays the report.

Persistence (step 7): no database. Each job folder holds the upload, the evidence JSON, the overlays and job.json
(the record: state, error, explanation, answers), rewritten atomically on every change. On start-up the list is
rebuilt from the folders, and every evidence file is checked again with load_evidence(). Work that was running
when the server stopped is not resumed (no CPU-heavy work starts by itself): an interrupted analysis becomes
failed / interrupted, an interrupted LLM explanation keeps the template report (unavailable, with a note).
Deleting a job folder deletes the job.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from deeptrace.app.evidence import Evidence, EvidenceError, load_evidence
from deeptrace.app.explain import Explanation, explain, explanation_from_dict, template_explanation
from deeptrace.app.llm import LLMSettings
from deeptrace.app.pipeline import PipelineError
from deeptrace.app.qa import Answer, answer_from_dict, ask

log = logging.getLogger(__name__)
RECORD = "job.json"
RECORD_VERSION = 1
SAVE_ATTEMPTS = 5   # about 0.5 s in total before a save is given up (see _save)
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
INTERRUPTED_NOTE = "The LLM explanation was interrupted by a server restart; the evidence is shown as text instead."


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
        self._restore()
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
            self._save(job)
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
            self._save(self._jobs[job_id])
        return answer

    def _set(self, job: Job, **values) -> None:
        with self._lock:
            for k, v in values.items():
                setattr(job, k, v)
            self._save(job)

    # --- persistence (job.json in the job folder) ---

    def _save(self, job: Job) -> None:
        """Called with the lock held. Atomic: a crash leaves the old or the new record, never half of one.
        Never raises: the in-memory job stays correct and the next change rewrites the whole record. (On Windows,
        replacing a file fails while another program, such as a virus scanner, has it open; that is retried.)"""
        try:
            self._write_record(job)
        except OSError as e:
            log.warning("could not save %s for job %s (%s); the next change will try again", RECORD, job.id, e)

    def _write_record(self, job: Job) -> None:
        record = {"record_version": RECORD_VERSION, "id": job.id, "file_name": job.file_name,
                  "media_type": job.media_type, "created_at": job.created_at, "state": job.state,
                  "explanation_state": job.explanation_state, "error": job.error,
                  "evidence_file": job.evidence.source.name if job.evidence else None,
                  "explanation": asdict(job.explanation) if job.explanation else None,
                  "answers": [asdict(a) for a in job.answers]}
        tmp = job.folder / f"{RECORD}.tmp"
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        for attempt in range(SAVE_ATTEMPTS):
            try:
                os.replace(tmp, job.folder / RECORD)
                return
            except PermissionError:
                if attempt == SAVE_ATTEMPTS - 1:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(0.05 * (attempt + 1))

    def _restore(self) -> None:
        for folder in sorted(p for p in self.jobs_dir.iterdir() if p.is_dir() and JOB_ID.match(p.name)):
            if not (folder / RECORD).is_file():
                continue  # an upload that never got a record (refused, or created before step 7)
            try:
                job = self._load(folder)
            except (ValueError, KeyError, TypeError) as e:  # a damaged record: keep the folder, skip the job
                log.warning("skipping job folder %s: unreadable %s (%s)", folder.name, RECORD, e)
                continue
            with self._lock:
                self._jobs[job.id] = job
                self._save(job)

    def _load(self, folder: Path) -> Job:
        r = json.loads((folder / RECORD).read_text(encoding="utf-8"))
        if r["record_version"] != RECORD_VERSION or r["id"] != folder.name:
            raise ValueError(f"record_version {r['record_version']!r} / id {r['id']!r} do not match this folder")
        job = Job(r["id"], r["file_name"], r["media_type"], folder, r["created_at"], r["state"],
                  r["explanation_state"], r["error"],
                  explanation=explanation_from_dict(r["explanation"]) if r["explanation"] else None,
                  answers=[answer_from_dict(a) for a in r["answers"]])
        if job.state in ("queued", "analyzing"):
            job.state, job.explanation_state = "failed", "not_needed"
            job.error = {"code": "interrupted", "message": "the server stopped during the analysis; upload the "
                                                           "file again"}
        elif job.state == "done":
            try:  # the record is not trusted for the evidence: check the file again
                job.evidence = load_evidence(folder / r["evidence_file"])
            except EvidenceError as e:
                job.state, job.explanation_state, job.explanation = "failed", "not_needed", None
                job.error = {"code": e.code, "message": f"the saved evidence is no longer usable: {e}"}
            if job.explanation_state in ("pending", "generating"):
                job.explanation_state = "unavailable"
                job.explanation = replace(job.explanation, notes=job.explanation.notes + (INTERRUPTED_NOTE,))
        return job

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
