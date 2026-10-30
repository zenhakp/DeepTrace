"""Run the Part 2 pipeline on one uploaded file: scripts/analyze.py as a subprocess (Decision B).

The subprocess runs exactly the code path Part 2 tested and schema-checked, keeps torch out of the API process, and
frees its memory before the LLM runs. analyze.py writes <stem>_evidence.json and overlay PNG(s) into the job folder;
the evidence is then loaded and re-checked with load_evidence(). Failures raise PipelineError with a stable code:
  checkpoint_missing  <DEEPTRACE_ROOT>/checkpoints/<run>/best.pt does not exist (checked before starting)
  analysis_timeout    analyze.py did not finish within analysis_timeout_s
  analysis_failed     analyze.py exited with an error (the last lines of its error output are kept)
  evidence_missing    analyze.py finished but wrote no evidence file
Evidence problems raise EvidenceError (evidence.py) with its own codes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from deeptrace.app.evidence import REPO_ROOT, Evidence, load_evidence
from deeptrace.common.paths import resolve_root

ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze.py"
# Same lists as deeptrace.forensics.evidence (copied: importing that module loads torch). If they ever differ,
# analyze.py still refuses the file and the job fails with analysis_failed.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ERROR_TAIL_LINES = 8


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def media_type_of(name: str) -> Optional[str]:
    ext = Path(name).suffix.lower()
    return "image" if ext in IMAGE_EXTENSIONS else "video" if ext in VIDEO_EXTENSIONS else None


def checkpoint_path(run_name: str) -> Path:
    return resolve_root() / "checkpoints" / run_name / "best.pt"


def analyze_command(input_path: Path, out_dir: Path, run_name: str) -> List[str]:
    return [sys.executable, str(ANALYZE_SCRIPT), "--input", str(input_path), "--out-dir", str(out_dir),
            "--run-name", run_name]


def run_analysis(input_path: Path, out_dir: Path, run_name: str = "mixed", timeout_s: float = 900,
                 command: Optional[Callable[[Path, Path, str], List[str]]] = None) -> Evidence:
    """command: builds the subprocess command (tests pass a fake one; then no checkpoint check is made)."""
    if command is None:
        ckpt = checkpoint_path(run_name)
        if not ckpt.is_file():
            raise PipelineError("checkpoint_missing", f"detector checkpoint not found at {ckpt}. Set DEEPTRACE_ROOT "
                                                      f"and place best.pt there (docs/detector_api.md)")
        command = analyze_command
    try:
        proc = subprocess.run(command(input_path, out_dir, run_name), cwd=REPO_ROOT, env=os.environ.copy(),
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise PipelineError("analysis_timeout", f"the analysis did not finish within {timeout_s:g} s") from None
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-ERROR_TAIL_LINES:])
        raise PipelineError("analysis_failed", f"analyze.py exited with code {proc.returncode}: {tail}")
    evidence = out_dir / f"{input_path.stem}_evidence.json"
    if not evidence.is_file():
        raise PipelineError("evidence_missing", f"analyze.py wrote no {evidence.name}")
    return load_evidence(evidence)
