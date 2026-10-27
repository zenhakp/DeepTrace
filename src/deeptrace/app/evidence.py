"""Load and check one Part 2 evidence document (schema 1.0) before Part 3 explains or shows it.

Part 3 depends only on the document and docs/part2_evidence_schema.json, not on Part 2's Python code (importing
deeptrace.forensics would also load torch). Checks, in order:
  1. the file is readable JSON (NaN / Infinity are rejected: they are not valid JSON),
  2. schema_version is supported (another version is a breaking change agreed with the Part 2 owner),
  3. the document matches the JSON Schema (full draft 2020-12 validator),
  4. cross-field rules the schema cannot express hold, for example NO_FACE has no verdict.
Unknown extra fields are allowed and kept (schema rule 7).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "docs" / "part2_evidence_schema.json"
SUPPORTED_SCHEMA_VERSIONS = ("1.0",)
MAX_BYTES = 5 * 1024 * 1024  # real documents are 4-12 KB; this only stops absurd inputs
VERDICT_FIELDS = ("prediction", "confidence", "fake_probability", "real_probability")
STABILITY_FIELDS = ("prediction_stability", "confidence_stability", "localization_stability", "evidence_stability",
                    "temporal_stability")


class EvidenceError(ValueError):
    """A document Part 3 must not explain. `code` is stable (the API will return it); `errors` lists every problem."""

    def __init__(self, code: str, errors: List[str]):
        super().__init__(f"{code}: " + "; ".join(errors[:10]))
        self.code, self.errors = code, errors


@dataclass(frozen=True)
class Evidence:
    """A checked document. `data` is the original dict, extra fields included; read it, do not modify it."""
    data: Dict[str, Any]
    source: Optional[Path] = None

    @property
    def has_prediction(self) -> bool:
        return self.data["status"] == "OK"

    @property
    def is_video(self) -> bool:
        return self.data["media_type"] == "video"

    def overlay_files(self) -> Dict[str, str]:
        """Overlay PNG names next to the document: {"image": name}, or {frame: name} for a video's key frames.

        `artifacts` is free-form in the schema, so only plain file names are returned (no folders, no ".."): the
        backend will serve these files and must not be pointed outside the job folder.
        """
        a = self.data["artifacts"]
        names = {"image": a.get("overlay")} if not self.is_video else dict(a.get("key_frame_overlays") or {})
        return {k: v for k, v in names.items()
                if isinstance(v, str) and v not in ("", ".", "..") and Path(v).name == v and "\\" not in v}


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


def _location(path) -> str:
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in path)


def _consistency_errors(d: Dict[str, Any]) -> List[str]:
    errors = []
    if d["status"] == "NO_FACE":
        errors += [f"$.{k}: must be null when status is NO_FACE (no verdict exists)"
                   for k in VERDICT_FIELDS + STABILITY_FIELDS if d[k] is not None]
        if d["suspicious_regions"]:
            errors.append("$.suspicious_regions: must be empty when status is NO_FACE")
    else:
        errors += [f"$.{k}: must not be null when status is OK" for k in VERDICT_FIELDS if d[k] is None]
        p, label = d["fake_probability"], d["prediction"]
        if p is not None and ((label == "REAL" and p > 0.5) or (label == "MANIPULATED" and p < 0.5)):
            errors.append(f"$.prediction: {label} contradicts fake_probability {p}")
    if d["media_type"] == "image" and (d["frames"] is not None or d["temporal_stability"] is not None):
        errors.append("$.frames / $.temporal_stability: must be null for an image")
    if d["media_type"] == "video" and d["frames"] is None:
        errors.append("$.frames: must be a list for a video")
    return errors


def check_evidence(doc: Any, source: Optional[Path] = None) -> Evidence:
    """Checks 2-4 on an already parsed document; raises EvidenceError."""
    if not isinstance(doc, dict):
        raise EvidenceError("schema_violation", [f"$: expected a JSON object, got {type(doc).__name__}"])
    version = doc.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise EvidenceError("unsupported_schema_version",
                            [f"$.schema_version: {version!r}; this app supports {list(SUPPORTED_SCHEMA_VERSIONS)}"])
    found = sorted(_validator().iter_errors(doc), key=lambda e: [str(p) for p in e.absolute_path])
    if found:
        raise EvidenceError("schema_violation", [f"{_location(e.absolute_path)}: {e.message[:200]}" for e in found])
    errors = _consistency_errors(doc)
    if errors:
        raise EvidenceError("inconsistent", errors)
    return Evidence(doc, source)


def parse_evidence(raw: Union[bytes, str], source: Optional[Path] = None) -> Evidence:
    """Check 1, then check_evidence. A UTF-8 byte-order mark (written by some Windows tools) is accepted."""
    try:
        text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw.lstrip("﻿")
        doc = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:  # includes JSONDecodeError and UnicodeDecodeError
        raise EvidenceError("invalid_json", [str(e)]) from None
    return check_evidence(doc, source)


def load_evidence(path) -> Evidence:
    """Read and check an evidence file; raises EvidenceError (code not_found, too_large or one of the above)."""
    p = Path(path)
    if not p.is_file():
        raise EvidenceError("not_found", [f"{p} does not exist or is not a file"])
    if p.stat().st_size > MAX_BYTES:
        raise EvidenceError("too_large", [f"{p.name} is {p.stat().st_size} bytes (limit {MAX_BYTES})"])
    return parse_evidence(p.read_bytes(), source=p)
