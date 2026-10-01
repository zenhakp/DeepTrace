"""Revision-pinned dataset download from the Hugging Face Hub.

Only datasets that are openly downloadable are handled here. Gated datasets are not
supported on purpose. Every download writes a record (repo, revision, file sizes) next
to the data so the exact dataset version used in an experiment is documented.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RECORD_NAME = "download_record.json"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int
    sha256: Optional[str] = None


def dataset_entry(cfg: Dict[str, Any], name: str, require_files: bool = True) -> Dict[str, Any]:
    """Return the validated config entry for one dataset."""
    entries = cfg.get("datasets") or {}
    if name not in entries:
        raise ValueError(
            f"Invalid configuration: unknown dataset '{name}'. Known: {sorted(entries)}")
    entry = entries[name]
    if not entry.get("hf_repo"):
        raise ValueError(f"Invalid configuration: datasets.{name}.hf_repo is missing")
    if not _SHA_RE.match(str(entry.get("revision", ""))):
        raise ValueError(
            f"Invalid configuration: datasets.{name}.revision must be a full commit sha "
            f"(40 hex characters) for reproducibility, got {entry.get('revision')!r}")
    if require_files and not entry.get("files"):
        raise ValueError(
            f"Invalid configuration: datasets.{name}.files must list the files to download "
            f"(refusing to download a whole repository)")
    return entry


def list_remote(repo: str, revision: str, wanted: Optional[Sequence[str]] = None) -> List[RemoteFile]:
    """Ask the Hub which files exist at this exact revision, with sizes."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    files: List[RemoteFile] = []
    for s in info.siblings:
        if wanted is not None and s.rfilename not in wanted:
            continue
        lfs = getattr(s, "lfs", None)
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files.append(RemoteFile(s.rfilename, int(s.size or 0), sha))
    missing = set(wanted or []) - {f.name for f in files}
    if missing:
        raise FileNotFoundError(f"Files not found in {repo}@{revision[:8]}: {sorted(missing)}")
    return files


def fetch_file(repo: str, revision: str, name: str, dest_dir: Path) -> Path:
    """Download one file (resumable) into dest_dir."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=name, repo_type="dataset",
                                revision=revision, local_dir=str(dest_dir)))


def require_free_space(path: Path, needed_bytes: int, margin: float = 1.1) -> None:
    free = shutil.disk_usage(path).free
    if free < needed_bytes * margin:
        raise OSError(
            f"Not enough free disk space at {path}: need about {needed_bytes * margin / 1e9:.1f} GB, "
            f"have {free / 1e9:.1f} GB. Run this on Colab/Drive or free up space.")


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def download_dataset(
    name: str,
    entry: Dict[str, Any],
    dest_root: Path,
    list_only: bool = False,
    lister: Callable[..., List[RemoteFile]] = list_remote,
    fetcher: Callable[..., Path] = fetch_file,
) -> Optional[Dict[str, Any]]:
    repo, revision = entry["hf_repo"], entry["revision"]
    files = lister(repo, revision, entry.get("files"))
    total = sum(f.size for f in files)

    print(f"{name}: {repo} @ {revision[:8]}")
    for f in files:
        print(f"  {f.name:24} {_gb(f.size)}")
    print(f"  {'TOTAL':24} {_gb(total)}")
    if list_only:
        return None

    dest = Path(dest_root) / name
    dest.mkdir(parents=True, exist_ok=True)

    todo = [f for f in files
            if not ((dest / f.name).is_file() and (dest / f.name).stat().st_size == f.size)]
    require_free_space(dest, sum(f.size for f in todo))

    for f in files:
        target = dest / f.name
        if f not in todo:
            print(f"  already complete: {f.name}")
            continue
        print(f"  downloading: {f.name}")
        path = fetcher(repo, revision, f.name, dest)
        actual = Path(path).stat().st_size
        if actual != f.size:
            raise RuntimeError(
                f"Size mismatch for {f.name}: expected {f.size} bytes, got {actual}. Delete it and retry.")

    record = {
        "dataset": name,
        "hf_repo": repo,
        "revision": revision,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"name": f.name, "size": f.size, "sha256": f.sha256} for f in files],
    }
    (dest / RECORD_NAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Done. Record written to {dest / RECORD_NAME}")
    return record
