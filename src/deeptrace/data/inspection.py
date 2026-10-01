"""Read-only inspection of dataset zip files. Nothing is extracted to disk.

Reports what is actually inside a zip so that manifest and split code is built on
measured facts, not assumptions about file layout or naming.
"""
from __future__ import annotations

import io
import random
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Sequence

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# IMDB-WIKI style stems (shown only to measure how many names follow them; not assumed)
WIKI_STEM = re.compile(r"^(\d+)_(\d{4}-\d{1,2}-\d{1,2})_(\d{4})$")
IMDB_STEM = re.compile(r"^(nm\d+)_rm\d+_(\d{4}-\d{1,2}-\d{1,2})_(\d{4})$")


def _is_image(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMG_EXT and "__MACOSX" not in name


def _require_zip(zip_path) -> Path:
    p = Path(zip_path)
    if not p.is_file():
        raise FileNotFoundError(f"Zip file not found: {p}")
    return p


def list_images(zip_path) -> List[str]:
    """All image entry names inside the zip, in archive order."""
    with zipfile.ZipFile(_require_zip(zip_path)) as zf:
        return [n for n in zf.namelist() if not n.endswith("/") and _is_image(n)]


def summarize_zip(zip_path, sample_n: int = 200, seed: int = 42) -> Dict[str, Any]:
    p = _require_zip(zip_path)
    with zipfile.ZipFile(p) as zf:
        entries = [n for n in zf.namelist() if not n.endswith("/")]
        images = [n for n in entries if _is_image(n)]

        ext = Counter(PurePosixPath(n).suffix.lower() or "<none>" for n in entries)
        top = Counter(PurePosixPath(n).parts[0] if len(PurePosixPath(n).parts) > 1 else "<root>"
                      for n in images)
        depth = Counter(len(PurePosixPath(n).parts) for n in images)
        stems = [PurePosixPath(n).stem for n in images]

        # candidate identity token = text before the first underscore in the file name
        tokens = Counter(s.split("_")[0] for s in stems)
        per_token = sorted(tokens.values())

        rng = random.Random(seed)
        sample = rng.sample(images, min(sample_n, len(images)))
        sizes: Counter = Counter()
        modes: Counter = Counter()
        unreadable: List[Dict[str, str]] = []
        for name in sample:
            try:
                with zf.open(name) as fh:
                    img = Image.open(io.BytesIO(fh.read()))
                    img.load()
                sizes[f"{img.width}x{img.height}"] += 1
                modes[img.mode] += 1
            except Exception as exc:  # report, never hide: unreadable data can invalidate experiments
                unreadable.append({"name": name, "error": type(exc).__name__})

    return {
        "zip": p.name,
        "zip_bytes": p.stat().st_size,
        "n_entries": len(entries),
        "n_images": len(images),
        "extensions": dict(ext),
        "top_level_dirs": dict(top.most_common(20)),
        "n_top_level_dirs": len(top),
        "path_depths": dict(depth),
        "example_names": sorted(images)[:5],
        "stems_matching_wiki_pattern": sum(bool(WIKI_STEM.match(s)) for s in stems),
        "stems_matching_imdb_pattern": sum(bool(IMDB_STEM.match(s)) for s in stems),
        "candidate_identity_tokens": len(tokens),
        "images_per_token": {
            "min": per_token[0] if per_token else None,
            "median": statistics.median(per_token) if per_token else None,
            "max": per_token[-1] if per_token else None,
        },
        "sample": {
            "requested": sample_n,
            "checked": len(sample),
            "unreadable": len(unreadable),
            "unreadable_examples": unreadable[:5],
            "sizes_top5": dict(sizes.most_common(5)),
            "n_distinct_sizes": len(sizes),
            "modes": dict(modes),
        },
    }


def _strip_top(name: str) -> str:
    parts = PurePosixPath(name).parts
    return "/".join(parts[1:]) if len(parts) > 1 else name


def name_overlap(reference: Sequence[str], other: Sequence[str]) -> Dict[str, int]:
    """How many names coincide under three notions of 'same file'. Reveals pairing structure."""
    ref, oth = set(reference), set(other)
    return {
        "n_reference": len(ref),
        "n_other": len(oth),
        "same_full_path": len(ref & oth),
        "same_path_without_top_dir": len({_strip_top(n) for n in ref} & {_strip_top(n) for n in oth}),
        "same_basename": len({PurePosixPath(n).name for n in ref} & {PurePosixPath(n).name for n in oth}),
        "same_stem": len({PurePosixPath(n).stem for n in ref} & {PurePosixPath(n).stem for n in oth}),
    }
