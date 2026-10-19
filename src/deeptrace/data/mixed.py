"""Combine several face-crop datasets into one training pool. Every row gets a 'source' and an out_path that is
relative to the shared processed/ folder, so a single crops directory serves all sources."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .dataset import load_split_rows
from .preprocess import faces_locations


def load_mixed_rows(paths, params: Dict[str, Any], datasets: Sequence[str], split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        crops_dir, faces_csv = faces_locations(paths, ds, params)
        rel = crops_dir.relative_to(paths.processed).as_posix()
        for r in load_split_rows(faces_csv, split):
            r = dict(r)
            r["source"] = ds
            r["out_path"] = f"{rel}/{r['out_path']}"
            rows.append(r)
    return rows
