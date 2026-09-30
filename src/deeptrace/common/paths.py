"""Configurable storage locations.

Colab's /content is wiped when the runtime resets, so persistent artifacts
(datasets, checkpoints, results) live under one configurable root, typically a
folder in Google Drive. Nothing here hard-codes a personal path.

Root resolution order:  explicit argument  >  $DEEPTRACE_ROOT  >  ./deeptrace_data
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

ENV_ROOT = "DEEPTRACE_ROOT"
DEFAULT_LOCAL_ROOT = "deeptrace_data"

PathLike = Union[str, os.PathLike]


def resolve_root(root: Optional[PathLike] = None) -> Path:
    if root is not None:
        chosen = Path(root)
    elif os.environ.get(ENV_ROOT):
        chosen = Path(os.environ[ENV_ROOT])
    else:
        chosen = Path(DEFAULT_LOCAL_ROOT)
    return chosen.expanduser().resolve()


@dataclass(frozen=True)
class DeepTracePaths:
    root: Path

    @property
    def datasets(self) -> Path:      # raw datasets exactly as downloaded
        return self.root / "datasets"

    @property
    def processed(self) -> Path:     # derived data, e.g. extracted face crops
        return self.root / "processed"

    @property
    def splits(self) -> Path:        # train/val/test manifests
        return self.root / "splits"

    @property
    def checkpoints(self) -> Path:   # model weights
        return self.root / "checkpoints"

    @property
    def experiments(self) -> Path:   # per-run config snapshots, logs, metrics
        return self.root / "experiments"

    @property
    def results(self) -> Path:       # final tables/figures worth keeping
        return self.root / "results"

    def all_dirs(self):
        return [self.datasets, self.processed, self.splits,
                self.checkpoints, self.experiments, self.results]

    def ensure(self) -> "DeepTracePaths":
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: Optional[PathLike] = None, create: bool = True) -> DeepTracePaths:
    paths = DeepTracePaths(resolve_root(root))
    return paths.ensure() if create else paths


# Explicit failures: a silently missing dataset or checkpoint can invalidate experiments.
def require_dataset_dir(path: PathLike) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {p}")
    return p


def require_checkpoint(path: PathLike) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {p}")
    return p
