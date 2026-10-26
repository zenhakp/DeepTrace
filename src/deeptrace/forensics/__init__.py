"""Part 2 forensics: localization, robustness, stability, video and structured evidence.

Consumes the Part 1 detector unchanged. Settings live in configs/part2_forensics.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from deeptrace.common.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "part2_forensics.yaml"
DEFAULT_OUT_DIR = Path.home() / "Downloads" / "deeptrace_outputs"


def load_forensics_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return load_config(path or DEFAULT_CONFIG, overrides)


def output_dir(path=None, repo_root: Path = REPO_ROOT) -> Path:
    """Create and return the output folder; refuse folders inside the repository (outputs must not be committed)."""
    out = Path(path).expanduser().resolve() if path else DEFAULT_OUT_DIR.resolve()
    root = Path(repo_root).resolve()
    if out == root or root in out.parents:
        raise ValueError(f"Output folder is inside the repository ({out}); choose a folder outside {root}")
    out.mkdir(parents=True, exist_ok=True)
    return out
