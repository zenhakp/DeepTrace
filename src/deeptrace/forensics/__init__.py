"""Part 2 forensics: localization, robustness, stability, video and structured evidence.

Consumes the Part 1 detector unchanged. Settings live in configs/part2_forensics.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from deeptrace.common.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "part2_forensics.yaml"


def load_forensics_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return load_config(path or DEFAULT_CONFIG, overrides)
