"""YAML config loading with explicit validation and per-run snapshots."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .device import VALID_DEVICES


def _set_dotted(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Invalid configuration: cannot override '{dotted_key}'")
    node[parts[-1]] = value


def validate_config(cfg: Dict[str, Any]) -> None:
    run = cfg.get("run")
    if not isinstance(run, dict):
        raise ValueError("Invalid configuration: missing 'run' section")
    seed = run.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"Invalid configuration: run.seed must be an integer, got {seed!r}")
    device = run.get("device")
    if device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid configuration: run.device must be one of {VALID_DEVICES}, got {device!r}")
    if not isinstance(run.get("deterministic", False), bool):
        raise ValueError("Invalid configuration: run.deterministic must be true or false")


def load_config(path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML, apply dotted-key overrides (e.g. {"run.seed": 7}), validate."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid configuration: {p} must contain a mapping at top level")
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        _set_dotted(cfg, key, value)
    validate_config(cfg)
    return cfg


def save_config_snapshot(cfg: Dict[str, Any], out_dir) -> Path:
    """Write the exact config used for a run, so the run can be reproduced."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "config.yaml"
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return target
