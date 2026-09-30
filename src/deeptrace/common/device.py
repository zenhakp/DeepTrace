"""Device selection (cpu / cuda / auto) and environment description."""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict

VALID_DEVICES = ("cpu", "cuda", "auto")


def _cuda_available() -> bool:
    """False if torch is not installed or no GPU is visible. Separate function so tests can patch it."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(requested: str = "auto") -> str:
    """Return 'cpu' or 'cuda'. Never silently downgrades an explicit 'cuda' request."""
    if requested not in VALID_DEVICES:
        raise ValueError(
            f"Invalid configuration: device must be one of {VALID_DEVICES}, got {requested!r}")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not _cuda_available():
            raise RuntimeError(
                "CUDA was requested but is not available. On Colab: Runtime > Change runtime type > GPU. "
                "Otherwise use device: cpu or auto.")
        return "cuda"
    return "cuda" if _cuda_available() else "cpu"


def describe_environment() -> Dict[str, Any]:
    """Facts worth saving next to every experiment for reproducibility."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("numpy", "torch"):
        try:
            mod = __import__(pkg)
            info[pkg] = mod.__version__
        except ImportError:
            info[pkg] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["cuda_available"] = False
        info["gpu_name"] = None
    return info
