"""Runtime helpers: device selection and reproducible seeding."""
import random

import numpy as np
import torch

_VALID_DEVICES = {"auto", "cpu", "cuda"}


def resolve_device(requested: str = "auto") -> torch.device:
    """Map 'auto' | 'cpu' | 'cuda' to a torch.device, failing loudly on bad input."""
    requested = str(requested).lower()
    if requested not in _VALID_DEVICES:
        raise ValueError(
            f"Invalid configuration: device must be one of {sorted(_VALID_DEVICES)}, got '{requested}'"
        )
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use device='cpu' or 'auto'.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
