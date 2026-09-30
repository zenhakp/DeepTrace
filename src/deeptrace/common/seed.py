"""Global seeding for reproducible experiments."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and (if installed) torch.

    deterministic=True also asks cuDNN for repeatable kernels; this is slower, and
    exact GPU repeatability still depends on the operations used.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"Invalid configuration: seed must be an integer, got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
