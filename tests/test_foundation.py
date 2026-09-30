import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from deeptrace.common import device as device_mod
from deeptrace.common.config import load_config, save_config_snapshot
from deeptrace.common.device import describe_environment, resolve_device
from deeptrace.common.paths import (ENV_ROOT, get_paths, require_checkpoint,
                                    require_dataset_dir)
from deeptrace.common.seed import seed_everything

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "part1_default.yaml"


# ---------- paths ----------
def test_paths_explicit_root_creates_layout(tmp_path):
    p = get_paths(tmp_path / "dt")
    for d in p.all_dirs():
        assert d.is_dir()


def test_paths_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_ROOT, str(tmp_path / "from_env"))
    assert get_paths().root == (tmp_path / "from_env").resolve()


def test_paths_explicit_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_ROOT, str(tmp_path / "env"))
    assert get_paths(tmp_path / "arg").root == (tmp_path / "arg").resolve()


def test_missing_dataset_and_checkpoint_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset path not found"):
        require_dataset_dir(tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        require_checkpoint(tmp_path / "nope.pt")


# ---------- config ----------
def test_default_config_loads():
    cfg = load_config(DEFAULT_CFG)
    assert cfg["run"]["seed"] == 42 and cfg["run"]["device"] == "auto"


def test_override_applies():
    assert load_config(DEFAULT_CFG, {"run.seed": 7})["run"]["seed"] == 7


def test_invalid_device_rejected():
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(DEFAULT_CFG, {"run.device": "tpu"})


def test_invalid_seed_rejected():
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(DEFAULT_CFG, {"run.seed": "abc"})


def test_missing_config_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_snapshot_roundtrip(tmp_path):
    cfg = load_config(DEFAULT_CFG)
    target = save_config_snapshot(cfg, tmp_path / "run1")
    assert yaml.safe_load(target.read_text()) == cfg


# ---------- seed ----------
def test_seed_reproducible():
    seed_everything(123)
    a = (random.random(), float(np.random.rand()))
    seed_everything(123)
    b = (random.random(), float(np.random.rand()))
    assert a == b


def test_seed_rejects_non_int():
    with pytest.raises(ValueError):
        seed_everything("42")


# ---------- device ----------
def test_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_device_auto_without_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_device_auto_with_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_device_cuda_requested_but_missing_raises(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_device_invalid():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_describe_environment_keys():
    info = describe_environment()
    assert {"python", "platform", "numpy", "torch", "cuda_available"} <= set(info)
