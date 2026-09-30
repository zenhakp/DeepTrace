import pytest
import torch

from deeptrace.utils.runtime import resolve_device, set_seed


def test_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_auto_returns_valid_device():
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_invalid_device_raises():
    with pytest.raises(ValueError, match="Invalid configuration"):
        resolve_device("tpu")


def test_seed_is_reproducible():
    set_seed(123)
    a = torch.rand(3)
    set_seed(123)
    b = torch.rand(3)
    assert torch.equal(a, b)
