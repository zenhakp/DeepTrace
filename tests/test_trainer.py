import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from deeptrace.training import trainer as tr

METHODS = ("real", "inpainting", "insight", "text2img")


def _rows(tmp_path, n=8, size=32):
    rng = np.random.default_rng(0)
    rows = []
    for method in METHODS:
        for i in range(n):
            rel = f"{method}/{i}.jpg"
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path, "JPEG")
            rows.append({"image_id": f"{method}/{i}", "method": method,
                         "label": 0 if method == "real" else 1, "out_path": rel, "kept": 1})
    return rows


def _cfg(bs=4):
    return {"loading": {"image_size": 32, "batch_size": bs, "num_workers": 0, "hflip": False,
                        "degrade": {"p_blur": 0, "blur_sigma": [0.1, 1.5], "p_resize": 0,
                                    "resize_scale": [0.5, 1.0], "p_jpeg": 0, "jpeg_quality": [40, 95]}},
            "train": {"epochs": 2, "lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 1, "amp": False,
                      "fakes_per_real": 1, "ckpt_every_steps": 2, "log_every_steps": 1,
                      "pretrained": False, "drop_rate": 0.0}}


def _tiny():
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, stride=2), torch.nn.AdaptiveAvgPool2d(1),
                               torch.nn.Flatten(), torch.nn.Linear(4, 2))


def _run(tmp_path, cfg=None, model=None, **kw):
    rows = _rows(tmp_path)
    return tr.run_training(model or _tiny(), rows, rows, tmp_path, cfg or _cfg(), tmp_path / "run",
                           tmp_path / "ckpt", "cpu", log=lambda *_: None, **kw)


def test_epoch_plan_is_balanced_deterministic_and_changes_per_epoch(tmp_path):
    rows = _rows(tmp_path)
    plan = tr.epoch_plan(rows, 0, 42, 1)
    labels = [rows[i]["label"] for i in plan]
    assert labels.count(0) == labels.count(1) == 8
    per_method = [sum(rows[i]["method"] == m for i in plan) for m in METHODS[1:]]
    assert max(per_method) - min(per_method) <= 1
    assert plan == tr.epoch_plan(rows, 0, 42, 1)
    assert plan != tr.epoch_plan(rows, 1, 42, 1)


def test_lr_factor_warmup_then_cosine():
    assert abs(tr.lr_factor(0, 10, 100) - 0.1) < 1e-12
    assert abs(tr.lr_factor(10, 10, 100) - 1.0) < 1e-12
    assert abs(tr.lr_factor(55, 10, 100) - 0.5) < 1e-12
    assert abs(tr.lr_factor(100, 10, 100)) < 1e-12


def test_train_settings_validation():
    assert tr.train_settings(_cfg())["epochs"] == 2
    bad = _cfg()
    bad["train"]["epochs"] = 0
    with pytest.raises(ValueError, match="Invalid configuration"):
        tr.train_settings(bad)


def test_training_runs_and_resumes_after_interruption(tmp_path):
    r1 = _run(tmp_path, max_steps=3)
    assert r1["stopped_early"] and r1["global_step"] == 3 and (tmp_path / "ckpt" / "last.pt").is_file()
    r2 = _run(tmp_path)                                   # new model object; state comes from last.pt
    assert not r2["stopped_early"] and r2["global_step"] == 2 * math.ceil(16 / 4) and r2["epoch"] == 2
    assert (tmp_path / "ckpt" / "best.pt").is_file()
    summary = json.loads((tmp_path / "run" / "training_summary.json").read_text())
    assert summary["global_step"] == r2["global_step"]


def test_resume_rejects_changed_settings(tmp_path):
    _run(tmp_path, max_steps=2)
    with pytest.raises(ValueError, match="Cannot resume"):
        _run(tmp_path, cfg=_cfg(bs=2))


def test_non_finite_loss_fails_loudly(tmp_path):
    class NaNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x.new_full((x.shape[0], 2), float("nan")) * self.w

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        _run(tmp_path, model=NaNModel())
