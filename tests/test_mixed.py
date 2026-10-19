import csv
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from deeptrace.common.paths import get_paths
from deeptrace.data import preprocess as pp
from deeptrace.data.mixed import load_mixed_rows
from deeptrace.data.preprocess import faces_locations
from deeptrace.evaluation import families as fam
from deeptrace.training import trainer as tr


def _src_rows(extra=()):
    rows = []
    for src, n_real, n_fake in (("A", 4, 12), ("B", 2, 3)) + tuple(extra):
        rows += [{"image_id": f"{src}/real/{i}", "label": 0, "source": src} for i in range(n_real)]
        rows += [{"image_id": f"{src}/gen/{i}", "label": 1, "source": src} for i in range(n_fake)]
    return rows


def test_epoch_plan_by_source_balances_each_source_and_is_deterministic():
    rows = _src_rows()
    plan = tr.epoch_plan_by_source(rows, 0, 42, 1)
    counts = Counter((rows[i]["source"], rows[i]["label"]) for i in plan)
    assert counts == {("A", 0): 4, ("A", 1): 4, ("B", 0): 2, ("B", 1): 2}
    assert plan == tr.epoch_plan_by_source(rows, 0, 42, 1)
    assert plan != tr.epoch_plan_by_source(rows, 1, 42, 1)


def test_epoch_plan_by_source_requires_both_classes_in_every_source():
    rows = _src_rows(extra=(("C", 3, 0),))
    with pytest.raises(ValueError, match="needs both"):
        tr.epoch_plan_by_source(rows, 0, 42, 1)


def test_planner_uses_source_balancing_only_for_several_sources():
    multi = _src_rows()
    single = [r for r in multi if r["source"] == "A"]
    nosrc = [{k: v for k, v in r.items() if k != "source"} for r in multi]
    assert tr._planner(multi) is tr.epoch_plan_by_source
    assert tr._planner(single) is tr.epoch_plan and tr._planner(nosrc) is tr.epoch_plan


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pp.FACE_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_load_mixed_rows_prefixes_paths_and_tags_sources(tmp_path):
    paths = get_paths(tmp_path)
    params = {"output_size": 380, "crop_margin": 1.3, "jpeg_quality": 95}
    for ds in ("A", "B"):
        _, csv_path = faces_locations(paths, ds, params)
        base = {"group_id": "g", "method": "gen", "split": "train", "kept": 1}
        _write_manifest(csv_path, [
            dict(base, image_id=f"real/{ds}1", label=0, method="real", out_path=f"real/{ds}1.jpg"),
            dict(base, image_id=f"gen/{ds}2", label=1, out_path=f"gen/{ds}2.jpg"),
            dict(base, image_id=f"gen/{ds}3", label=1, split="val", out_path=f"gen/{ds}3.jpg"),
            dict(base, image_id=f"gen/{ds}4", label=1, kept=0, out_path=f"gen/{ds}4.jpg"),
        ])
    rows = load_mixed_rows(paths, params, ["A", "B"], "train")
    assert len(rows) == 4 and {r["source"] for r in rows} == {"A", "B"}
    assert all(r["out_path"].startswith(f"{r['source']}/faces_s380_m1.3_q95/") for r in rows)


def test_family_helpers_and_report():
    assert fam.family_of("flux-2-klein-9b") == "flux" and fam.family_of("illustrious") == "illustrious"
    seen = fam.seen_families(["flux-1-dev", "sd-3-5"])
    assert seen == {"flux", "sd"}
    recs = [{"image_id": f"real/r{i}", "label": 0, "fake_prob": p} for i, p in enumerate([0.1, 0.2, 0.3, 0.4])]
    recs += [{"image_id": f"flux-2-klein-9b/a{i}", "label": 1, "fake_prob": p} for i, p in enumerate([0.9, 0.8])]
    recs += [{"image_id": f"z-image-turbo/b{i}", "label": 1, "fake_prob": p} for i, p in enumerate([0.2, 0.6])]
    rep = fam.family_report(recs, seen)
    assert rep["per_generator"]["flux-2-klein-9b"] == {"n": 2, "recall": 1.0, "family": "flux",
                                                       "family_seen_in_training": True}
    assert rep["per_generator"]["z-image-turbo"]["recall"] == 0.5
    assert rep["by_group"]["seen_family"]["roc_auc_vs_all_real"] == 1.0
    assert abs(rep["by_group"]["unseen_family"]["roc_auc_vs_all_real"] - 0.6875) < 1e-9


def _rows(tmp_path, n=8, size=32):
    rng = np.random.default_rng(0)
    rows = []
    for method in ("real", "inpainting", "insight", "text2img"):
        for i in range(n):
            rel = f"{method}/{i}.jpg"
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path, "JPEG")
            rows.append({"image_id": f"{method}/{i}", "method": method,
                         "label": 0 if method == "real" else 1, "out_path": rel, "kept": 1})
    return rows


def _cfg():
    return {"loading": {"image_size": 32, "batch_size": 4, "num_workers": 0, "hflip": False,
                        "degrade": {"p_blur": 0, "blur_sigma": [0.1, 1.5], "p_resize": 0,
                                    "resize_scale": [0.5, 1.0], "p_jpeg": 0, "jpeg_quality": [40, 95]}},
            "train": {"epochs": 2, "lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 1, "amp": False,
                      "fakes_per_real": 1, "ckpt_every_steps": 2, "log_every_steps": 1,
                      "pretrained": False, "drop_rate": 0.0}}


def test_training_on_two_sources_runs_and_reports_per_source_validation(tmp_path):
    rows = _rows(tmp_path)
    for i, r in enumerate(rows):
        r["source"] = "A" if i % 2 == 0 else "B"
    tiny = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, stride=2), torch.nn.AdaptiveAvgPool2d(1),
                               torch.nn.Flatten(), torch.nn.Linear(4, 2))
    res = tr.run_training(tiny, rows, rows, tmp_path, _cfg(), tmp_path / "run", tmp_path / "ckpt", "cpu",
                          log=lambda *_: None)
    assert res["global_step"] == 8 and res["epoch"] == 2
    meta = torch.load(tmp_path / "ckpt" / "best.pt", weights_only=True)["meta"]
    assert set(meta["val_metrics"]["per_source"]) == {"A", "B"}
