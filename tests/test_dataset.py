import csv
import random

import numpy as np
import pytest
from PIL import Image

from deeptrace.data import dataset as ds
from deeptrace.data import preprocess as pp

DEG0 = {"p_blur": 0, "blur_sigma": [0.1, 1.5], "p_resize": 0, "resize_scale": [0.5, 1.0],
        "p_jpeg": 0, "jpeg_quality": [40, 95]}
DEG1 = dict(DEG0, p_blur=1, p_resize=1, p_jpeg=1)


def _crops(tmp_path, n=4, size=48):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rel = f"real/00/{i}.jpg"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path, "JPEG", quality=95)
        rows.append({"image_id": f"real/00/{i}", "label": i % 2, "out_path": rel})
    return rows


def test_item_shape_and_dtype(tmp_path):
    d = ds.FaceCropDataset(_crops(tmp_path), tmp_path, image_size=32)
    x, y, iid = d[1]
    assert tuple(x.shape) == (3, 32, 32) and str(x.dtype) == "torch.float32"
    assert y == 1 and iid == "real/00/1"


def test_missing_crop_fails_loudly(tmp_path):
    rows = _crops(tmp_path) + [{"image_id": "x", "label": 0, "out_path": "real/00/nope.jpg"}]
    with pytest.raises(FileNotFoundError, match="Face crop not found"):
        ds.FaceCropDataset(rows, tmp_path)


def test_degrade_is_identity_when_probabilities_are_zero():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 40, 3), dtype=np.uint8))
    out = ds.degrade(img, DEG0, random.Random(0))
    assert np.array_equal(np.asarray(img), np.asarray(out))


def test_degrade_changes_pixels_but_keeps_size():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, 40, 3), dtype=np.uint8))
    out = ds.degrade(img, DEG1, random.Random(0))
    assert out.size == img.size and not np.array_equal(np.asarray(img), np.asarray(out))


def test_load_split_rows_keeps_only_kept_rows_of_the_split(tmp_path):
    path = tmp_path / "faces.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pp.FACE_FIELDS)
        w.writeheader()
        for i, (split, kept) in enumerate([("train", 1), ("train", 0), ("val", 1)]):
            w.writerow({"image_id": f"i{i}", "group_id": f"g{i}", "method": "real", "label": 0,
                        "split": split, "kept": kept, "out_path": f"real/g{i}.jpg"})
    rows = ds.load_split_rows(path, "train")
    assert [r["image_id"] for r in rows] == ["i0"]
    assert ds.class_counts(rows) == {0: 1, 1: 0}


def test_loading_config_validation():
    good = {"loading": {"image_size": 380, "batch_size": 16, "num_workers": 0, "degrade": DEG0}}
    assert ds.loading_from_cfg(good)["image_size"] == 380
    bad = {"loading": dict(good["loading"], degrade=dict(DEG0, p_blur=2))}
    with pytest.raises(ValueError, match="Invalid configuration"):
        ds.loading_from_cfg(bad)
