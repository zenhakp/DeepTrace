import numpy as np
import pytest

from deeptrace.data import matching as mt
from deeptrace.data.preprocess import auc


def _rows(real_range=(2, 10), fake_range=(1, 4)):
    rng = np.random.default_rng(0)
    rows = [{"image_id": f"r{i}", "label": 0, "kept": 1, "split": "test",
             "q_lum_mean": float(rng.uniform(*real_range))} for i in range(400)]
    rows += [{"image_id": f"f{i}", "label": 1, "kept": 1, "split": "test",
              "q_lum_mean": float(rng.uniform(*fake_range))} for i in range(600)]
    return rows


def test_matching_balances_classes_and_removes_the_feature_signal():
    rows = _rows()
    raw = auc([r["label"] for r in rows], [r["q_lum_mean"] for r in rows])
    assert raw < 0.3                                      # strongly separable before matching
    ids = set(mt.matched_image_ids(rows, "q_lum_mean", n_bins=10, seed=1))
    sub = [r for r in rows if r["image_id"] in ids]
    assert sum(r["label"] == 0 for r in sub) == sum(r["label"] == 1 for r in sub) > 0
    matched = auc([r["label"] for r in sub], [r["q_lum_mean"] for r in sub])
    assert 0.4 < matched < 0.6


def test_no_overlap_raises():
    with pytest.raises(ValueError, match="no overlapping range"):
        mt.matched_image_ids(_rows(real_range=(5, 6), fake_range=(1, 2)), "q_lum_mean")


def test_matching_is_deterministic_and_seeded():
    rows = _rows()
    a = mt.matched_image_ids(rows, "q_lum_mean", seed=1)
    assert a == mt.matched_image_ids(list(reversed(rows)), "q_lum_mean", seed=1)
    assert a != mt.matched_image_ids(rows, "q_lum_mean", seed=2)
