import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from deeptrace.data import faces as fc
from deeptrace.data import manifest as mf
from deeptrace.data import preprocess as pp

PARAMS = {"output_size": 64, "jpeg_quality": 90, "crop_margin": 1.3, "gray_chroma_eps": 2.0}
FACE = (10.0, 10.0, 30.0, 30.0, 0.9)
FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _jpg(size, mode):
    buf = io.BytesIO()
    Image.new(mode, size, 128 if mode == "L" else (200, 40, 40)).save(buf, format="JPEG")
    return buf.getvalue()


def _rows(tmp_path, n=10, real_mode="L"):
    rows = []
    for top, size, mode in (("real", (60, 80), real_mode), ("f1", (128, 128), "RGB"),
                            ("f2", (128, 128), "RGB"), ("f3", (128, 128), "RGB")):
        p = tmp_path / f"{top}.zip"
        with zipfile.ZipFile(p, "w") as zf:
            for i in range(n):
                zf.writestr(f"{top}/0{i % 2}/{i}_1980-01-01_2010.jpg", _jpg(size, mode))
        rows += mf.scan_zip(p, top, 0 if top == "real" else 1, progress_every=0)
    assign = mf.assign_splits([r["group_id"] for r in rows], FRACS, 42)
    for r in rows:
        r["split"] = assign[r["group_id"]]
    return rows


@pytest.mark.parametrize("face,size", [((10, 10, 30, 30, .9), (200, 100)),
                                       ((190, 5, 30, 30, .9), (200, 100)),
                                       ((0, 0, 400, 400, .9), (200, 100))])
def test_square_crop_box_inside_and_square(face, size):
    l, t, r, b = fc.square_crop_box(face, size[0], size[1], 1.3)
    assert r - l == b - t > 0 and l >= 0 and t >= 0 and r <= size[0] and b <= size[1]


def test_gray_source_detection():
    gray = np.full((8, 8, 3), 100, np.uint8)
    colour = np.zeros((8, 8, 3), np.uint8)
    colour[..., 0] = 200
    assert fc.is_gray_source("RGB", gray, 2.0) is True
    assert fc.is_gray_source("RGB", colour, 2.0) is False
    assert fc.is_gray_source("L", colour, 2.0) is True


def test_auc_and_threshold_separable():
    labels, scores = [0, 0, 0, 1, 1, 1], [1, 2, 3, 10, 11, 12]
    assert pp.auc(labels, scores) == 1.0
    thr, d, ba = pp.best_threshold(scores, labels)
    assert ba == 1.0 and d == 1 and pp.eval_threshold(scores, labels, thr, d) == 1.0


def test_auc_uninformative_and_inverted():
    assert pp.auc([0, 1, 0, 1], [5, 5, 5, 5]) == 0.5
    assert pp.auc([0, 0, 1, 1], [3, 4, 1, 2]) == 0.0


def test_process_group_matches_colour_mode_and_saves(tmp_path):
    rows = _rows(tmp_path)
    gid = rows[0]["group_id"]
    group = [r for r in rows if r["group_id"] == gid]
    zips = {n: zipfile.ZipFile(tmp_path / n) for n in {r["zip"] for r in group}}
    out = pp.process_group(group, zips, lambda rgb: FACE, PARAMS, tmp_path / "out")
    assert len(out) == 4 and all(r["kept"] == 1 and r["gray_matched"] == 1 for r in out)
    for r in out:
        im = Image.open(tmp_path / "out" / r["out_path"])
        assert im.size == (64, 64)
        assert fc.mean_chroma(np.asarray(im.convert("RGB"))) < 2.0     # coloured fakes were matched to gray


def test_group_dropped_if_any_face_missing(tmp_path):
    rows = _rows(tmp_path)
    gid = rows[0]["group_id"]
    group = [r for r in rows if r["group_id"] == gid]
    zips = {n: zipfile.ZipFile(tmp_path / n) for n in {r["zip"] for r in group}}
    detector = lambda rgb: None if rgb.shape[0] == 128 else FACE      # fails on the fakes only
    out = pp.process_group(group, zips, detector, PARAMS, tmp_path / "out")
    assert all(r["kept"] == 0 for r in out)
    assert not list((tmp_path / "out").rglob("*.jpg"))


def test_subset_groups_deterministic():
    ids = [f"g{i}" for i in range(50)]
    a = pp.subset_groups(ids, 10, 42)
    assert a == pp.subset_groups(list(reversed(ids)), 10, 42) and len(a) == 10
    assert a != pp.subset_groups(ids, 10, 7)


def test_params_validation():
    good = {"preprocess": {"face_model_url": "https://x/y.onnx", "face_model_file": "y.onnx",
                           "face_score_threshold": 0.6, "crop_margin": 1.3, "output_size": 380,
                           "jpeg_quality": 95, "gray_chroma_eps": 2.0}}
    assert pp.params_from_cfg(good)["output_size"] == 380
    with pytest.raises(ValueError, match="Invalid configuration"):
        pp.params_from_cfg({"preprocess": dict(good["preprocess"], crop_margin=0.5)})


def test_run_preprocess_resumes(tmp_path):
    rows = _rows(tmp_path)
    csv_path, det = tmp_path / "faces.csv", (lambda rgb: FACE)
    assert pp.run_preprocess(rows, tmp_path, tmp_path / "out", csv_path, det, PARAMS,
                             limit_groups=4, seed=42, progress_every=0) == 4
    assert pp.run_preprocess(rows, tmp_path, tmp_path / "out", csv_path, det, PARAMS,
                             seed=42, progress_every=0) == 6
    assert pp.run_preprocess(rows, tmp_path, tmp_path / "out", csv_path, det, PARAMS,
                             seed=42, progress_every=0) == 0
    got = pp.read_faces_csv(csv_path)
    assert len(got) == 40 and all(r["kept"] == 1 for r in got)
    assert set(pp.shortcut_audit(got)) == set(pp.AUDIT_FEATURES)
