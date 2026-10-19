import numpy as np
import pytest
from PIL import Image

from deeptrace.data import external as ex
from deeptrace.data.preprocess import read_faces_csv
from deeptrace.evaluation.analysis import binned_rates

PARAMS = {"output_size": 64, "jpeg_quality": 90, "crop_margin": 1.3}
FACE = (40.0, 40.0, 60.0, 60.0, 0.9)


def _folder(tmp_path, name, n, size=(160, 120)):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    d = tmp_path / name
    d.mkdir(parents=True)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(d / f"{i}.jpg", "JPEG")
    return d


def _items(tmp_path, n=4):
    import itertools
    return itertools.chain(ex.folder_items(_folder(tmp_path, "r", n), 0, "real"),
                           ex.folder_items(_folder(tmp_path, "f", n), 1, "stylegan"))


def test_folder_run_creates_crops_and_manifest(tmp_path):
    n = ex.run_external(_items(tmp_path), lambda rgb: FACE, PARAMS, tmp_path / "out", tmp_path / "m.csv",
                        min_crop_side=32, progress_every=0)
    rows = read_faces_csv(tmp_path / "m.csv")
    assert n == len(rows) == 8 and all(r["kept"] == 1 and r["split"] == "test" for r in rows)
    assert all((tmp_path / "out" / r["out_path"]).is_file() for r in rows)
    assert {r["image_id"].split("/")[0] for r in rows} == {"real", "stylegan"}
    assert sorted({r["label"] for r in rows}) == [0, 1]
    assert ex.summarize_external(rows)["kept_by_class"] == {"real": 4, "manipulated": 4}


def test_min_crop_side_filter_keeps_face_stats_but_drops_image(tmp_path):
    ex.run_external(_items(tmp_path), lambda rgb: (10.0, 10.0, 10.0, 10.0, 0.9), PARAMS, tmp_path / "out",
                    tmp_path / "m.csv", min_crop_side=32, progress_every=0)
    rows = read_faces_csv(tmp_path / "m.csv")
    assert all(r["face_found"] == 1 and r["kept"] == 0 for r in rows)
    assert not list((tmp_path / "out").rglob("*.jpg"))


def test_sampling_is_deterministic_and_respects_rate(tmp_path):
    def run(sub):
        items = ex.folder_items(_folder(tmp_path / sub, "r", 200), 0, "real")
        ex.run_external(items, lambda rgb: FACE, PARAMS, tmp_path / sub / "out", tmp_path / sub / "m.csv",
                        sample_rate=0.5, min_crop_side=32, progress_every=0)
        return {r["image_id"] for r in read_faces_csv(tmp_path / sub / "m.csv")}
    a, b = run("a"), run("b")
    assert a == b and 70 < len(a) < 130


def test_resume_skips_images_already_done(tmp_path):
    kw = dict(min_crop_side=32, progress_every=0)
    folders = (_folder(tmp_path, "r", 4), _folder(tmp_path, "f", 4))

    def items():
        import itertools
        return itertools.chain(ex.folder_items(folders[0], 0, "real"), ex.folder_items(folders[1], 1, "stylegan"))

    first = ex.run_external(items(), lambda rgb: FACE, PARAMS, tmp_path / "out", tmp_path / "m.csv", limit=3, **kw)
    second = ex.run_external(items(), lambda rgb: FACE, PARAMS, tmp_path / "out", tmp_path / "m.csv", **kw)
    third = ex.run_external(items(), lambda rgb: FACE, PARAMS, tmp_path / "out", tmp_path / "m.csv", **kw)
    ids = [r["image_id"] for r in read_faces_csv(tmp_path / "m.csv")]
    assert (first, second, third) == (3, 5, 0) and len(ids) == len(set(ids)) == 8


def test_slug():
    assert ex.slug("GPT Image 1") == "gpt-image-1" and ex.slug("///") == "unknown"


def test_binned_rates_detects_dependence_on_a_feature():
    rows = {f"real/{i}": {"q": float(i)} for i in range(100)}
    preds = [{"image_id": f"real/{i}", "label": "0", "fake_prob": i / 100} for i in range(100)]
    out = binned_rates(rows, preds, "q", 0, n_bins=5)
    assert [b["n"] for b in out] == [20] * 5
    assert out[0]["share_predicted_manipulated"] == 0.0 and out[-1]["share_predicted_manipulated"] == 1.0
    with pytest.raises(ValueError, match="not found in the manifest"):
        binned_rates({}, preds, "q", 0)


def test_split_label_is_written(tmp_path):
    ex.run_external(_items(tmp_path), lambda rgb: FACE, PARAMS, tmp_path / "out", tmp_path / "m.csv",
                    min_crop_side=32, progress_every=0, split="val")
    assert {r["split"] for r in read_faces_csv(tmp_path / "m.csv")} == {"val"}


def test_choose_shards_is_spread_deterministic_and_last_is_val():
    names = [f"s{i:03d}" for i in range(100)]
    plan = ex.choose_shards(names, 3, 1)
    assert plan == ex.choose_shards(list(reversed(names)), 3, 1)
    assert [s for _, s in plan] == ["train", "train", "train", "val"]
    idx = [names.index(n) for n, _ in plan]
    assert idx == sorted(set(idx)) and idx[0] < 20 and idx[-1] > 80
    with pytest.raises(ValueError, match="only"):
        ex.choose_shards(names[:2], 3, 1)
