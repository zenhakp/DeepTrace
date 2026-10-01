import io
import zipfile

import pytest
from PIL import Image

from deeptrace.data import manifest as mf

FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _jpg(size, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size, 128 if mode == "L" else (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _zip(path, top, n, size, mode="RGB", skip=()):
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            if i in skip:
                continue
            zf.writestr(f"{top}/0{i % 2}/{i}_1980-01-01_2010.jpg", _jpg(size, mode))
    return path


def _records(tmp_path, n=10, skip_fake=()):
    recs = mf.scan_zip(_zip(tmp_path / "real.zip", "real", n, (40, 30)), "real", 0, progress_every=0)
    recs += mf.scan_zip(_zip(tmp_path / "f1.zip", "f1", n, (512, 512), skip=skip_fake), "f1", 1, progress_every=0)
    return recs


def test_group_id_strips_top_dir_and_extension():
    assert mf.group_id_of("insight/00/12_1980-01-01_2010.jpg") == "00/12_1980-01-01_2010"


def test_scan_and_group_check(tmp_path):
    recs = _records(tmp_path)
    assert len(recs) == 20 and {r["label"] for r in recs} == {0, 1}
    assert mf.check_groups(recs, ["real", "f1"]) == 10


def test_missing_fake_member_fails_loudly(tmp_path):
    recs = _records(tmp_path, skip_fake=(3,))
    with pytest.raises(ValueError, match="Invalid group structure"):
        mf.check_groups(recs, ["real", "f1"])


def test_split_deterministic_and_seed_sensitive():
    ids = [f"g{i}" for i in range(100)]
    a = mf.assign_splits(ids, FRACS, 42)
    assert a == mf.assign_splits(list(reversed(ids)), FRACS, 42)   # independent of input order
    assert a != mf.assign_splits(ids, FRACS, 7)


def test_split_sizes():
    out = mf.assign_splits([f"g{i}" for i in range(10)], FRACS, 42)
    counts = {s: list(out.values()).count(s) for s in mf.SPLITS}
    assert counts == {"train": 8, "val": 1, "test": 1}


def test_leakage_check(tmp_path):
    recs = _records(tmp_path)
    assign = mf.assign_splits([r["group_id"] for r in recs], FRACS, 42)
    for r in recs:
        r["split"] = assign[r["group_id"]]
    mf.verify_no_leakage(recs)
    other = "val" if recs[0]["split"] != "val" else "test"
    recs[0]["split"] = other                      # move one image away from its group
    with pytest.raises(ValueError, match="Leakage detected"):
        mf.verify_no_leakage(recs)


def test_invalid_fractions_rejected():
    bad = {"split": {"seed": 1, "fractions": {"train": 0.9, "val": 0.2, "test": 0.1}}}
    with pytest.raises(ValueError, match="Invalid configuration"):
        mf.split_settings(bad)


def test_shortcut_report_size_rule(tmp_path):
    recs = _records(tmp_path)
    assign = mf.assign_splits([r["group_id"] for r in recs], FRACS, 42)
    for r in recs:
        r["split"] = assign[r["group_id"]]
    rep = mf.shortcut_report(recs)
    assert rep["class_profile"]["manipulated"]["share_512x512"] == 1.0
    assert rep["class_profile"]["real"]["share_512x512"] == 0.0
    assert rep["rule_on_split"]["train"]["accuracy"] == 1.0


def test_unreadable_header_is_flagged(tmp_path):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("real/00/1_x_1.jpg", b"not an image")
    recs = mf.scan_zip(p, "real", 0, progress_every=0)
    assert recs[0]["header_ok"] == 0
    assert mf.shortcut_report([dict(recs[0], split="train")])["header_failures"] == 1


def test_manifest_written_with_hash(tmp_path):
    recs = _records(tmp_path)
    assign = mf.assign_splits([r["group_id"] for r in recs], FRACS, 42)
    for r in recs:
        r["split"] = assign[r["group_id"]]
    sha = mf.write_manifest(recs, tmp_path / "m.csv")
    assert len(sha) == 64
    assert len((tmp_path / "m.csv").read_text().strip().splitlines()) == 21   # header + 20 rows
