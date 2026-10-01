import collections
import json
from pathlib import Path

import pytest

from deeptrace.common.config import load_config
from deeptrace.data import download as dl

SHA = "a" * 40
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "part1_default.yaml"


def _entry(files=("a.zip", "b.zip")):
    return {"hf_repo": "x/y", "revision": SHA, "files": list(files)}


def _lister(sizes):
    return lambda repo, revision, wanted=None: [dl.RemoteFile(n, s) for n, s in sizes.items()]


def _fetcher(sizes):
    def fetch(repo, revision, name, dest_dir):
        p = Path(dest_dir) / name
        p.write_bytes(b"x" * sizes[name])
        return p
    return fetch


def test_entry_ok():
    assert dl.dataset_entry({"datasets": {"d": _entry()}}, "d")["hf_repo"] == "x/y"


def test_unknown_dataset():
    with pytest.raises(ValueError, match="Invalid configuration"):
        dl.dataset_entry({"datasets": {}}, "nope")


def test_unpinned_revision_rejected():
    bad = _entry()
    bad["revision"] = "main"
    with pytest.raises(ValueError, match="full commit sha"):
        dl.dataset_entry({"datasets": {"d": bad}}, "d")


def test_empty_files_rejected():
    with pytest.raises(ValueError, match="refusing to download"):
        dl.dataset_entry({"datasets": {"d": _entry(())}}, "d")


def test_default_config_datasets_are_pinned():
    cfg = load_config(DEFAULT_CFG)
    dl.dataset_entry(cfg, "deepfakeface")
    dl.dataset_entry(cfg, "openfake", require_files=False)


def test_free_space_check(tmp_path, monkeypatch):
    Usage = collections.namedtuple("Usage", "total used free")
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: Usage(100, 90, 10))
    with pytest.raises(OSError, match="Not enough free disk space"):
        dl.require_free_space(tmp_path, 1000)


def test_download_writes_record(tmp_path):
    sizes = {"a.zip": 5, "b.zip": 7}
    dl.download_dataset("d", _entry(), tmp_path, lister=_lister(sizes), fetcher=_fetcher(sizes))
    rec = json.loads((tmp_path / "d" / dl.RECORD_NAME).read_text())
    assert rec["revision"] == SHA
    assert {f["name"] for f in rec["files"]} == {"a.zip", "b.zip"}


def test_size_mismatch_raises(tmp_path):
    with pytest.raises(RuntimeError, match="Size mismatch"):
        dl.download_dataset("d", _entry(("a.zip",)), tmp_path,
                            lister=_lister({"a.zip": 5}), fetcher=_fetcher({"a.zip": 3}))


def test_complete_file_is_skipped(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.zip").write_bytes(b"x" * 5)

    def must_not_fetch(*args, **kwargs):
        raise AssertionError("should not fetch an already complete file")

    dl.download_dataset("d", _entry(("a.zip",)), tmp_path,
                        lister=_lister({"a.zip": 5}), fetcher=must_not_fetch)


def test_list_only_downloads_nothing(tmp_path):
    dl.download_dataset("d", _entry(), tmp_path, list_only=True,
                        lister=_lister({"a.zip": 5, "b.zip": 7}), fetcher=None)
    assert not (tmp_path / "d").exists()
