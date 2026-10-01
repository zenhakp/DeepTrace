import io
import zipfile

import pytest
from PIL import Image

from deeptrace.data import inspection as ins


def _jpg(size=(8, 6)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_counts_and_layout(tmp_path):
    z = _zip(tmp_path / "a.zip", {
        "wiki/00/1_1980-01-01_2010.jpg": _jpg(),
        "wiki/00/2_1981-02-02_2011.jpg": _jpg(),
        "wiki/01/3_1982-03-03_2012.jpg": _jpg(),
        "readme.txt": b"hi",
    })
    s = ins.summarize_zip(z, sample_n=10)
    assert s["n_images"] == 3 and s["n_entries"] == 4
    assert s["extensions"][".jpg"] == 3 and s["extensions"][".txt"] == 1
    assert s["sample"]["unreadable"] == 0
    assert s["sample"]["sizes_top5"] == {"8x6": 3}


def test_pattern_and_identity_tokens(tmp_path):
    z = _zip(tmp_path / "a.zip", {
        "x/10_1980-01-01_2010.jpg": _jpg(),
        "x/10_1980-01-01_2012.jpg": _jpg(),
        "x/20_1990-05-05_2015.jpg": _jpg(),
        "x/odd_name.jpg": _jpg(),
    })
    s = ins.summarize_zip(z)
    assert s["stems_matching_wiki_pattern"] == 3
    assert s["candidate_identity_tokens"] == 3          # '10', '20', 'odd'
    assert s["images_per_token"]["max"] == 2


def test_unreadable_image_is_reported(tmp_path):
    z = _zip(tmp_path / "a.zip", {"x/ok.jpg": _jpg(), "x/bad.jpg": b"not an image"})
    s = ins.summarize_zip(z, sample_n=10)
    assert s["sample"]["unreadable"] == 1


def test_overlap_detects_pairing_structure(tmp_path):
    a = _zip(tmp_path / "a.zip", {"wiki/00/1_x_1.jpg": _jpg(), "wiki/00/2_x_1.jpg": _jpg()})
    b = _zip(tmp_path / "b.zip", {"inpainting/00/1_x_1.jpg": _jpg(), "inpainting/00/9_x_1.jpg": _jpg()})
    o = ins.name_overlap(ins.list_images(a), ins.list_images(b))
    assert o["same_full_path"] == 0
    assert o["same_path_without_top_dir"] == 1
    assert o["same_basename"] == 1


def test_missing_zip_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Zip file not found"):
        ins.summarize_zip(tmp_path / "nope.zip")
