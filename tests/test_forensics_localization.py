import json
from collections import OrderedDict

import numpy as np
import pytest
import torch
from PIL import Image

from deeptrace.forensics import load_forensics_config, output_dir
from deeptrace.forensics.attribution import explain_image
from deeptrace.forensics.localization import (crop_to_original, evidence_strength, extract_regions,
                                              format_probability, localize)
from deeptrace.forensics.overlay import render_overlay
from deeptrace.models.detector import Detector
from deeptrace.models.inference import DeepTraceDetector

FACE = (40.0, 40.0, 60.0, 60.0, 0.9)
PARAMS = {"crop_margin": 1.3, "jpeg_quality": 95}
ATTR = {"empty_map": False}


def _two_blobs():
    cam = np.zeros((100, 100), np.float32)
    cam[10:30, 10:30] = 1.0       # big, strong blob
    cam[60:70, 60:70] = 0.8       # smaller blob
    cam[90:92, 0:2] = 0.9         # 4 px: below the minimum area
    return cam


def test_regions_are_ranked_by_share_of_map_and_tiny_ones_dropped():
    regions = extract_regions(_two_blobs(), crop_box=(0, 0, 100, 100), threshold=0.5, min_area=0.005)
    assert [r["rank"] for r in regions] == [1, 2]
    assert regions[0]["bbox_crop"] == [10, 10, 30, 30] and regions[1]["bbox_crop"] == [60, 60, 70, 70]
    assert regions[0]["share_of_map"] > regions[1]["share_of_map"] and regions[0]["peak"] == 1.0
    assert abs(regions[0]["area_fraction"] - 0.04) < 1e-6


def test_max_regions_and_threshold_are_respected():
    assert len(extract_regions(_two_blobs(), (0, 0, 100, 100), 0.5, 0.005, max_regions=1)) == 1
    assert [r["bbox_crop"] for r in extract_regions(_two_blobs(), (0, 0, 100, 100), 0.85, 0.005)] == [[10, 10, 30, 30]]
    assert extract_regions(np.zeros((10, 10), np.float32), (0, 0, 10, 10)) == []


def test_a_single_feature_cell_peak_survives_the_default_settings():
    low = torch.zeros(1, 1, 12, 12)
    low[0, 0, 5, 7] = 1.0
    cam = torch.nn.functional.interpolate(low, size=(380, 380), mode="bilinear", align_corners=False)[0, 0].numpy()
    cfg = load_forensics_config()["localization"]
    regions = extract_regions(cam, (0, 0, 380, 380), cfg["region_threshold"], cfg["min_region_area"])
    assert len(regions) == 1


def test_probabilities_are_never_shown_as_zero_or_one():
    assert [format_probability(p) for p in (0.002, 0.01, 0.6518, 0.99, 0.9996)] == \
        ["below 0.01", "0.01", "0.65", "0.99", "above 0.99"]


def test_boxes_map_back_into_original_pixels():
    assert crop_to_original([0, 0, 380, 190], crop_box=(100, 50, 290, 240), map_size=380) == [100, 50, 290, 145]


@pytest.mark.parametrize("fake_p, attr, expected", [
    (0.91, ATTR, "model_indicated"),
    (0.50, ATTR, "model_indicated"),       # boundary: the detector's own decision point
    (0.49, ATTR, "weak"),
    (0.02, ATTR, "weak"),
    (0.91, {"empty_map": True}, "none"),
    (None, None, "not_applicable"),
])
def test_evidence_strength_rule(fake_p, attr, expected):
    label, message = evidence_strength(fake_p, attr, min_fake_probability=0.5)
    assert label == expected and message
    if label == "weak":
        assert "transparency only" in message and "fake-class probability" in message


def _tiny_det(face=FACE):
    m = torch.nn.Sequential(OrderedDict(conv=torch.nn.Conv2d(3, 4, 3, stride=2), act=torch.nn.ReLU(),
                                        pool=torch.nn.AdaptiveAvgPool2d(1), flat=torch.nn.Flatten(),
                                        fc=torch.nn.Linear(4, 2)))
    return DeepTraceDetector(Detector(m, "cpu"), {"epoch": 3, "image_size": 64}, lambda rgb: face, PARAMS,
                             checkpoint="x/best.pt", min_crop_side=32)


def _image(tmp_path):
    path = tmp_path / "a.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(path, "JPEG")
    return path


def test_localize_and_overlay_end_to_end(tmp_path):
    path = _image(tmp_path)
    e = explain_image(_tiny_det(), path, target_layer="act")
    loc = localize(e, **load_forensics_config()["localization"])
    json.dumps(loc)
    assert loc["evidence_strength"] in ("model_indicated", "weak", "none")
    for r in loc["regions"]:
        l, t, rr, b = r["bbox_original"]
        cl, ct, cr, cb = e.result["crop_box"]
        assert cl <= l < rr <= cr and ct <= t < b <= cb
    img = render_overlay(e, loc, Image.open(path))
    assert img.size[0] == 64 + round(160 * 64 / 120) and img.size[1] > 64


def test_no_face_gives_no_regions_and_still_renders(tmp_path):
    path = _image(tmp_path)
    e = explain_image(_tiny_det(face=None), path, target_layer="act")
    loc = localize(e)
    assert loc["evidence_strength"] == "not_applicable" and loc["regions"] == [] and loc["attribution"] is None
    assert render_overlay(e, loc, Image.open(path)).size[0] == round(160 * 380 / 120)


def test_output_dir_refuses_the_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="inside the repository"):
        output_dir(repo / "outputs", repo_root=repo)
    assert output_dir(tmp_path / "out", repo_root=repo).is_dir()
