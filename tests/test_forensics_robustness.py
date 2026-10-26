import json
from collections import OrderedDict

import numpy as np
import pytest
import torch
from PIL import Image

from deeptrace.forensics import load_forensics_config
from deeptrace.forensics.robustness import (PERTURBATIONS, apply, box_iou, crop_to_tensor, map_similarity,
                                            perturbation_list, run_robustness, summarise)
from deeptrace.models.detector import Detector
from deeptrace.models.inference import DeepTraceDetector

FACE = (40.0, 40.0, 60.0, 60.0, 0.9)
PARAMS = {"crop_margin": 1.3, "jpeg_quality": 95}


def _image(size=(160, 120), seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(60, 200, (size[1], size[0], 3), dtype=np.uint8))


@pytest.mark.parametrize("family, level", [("jpeg_quality", 50), ("downscale", 0.5), ("gaussian_blur_sigma", 2.0),
                                           ("brightness", 1.1), ("contrast", 0.9), ("gaussian_noise_sigma", 2.0)])
def test_every_perturbation_keeps_size_and_changes_pixels(family, level):
    img = _image()
    out = apply(family, img, level, seed=42)
    assert out.size == img.size and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_identity_levels_and_seeded_noise_are_reproducible():
    img = _image()
    assert np.array_equal(np.asarray(apply("brightness", img, 1.0, 0)), np.asarray(img))
    a, b = apply("gaussian_noise_sigma", img, 2.0, 7), apply("gaussian_noise_sigma", img, 2.0, 7)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(apply("gaussian_noise_sigma", img, 2.0, 8)))


def test_unknown_perturbation_and_mode_fail_loudly():
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturbation_list({"rotate": [5]})
    with pytest.raises(ValueError, match="Unknown robustness mode"):
        run_robustness(None, None, {}, {}, {}, modes=["crop"])


def test_config_levels_are_valid_and_mild():
    rob = load_forensics_config()["robustness"]
    assert len(perturbation_list(rob["perturbations"])) == 12 and set(rob["perturbations"]) <= set(PERTURBATIONS)
    assert all(0.9 <= v <= 1.1 for v in rob["perturbations"]["brightness"] + rob["perturbations"]["contrast"])


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_map_similarity_uses_only_the_overlap():
    base = np.full((10, 10), np.nan, np.float32)
    base[:, :8] = np.linspace(0, 1, 80).reshape(10, 8)
    other = base.copy()
    other[:, :4] = np.nan                         # half of the baseline crop not covered
    s = map_similarity(base, other, 0.5, min_overlap=0.4)
    assert s["overlap_fraction"] == 0.5 and s["map_correlation"] == pytest.approx(1.0) and s["region_iou"] == 1.0
    assert map_similarity(base, other, 0.5, min_overlap=0.6)["map_correlation"] is None


def _brightness_model():
    """MANIPULATED logit = 10 x mean normalised red: darker images move towards REAL."""
    m = torch.nn.Sequential(OrderedDict(feat=torch.nn.Conv2d(3, 1, 1, bias=False), pool=torch.nn.AdaptiveAvgPool2d(1),
                                        flat=torch.nn.Flatten(), fc=torch.nn.Linear(1, 2)))
    with torch.no_grad():
        m.feat.weight.copy_(torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1))
        m.fc.weight.copy_(torch.tensor([[0.0], [10.0]]))
        m.fc.bias.zero_()
    return m.eval()


def _det(face_when=lambda rgb: True):
    return DeepTraceDetector(Detector(_brightness_model(), "cpu"), {"epoch": 3, "image_size": 64},
                             lambda rgb: FACE if face_when(rgb) else None, PARAMS, checkpoint="x/best.pt",
                             min_crop_side=32)


def _run(det, image, perturbations, modes=("pipeline", "fixed_crop")):
    cfg = load_forensics_config()
    return run_robustness(det, image, {**cfg["attribution"], "target_layer": "feat"}, cfg["localization"],
                          perturbations, modes, seed=42, min_overlap_fraction=0.5)


def test_fixed_crop_tensor_matches_prepare():
    det = _det()
    x, _, crop = det.prepare(_image())
    assert torch.equal(crop_to_tensor(crop), x)


def test_identity_perturbation_reproduces_the_baseline_in_both_modes():
    rep = _run(_det(), _image(), {"brightness": [1.0]})
    json.dumps(rep)
    assert rep["status"] == "OK" and len(rep["perturbations"]) == 2 and rep["observations"] == []
    for row in rep["perturbations"]:
        assert row["delta_fake_probability"] == pytest.approx(0, abs=1e-6) and row["prediction_flipped"] is False
        assert row["localization"]["map_correlation"] in (None, pytest.approx(1.0, abs=1e-4))
    assert rep["perturbations"][0]["crop_iou"] == 1.0 and rep["perturbations"][1]["crop_iou"] is None


def test_flips_and_lost_faces_are_recorded_as_observations():
    det = _det(face_when=lambda rgb: rgb.mean() > 80)          # darkening makes the face detector fail
    rep = _run(det, _image(), {"brightness": [0.4]})
    assert rep["baseline"]["prediction"] == "MANIPULATED"
    pipe, fixed = rep["perturbations"]
    assert pipe["no_face"] and pipe["prediction"] is None and pipe["prediction_flipped"] is None
    assert pipe["localization"] is None and pipe["evidence_strength"] == "not_applicable"
    assert fixed["prediction"] == "REAL" and fixed["prediction_flipped"] is True
    assert [o["kind"] for o in rep["observations"]] == ["no_face", "prediction_flip"]
    assert "MANIPULATED to REAL" in rep["observations"][1]["message"]
    table = summarise([rep])
    assert {(r["mode"], r["flips"], r["no_face"]) for r in table} == {("pipeline", 0, 1), ("fixed_crop", 1, 0)}


def test_no_face_in_the_original_is_not_applicable():
    rep = _run(_det(face_when=lambda rgb: False), _image(), {"brightness": [0.9]})
    assert rep["status"] == "NOT_APPLICABLE" and rep["perturbations"] == [] and "No face" in rep["reason"]
    assert summarise([rep]) == []
