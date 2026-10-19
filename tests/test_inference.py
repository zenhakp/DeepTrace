import json

import numpy as np
import pytest
import torch
from PIL import Image

from deeptrace.models.detector import LABELS, Detector
from deeptrace.models.inference import DeepTraceDetector

FACE = (40.0, 40.0, 60.0, 60.0, 0.9)
PARAMS = {"crop_margin": 1.3, "jpeg_quality": 95}


def _tiny():
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, stride=2), torch.nn.AdaptiveAvgPool2d(1),
                               torch.nn.Flatten(), torch.nn.Linear(4, 2))


def _det(face=FACE, min_crop_side=32):
    return DeepTraceDetector(Detector(_tiny(), "cpu"), {"epoch": 3, "image_size": 64}, lambda rgb: face,
                             PARAMS, checkpoint="x/best.pt", min_crop_side=min_crop_side)


def _image(tmp_path, size=(160, 120)):
    path = tmp_path / "a.jpg"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path, "JPEG")
    return path


def test_analyze_follows_the_contract_and_is_json_serialisable(tmp_path):
    r = _det().analyze(_image(tmp_path))
    json.dumps(r)
    assert r["status"] == "OK" and r["prediction"] in LABELS and r["model_name"] == "EfficientNet-B4"
    assert abs(r["real_probability"] + r["fake_probability"] - 1) < 1e-5
    assert r["face_found"] and len(r["crop_box"]) == 4 and r["native_crop_side"] > 0
    assert r["limitations"] and r["warnings"] == []


def test_no_face_gives_no_prediction(tmp_path):
    r = _det(face=None).analyze(_image(tmp_path))
    assert r["status"] == "NO_FACE" and r["prediction"] is None and r["confidence"] is None
    assert r["warnings"][0]["code"] == "no_face_detected"


def test_small_face_adds_a_warning_but_still_predicts(tmp_path):
    r = _det(min_crop_side=96).analyze(_image(tmp_path))         # native crop side is about 78
    assert r["status"] == "OK" and r["prediction"] in LABELS
    assert [w["code"] for w in r["warnings"]] == ["small_face"]


def test_missing_and_unreadable_images_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        _det().analyze(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="Unable to read image"):
        _det().analyze(bad)


def test_prepare_returns_a_model_ready_square_crop(tmp_path):
    x, info, crop = _det().prepare(_image(tmp_path))
    assert tuple(x.shape) == (1, 3, 64, 64) and torch.isfinite(x).all()
    assert crop.size == (64, 64) and info["input_size"] == [160, 120]


def test_result_reports_checkpoint_provenance(tmp_path):
    det = DeepTraceDetector(Detector(_tiny(), "cpu"), {"epoch": 3, "image_size": 64}, lambda rgb: FACE, PARAMS,
                            checkpoint="best.pt", min_crop_side=32, checkpoint_sha256="abc123")
    r = det.analyze(_image(tmp_path))
    assert r["model"]["checkpoint"] == "best.pt" and r["model"]["checkpoint_sha256"] == "abc123"


def _plain_det(detector):
    return DeepTraceDetector(Detector(_tiny(), "cpu"), {"epoch": 3, "image_size": 64}, detector, PARAMS,
                             min_crop_side=32, detect_max_side=1280)


def test_large_images_are_downscaled_for_detection_and_boxes_mapped_back(tmp_path):
    seen = []

    def detector(rgb):
        seen.append(rgb.shape[:2])
        return (100.0, 80.0, 200.0, 200.0, 0.9)

    path = tmp_path / "big.jpg"
    Image.new("RGB", (2560, 1920), (120, 130, 140)).save(path, "JPEG")
    r = _plain_det(detector).analyze(path)
    assert seen == [(960, 1280)]
    assert r["face_box"] == [200.0, 160.0, 400.0, 400.0]
    assert r["input_size"] == [2560, 1920] and r["crop_box"][2] <= 2560 and r["crop_box"][3] <= 1920


def test_small_images_are_not_resized_for_detection(tmp_path):
    seen = []

    def detector(rgb):
        seen.append(rgb.shape[:2])
        return FACE

    _plain_det(detector).analyze(_image(tmp_path))
    assert seen == [(120, 160)]
