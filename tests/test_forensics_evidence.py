import json
from collections import OrderedDict

import numpy as np
import pytest
import torch
from PIL import Image

from deeptrace.forensics import REPO_ROOT, load_forensics_config
from deeptrace.forensics.evidence import (SCHEMA_PATH, analyze_file, image_evidence, media_type, validate,
                                          video_evidence)
from deeptrace.models.detector import Detector
from deeptrace.models.inference import DeepTraceDetector

FACE = (40.0, 40.0, 60.0, 60.0, 0.9)
PARAMS = {"crop_margin": 1.3, "jpeg_quality": 95}
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
REQUIRED_BY_PROJECT = ["prediction", "confidence", "suspicious_regions", "prediction_stability", "confidence_stability",
                       "localization_stability", "evidence_stability", "temporal_stability", "suspicious_intervals",
                       "robustness_observations", "limitations"]
EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"


def _det(face=FACE):
    m = torch.nn.Sequential(OrderedDict(feat=torch.nn.Conv2d(3, 1, 1, bias=False), pool=torch.nn.AdaptiveAvgPool2d(1),
                                        flat=torch.nn.Flatten(), fc=torch.nn.Linear(1, 2)))
    with torch.no_grad():
        m.feat.weight.copy_(torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1))
        m.fc.weight.copy_(torch.tensor([[0.0], [10.0]]))
        m.fc.bias.zero_()
    return DeepTraceDetector(Detector(m.eval(), "cpu"), {"epoch": 3, "image_size": 64},
                             (lambda rgb: face) if not callable(face) else face, PARAMS, checkpoint="best.pt",
                             min_crop_side=32, checkpoint_sha256="ab" * 32, model_version="test")


def _cfg():
    cfg = load_forensics_config()
    cfg["attribution"]["target_layer"] = "feat"
    cfg["robustness"]["perturbations"] = {"brightness": [0.9], "jpeg_quality": [50]}
    return cfg


def _image(tmp_path, name="a.jpg", size=(160, 120)):
    path = tmp_path / name
    Image.fromarray(np.random.default_rng(0).integers(60, 200, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def test_validator_reports_types_required_keys_enums_and_ranges():
    schema = {"type": "object", "required": ["a", "b"],
              "properties": {"a": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                             "b": {"type": "string", "enum": ["x"]},
                             "c": {"type": "array", "items": {"type": "integer"}}}}
    assert validate({"a": 0.5, "b": "x", "c": [1, 2]}, schema) == []
    assert validate({"a": None, "b": "x"}, schema) == []
    errors = validate({"a": 2, "b": "y", "c": [1, "2"]}, schema)
    assert len(errors) == 3 and any("maximum" in e for e in errors) and any("$.c[1]" in e for e in errors)
    assert validate({"b": "x"}, schema) == ["$: missing required key 'a'"]
    assert validate({"a": True, "b": "x"}, schema)          # a boolean is not a number


def test_schema_requires_every_field_the_project_asked_for():
    required = set(SCHEMA["required"])
    assert set(REQUIRED_BY_PROJECT) <= required and {"model", "status", "warnings"} <= required
    assert {"version", "checkpoint_sha256"} <= set(SCHEMA["properties"]["model"]["required"])


@pytest.mark.parametrize("name", ["example_image", "example_video", "example_no_face"])
def test_committed_examples_match_the_schema_and_are_anonymised(name):
    ev = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    assert validate(ev, SCHEMA) == []
    assert ev["input"]["file_name"].startswith(name) and ev["input"]["sha256"] == "0" * 64
    assert ev["model"]["version"] == "v2"
    text = json.dumps(ev)
    assert "Users" not in text and ":\\\\" not in text


def test_image_evidence_ok(tmp_path):
    ev, e, loc = image_evidence(_det(), _image(tmp_path), _cfg())
    json.dumps(ev)
    assert validate(ev, SCHEMA) == []
    assert ev["status"] == "OK" and ev["media_type"] == "image" and ev["temporal_stability"] is None
    assert ev["suspicious_intervals"] == [] and ev["frames"] is None and ev["observations"] == []
    assert ev["model"]["version"] == "test" and ev["model"]["checkpoint_sha256"] == "ab" * 32
    assert ev["robustness"]["modes"] == ["pipeline"] and ev["robustness"]["runs"] == 2
    assert ev["evidence_stability"]["mode"] == "pipeline" and "not a probability" in ev["evidence_stability"]["note"]
    assert any("not ground-truth manipulated pixels" in lim for lim in ev["limitations"])
    assert len(ev["input"]["sha256"]) == 64 and ev["settings"]["seed"] == 42


def test_small_face_under_122_px_becomes_a_warning_and_a_limitation(tmp_path):
    ev, _, _ = image_evidence(_det(), _image(tmp_path), _cfg())     # native crop side is about 78 px
    assert "small_face_part2" in [w["code"] for w in ev["warnings"]]
    assert any("under 122 px" in lim for lim in ev["limitations"])


def test_no_face_image_has_no_verdict_and_no_scores(tmp_path):
    ev, e, loc = image_evidence(_det(face=None), _image(tmp_path), _cfg())
    assert validate(ev, SCHEMA) == []
    assert ev["status"] == "NO_FACE" and ev["prediction"] is None and ev["confidence"] is None
    assert all(ev[k] is None for k in ("prediction_stability", "confidence_stability", "localization_stability",
                                       "evidence_stability", "temporal_stability", "localization"))
    assert ev["suspicious_regions"] == [] and ev["robustness_observations"] == []
    assert [w["code"] for w in ev["warnings"]] == ["no_face_detected"]
    assert any("No face was detected" in lim for lim in ev["limitations"])


def _video(path, brightness, fps=10.0, size=(160, 120)):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    rng = np.random.default_rng(0)
    for b in brightness:
        w.write(np.clip(rng.integers(90, 160, (size[1], size[0], 3)) * b, 0, 255).astype(np.uint8))
    w.release()
    return path


def test_video_evidence_matches_the_schema(tmp_path):
    det = _det(face=lambda rgb: FACE if rgb.mean() > 40 else None)
    ev = video_evidence(det, _video(tmp_path / "v.avi", [1.5] * 10 + [0.1] * 10 + [1.5] * 10), _cfg())
    json.dumps(ev)
    assert validate(ev, SCHEMA) == []
    assert ev["media_type"] == "video" and ev["temporal_stability"] is not None and len(ev["frames"]) == 6
    assert ev["face"]["usable_face_fraction"] == pytest.approx(4 / 6, abs=1e-4)
    assert [o["kind"] for o in ev["observations"]][:1] == ["no_face_frames"]
    assert ev["evidence_stability"]["driver"] in ("prediction", "confidence", "localization", None)
    assert all("frame" in r and "time_s" in r for r in ev["suspicious_regions"])
    assert ev["prediction_stability"]["method"] == "median over key frames"


def test_file_type_is_chosen_by_extension_and_unknown_types_fail(tmp_path):
    assert media_type("a.JPG") == "image" and media_type("b.mp4") == "video"
    with pytest.raises(ValueError, match="Unsupported file type"):
        media_type("c.gif")
    ev = analyze_file(_det(), _image(tmp_path, "x.png"), _cfg())
    assert ev["media_type"] == "image" and validate(ev, SCHEMA) == []
