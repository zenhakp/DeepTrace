import copy
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deeptrace.common.paths import get_paths
from deeptrace.forensics import load_forensics_config
from deeptrace.forensics.attribution import GradCAM, explain_image, to_original
from deeptrace.models.detector import Detector
from deeptrace.models.inference import DeepTraceDetector

FACE = (40.0, 40.0, 60.0, 60.0, 0.9)
PARAMS = {"crop_margin": 1.3, "jpeg_quality": 95}


def _toy(fc_weight=((0.0, 1.0), (1.0, 0.0))):
    """Channel 0 copies red, channel 1 copies green; REAL logit = green mean, MANIPULATED logit = red mean."""
    m = torch.nn.Sequential(OrderedDict(feat=torch.nn.Conv2d(3, 2, 1, bias=False),
                                        pool=torch.nn.AdaptiveAvgPool2d(1), flat=torch.nn.Flatten(),
                                        fc=torch.nn.Linear(2, 2)))
    with torch.no_grad():
        m.feat.weight.zero_()
        m.feat.weight[0, 0] = 1.0
        m.feat.weight[1, 1] = 1.0
        m.fc.weight.copy_(torch.tensor(fc_weight))
        m.fc.bias.copy_(torch.tensor([0.3, -0.2]))
    return m.eval()


def _spots():
    x = torch.zeros(1, 3, 16, 16)
    x[0, 0, 2:5, 2:5] = 1.0       # red spot top-left  -> MANIPULATED evidence
    x[0, 1, 10:13, 10:13] = 1.0   # green spot bottom-right -> REAL evidence
    return x


def _peak(cam):
    return np.unravel_index(np.argmax(cam), cam.shape)


@pytest.mark.parametrize("method", ["gradcam", "gradcam++"])
@pytest.mark.parametrize("target_score", ["logit", "logit_diff"])
def test_map_peaks_on_the_evidence_for_the_target_class(method, target_score):
    cam_fake = GradCAM(_toy(), "feat", method)(_spots(), 1, target_score)["cam"]
    cam_real = GradCAM(_toy(), "feat", method)(_spots(), 0, target_score)["cam"]
    assert cam_fake.shape == (16, 16) and cam_fake.min() >= 0 and abs(cam_fake.max() - 1) < 1e-6
    r, c = _peak(cam_fake)
    assert 2 <= r < 5 and 2 <= c < 5
    r, c = _peak(cam_real)
    assert 10 <= r < 13 and 10 <= c < 13


def test_gradcam_signed_map_decomposes_the_logit_difference_exactly():
    m, x = _toy(), _spots()
    out = GradCAM(m, "feat", "gradcam")(x, 1, "logit_diff")
    bias = float((m.fc.bias[1] - m.fc.bias[0]).detach())
    assert abs(out["cam_signed_lowres"].sum() + bias - (out["logits"][1] - out["logits"][0])) < 1e-5
    assert abs(out["positive_mass"] + out["negative_mass"] - out["cam_signed_lowres"].sum()) < 1e-5


def test_no_supporting_evidence_gives_an_empty_map():
    out = GradCAM(_toy(fc_weight=((0.0, 0.0), (-1.0, 0.0))), "feat", "gradcam")(_spots(), 1, "logit")
    assert out["raw_max"] == 0 and np.all(out["cam"] == 0) and out["negative_mass"] < 0


def test_attribution_leaves_the_model_untouched():
    m = _toy()
    before = copy.deepcopy(m.state_dict())
    GradCAM(m, "feat")(_spots())
    assert all(p.grad is None for p in m.parameters())
    assert not m.feat._forward_hooks
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_invalid_requests_fail_loudly():
    with pytest.raises(ValueError, match="Unknown attribution method"):
        GradCAM(_toy(), "feat", "lime")
    with pytest.raises(ValueError, match="not found"):
        GradCAM(_toy(), "nope")
    with pytest.raises(ValueError, match="Unknown target score"):
        GradCAM(_toy(), "feat")(_spots(), 1, "softmax")
    with pytest.raises(ValueError, match="one image"):
        GradCAM(_toy(), "feat")(torch.zeros(2, 3, 16, 16))
    with pytest.raises(ValueError, match="eval mode"):
        GradCAM(_toy().train(), "feat")(_spots())


def test_to_original_pastes_the_map_and_marks_the_rest_not_analysed():
    out = to_original(np.ones((8, 8), np.float32), crop_box=(10, 20, 30, 40), input_size=(50, 60))
    assert out.shape == (60, 50)
    assert np.all(out[20:40, 10:30] == 1.0)
    assert np.isnan(out[:20]).all() and np.isnan(out[40:]).all() and np.isnan(out[:, :10]).all()


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


def test_explain_image_matches_the_detector_and_is_json_serialisable(tmp_path):
    e = explain_image(_tiny_det(), _image(tmp_path), target_layer="act")
    json.dumps(e.attribution)
    assert e.result["status"] == "OK" and e.cam.shape == (64, 64) and e.crop.size == (64, 64)
    a = e.attribution
    assert a["consistency_max_abs_logit_diff"] <= 1e-4 and a["crop_box"] == e.result["crop_box"]
    assert a["method"] == "gradcam" and a["target_label"] == "MANIPULATED" and a["map_space"] == "prepared_crop"


def test_no_face_is_propagated_without_a_map(tmp_path):
    e = explain_image(_tiny_det(face=None), _image(tmp_path), target_layer="act")
    assert e.result["status"] == "NO_FACE" and e.attribution is None and e.cam is None


def test_disagreement_with_the_detector_fails_loudly(tmp_path):
    det = _tiny_det()
    real_analyze = det.analyze
    det.analyze = lambda image: {**real_analyze(image), "logits": [9.0, 9.0]}
    with pytest.raises(RuntimeError, match="disagrees with the detector"):
        explain_image(det, _image(tmp_path), target_layer="act")


def test_config_defaults_load():
    cfg = load_forensics_config()["attribution"]
    assert cfg["method"] == "gradcam" and cfg["target_layer"] == "bn2" and cfg["target_score"] == "logit_diff"


# Real-detector checks: need the private checkpoint and a face photo outside the repo (DEEPTRACE_TEST_FACE).
CKPT = get_paths(create=False).checkpoints / "mixed" / "best.pt"
FACE_IMG = os.environ.get("DEEPTRACE_TEST_FACE", "")
needs_real = pytest.mark.skipif(not (CKPT.is_file() and FACE_IMG and Path(FACE_IMG).is_file()),
                                reason="needs checkpoints/mixed/best.pt and DEEPTRACE_TEST_FACE=<face photo>")


@pytest.fixture(scope="module")
def real():
    det = DeepTraceDetector.from_run("mixed", device="cpu")
    x, _, _ = det.prepare(FACE_IMG)
    return det, x


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@needs_real
def test_real_detector_consistency_and_exact_decomposition(real):
    det, _ = real
    e = explain_image(det, FACE_IMG, **load_forensics_config()["attribution"])
    a = e.attribution
    assert a["consistency_max_abs_logit_diff"] <= 1e-4 and a["feature_size"] == [12, 12] and e.cam.shape == (380, 380)
    bias = det.detector.model.classifier.bias.detach()
    logits = e.result["logits"]
    assert abs(a["positive_mass"] + a["negative_mass"] + float(bias[1] - bias[0]) - (logits[1] - logits[0])) < 1e-3


@needs_real
def test_real_detector_gradcam_is_class_sensitive(real):
    det, x = real
    cam = lambda c: GradCAM(det.detector.model, "bn2", "gradcam")(x, c, "logit")["cam_lowres"]
    assert _corr(cam(1), cam(0)) < 0.5         # design threshold; measured -0.15 to -0.34 on our 11 samples


@needs_real
def test_real_detector_map_changes_when_top_layers_are_randomised(real):
    """Model-randomisation sanity check (Adebayo et al. 2018): a map that survives random weights explains nothing."""
    det, x = real
    model = copy.deepcopy(det.detector.model)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for module in (model.classifier, model.conv_head, model.bn2, model.blocks[-1]):
            for p in module.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * p.std().clamp_min(1e-3))
    trained = GradCAM(det.detector.model)(x)["cam"]
    randomised = GradCAM(model.eval())(x)["cam"]
    assert _corr(trained, randomised) < 0.5    # design threshold
