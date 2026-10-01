import torch

from deeptrace.models.detector import LABELS, MODEL_NAME, Detector, build_model


def test_label_order_is_fixed():
    assert LABELS == ("REAL", "MANIPULATED")


def test_predict_follows_the_output_contract():
    det = Detector(build_model(pretrained=False, drop_rate=0.0), device="cpu")
    out = det.predict(torch.randn(2, 3, 64, 64))
    assert len(out) == 2
    for o in out:
        assert set(o) == {"prediction", "confidence", "real_probability", "fake_probability",
                          "logits", "model_name"}
        assert abs(o["real_probability"] + o["fake_probability"] - 1) < 1e-5
        assert o["model_name"] == MODEL_NAME and len(o["logits"]) == 2
        top = "REAL" if o["real_probability"] >= o["fake_probability"] else "MANIPULATED"
        assert o["prediction"] == top
        assert o["confidence"] == max(o["real_probability"], o["fake_probability"])


def test_load_detector_roundtrip(tmp_path):
    from deeptrace.models.detector import load_detector
    torch.manual_seed(0)
    model = build_model(pretrained=False, drop_rate=0.0).eval()
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "meta": {"image_size": 64, "epoch": 1}}, path)
    det, meta = load_detector(path, "cpu")
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), det.model(x), atol=1e-5)
    assert meta["image_size"] == 64


def test_load_detector_missing_and_invalid(tmp_path):
    import pytest
    from deeptrace.models.detector import load_detector
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_detector(tmp_path / "nope.pt")
    torch.save({"model": {}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        load_detector(tmp_path / "bad.pt")
