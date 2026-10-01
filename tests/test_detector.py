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
