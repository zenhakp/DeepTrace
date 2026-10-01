"""EfficientNet-B4 deepfake detector wrapper.

Label order is fixed everywhere: index 0 = REAL, index 1 = MANIPULATED.
Detector confidence is the softmax probability of the predicted class. It is not calibrated and it is
not forensic certainty.
"""
from __future__ import annotations

from typing import Any, Dict, List

import timm
import torch

LABELS = ("REAL", "MANIPULATED")
MODEL_NAME = "EfficientNet-B4"


def build_model(pretrained: bool = False, drop_rate: float = 0.2) -> torch.nn.Module:
    """timm EfficientNet-B4 with a 2-class head. pretrained=True downloads ImageNet weights."""
    return timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=2, drop_rate=drop_rate)


class Detector:
    """Inference wrapper producing the structured prediction for downstream components."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()     # Part 2 receives this module to attach attribution

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> List[Dict[str, Any]]:
        """x: normalized float tensor (N, 3, H, W)."""
        logits = self.model(x.to(self.device)).float().cpu()
        probs = torch.softmax(logits, dim=1)
        out = []
        for lg, pr in zip(logits, probs):
            k = int(pr.argmax())
            out.append({
                "prediction": LABELS[k],
                "confidence": float(pr[k]),
                "real_probability": float(pr[0]),
                "fake_probability": float(pr[1]),
                "logits": [float(v) for v in lg],
                "model_name": MODEL_NAME,
            })
        return out


def load_detector(path, device: str = "cpu"):
    """Load a detector from a best.pt checkpoint written by training. Returns (Detector, meta)."""
    from deeptrace.common.paths import require_checkpoint

    p = require_checkpoint(path)
    ck = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "model" not in ck or "meta" not in ck:
        raise ValueError(f"Invalid checkpoint (expected 'model' and 'meta', i.e. best.pt, not last.pt): {p}")
    model = build_model(pretrained=False, drop_rate=0.0)
    model.load_state_dict(ck["model"])
    return Detector(model, device), ck["meta"]
