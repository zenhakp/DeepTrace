"""Gradient-based attribution (Grad-CAM and Grad-CAM++) for the Part 1 detector.

The maps show where the prepared 380x380 face crop most supports the target score: model-indicated
suspicious regions, not ground-truth manipulated pixels. No manipulation masks exist for our data, so
localization accuracy is not measured. The detector is used unchanged: the hook is removed after each call
and torch.autograd.grad is used, so no parameter gradients are accumulated.

At bn2 the head is global average pooling + linear, so the gradient is the same at every position (w / h*w).
Grad-CAM there equals CAM: its signed map sums exactly to the target score minus the bias. Grad-CAM++ reduces
to the positive-weight channels only and, on this detector, gave the same map for both classes (measured on
our samples, see docs), so the default is Grad-CAM on the logit difference MANIPULATED - REAL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deeptrace.models.detector import LABELS

METHODS = ("gradcam", "gradcam++")
TARGET_SCORES = ("logit_diff", "logit")    # logit_diff: logit[target] - logit[other] (2-class log-odds)
DEFAULT_LAYER = "bn2"       # timm EfficientNet: conv_head -> BatchNorm + SiLU, the map pooled into the classifier
MANIPULATED = LABELS.index("MANIPULATED")
EPS = 1e-12


def cam_from_activations(acts: torch.Tensor, grads: torch.Tensor, method: str) -> torch.Tensor:
    """acts, grads: (C, h, w) for one image. Returns the un-normalised signed map (h, w); ReLU it for display."""
    if method == "gradcam":
        weights = grads.mean(dim=(1, 2))
    elif method == "gradcam++":
        # Chattopadhay et al. 2018, closed form for an exponential of the class score
        g2, g3 = grads.pow(2), grads.pow(3)
        denom = 2 * g2 + acts.sum(dim=(1, 2), keepdim=True) * g3
        alpha = torch.where(grads != 0, g2 / torch.where(denom.abs() > EPS, denom, torch.ones_like(denom)),
                            torch.zeros_like(grads))
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))
    else:
        raise ValueError(f"Unknown attribution method {method!r}; expected one of {METHODS}")
    return (weights[:, None, None] * acts).sum(dim=0)


class GradCAM:
    """Class-activation map for one target class of an eval-mode model, at one named layer."""

    def __init__(self, model: torch.nn.Module, target_layer: str = DEFAULT_LAYER, method: str = "gradcam++"):
        if method not in METHODS:
            raise ValueError(f"Unknown attribution method {method!r}; expected one of {METHODS}")
        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise ValueError(f"Target layer {target_layer!r} not found in the model")
        self.model, self.layer, self.method, self.target_layer = model, modules[target_layer], method, target_layer

    def __call__(self, x: torch.Tensor, target_class: int = MANIPULATED,
                 target_score: str = "logit_diff") -> Dict[str, Any]:
        if target_score not in TARGET_SCORES:
            raise ValueError(f"Unknown target score {target_score!r}; expected one of {TARGET_SCORES}")
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected one image of shape (1, 3, H, W), got {tuple(x.shape)}")
        if self.model.training:
            raise ValueError("The model must be in eval mode; dropout and batch statistics would change the map")
        store: Dict[str, torch.Tensor] = {}
        handle = self.layer.register_forward_hook(lambda m, i, o: store.__setitem__("acts", o))
        try:
            with torch.enable_grad():
                device = next(self.model.parameters()).device
                inp = x.detach().to(device).clone().requires_grad_(True)   # graph exists even if weights are frozen
                logits = self.model(inp)
                acts = store["acts"]
                if target_score == "logit_diff":
                    if logits.shape[1] != 2:
                        raise ValueError("target_score 'logit_diff' needs a 2-class model")
                    score = logits[0, target_class] - logits[0, 1 - target_class]
                else:
                    score = logits[0, target_class]
                grads = torch.autograd.grad(score, acts)[0]
        finally:
            handle.remove()
        signed = cam_from_activations(acts[0].detach(), grads[0].detach(), self.method)
        cam = F.relu(signed)
        raw_max = float(cam.max())
        low = cam / raw_max if raw_max > EPS else torch.zeros_like(cam)
        up = F.interpolate(low[None, None], size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
        return {"cam": up.clamp(0, 1).cpu().numpy().astype(np.float32),
                "cam_lowres": low.cpu().numpy().astype(np.float32),
                "logits": [float(v) for v in logits[0].detach().cpu()],
                "cam_signed_lowres": signed.cpu().numpy().astype(np.float32),
                "raw_max": raw_max,
                "positive_mass": float(cam.sum()),                  # parts of the map supporting the target
                "negative_mass": float((signed - cam).sum())}       # parts against it (<= 0)


def to_original(cam: np.ndarray, crop_box: Sequence[int], input_size: Sequence[int]) -> np.ndarray:
    """Paste a crop-space map into original-image pixels (H, W). Outside the crop is NaN: not analysed."""
    left, top, right, bottom = (int(v) for v in crop_box)
    width, height = (int(v) for v in input_size)
    out = np.full((height, width), np.nan, dtype=np.float32)
    resized = Image.fromarray(np.asarray(cam, dtype=np.float32)).resize((right - left, bottom - top),
                                                                        Image.Resampling.BILINEAR)
    out[top:bottom, left:right] = np.clip(np.asarray(resized), 0.0, 1.0)
    return out


@dataclass
class Explanation:
    result: Dict[str, Any]                  # the Part 1 detector result, unchanged
    attribution: Optional[Dict[str, Any]]   # JSON-serialisable metadata; None when no prediction exists
    cam: Optional[np.ndarray]               # (S, S) map in [0, 1] over the prepared crop
    crop: Optional[Image.Image]             # the crop exactly as the model saw it


def explain_image(det, image, method: str = "gradcam", target_layer: str = DEFAULT_LAYER,
                  target_class: int = MANIPULATED, target_score: str = "logit_diff",
                  consistency_atol: float = 1e-4) -> Explanation:
    """Detector result plus an attribution map for the same prepared crop.

    The map comes from a second forward pass; its logits must match the detector's, otherwise it would not
    explain this prediction and we fail loudly. NO_FACE is propagated with no map.
    """
    result = det.analyze(image)
    if result["status"] != "OK":
        return Explanation(result, None, None, None)
    tensor, info, crop = det.prepare(image)
    if tensor is None or info["crop_box"] != result["crop_box"]:
        raise RuntimeError("Preprocessing was not repeatable: the second pass found a different face crop")
    out = GradCAM(det.detector.model, target_layer, method)(tensor, target_class, target_score)
    diff = max(abs(a - b) for a, b in zip(out["logits"], result["logits"]))
    if diff > consistency_atol:
        raise RuntimeError(f"Attribution forward pass disagrees with the detector (max logit difference "
                           f"{diff:.2e} > {consistency_atol}); the map would not explain this prediction")
    size = list(out["cam"].shape)
    attribution = {
        "method": method, "target_layer": target_layer, "target_class": target_class,
        "target_label": LABELS[target_class], "target_score": target_score,
        "map_size": size, "feature_size": list(out["cam_lowres"].shape), "map_space": "prepared_crop",
        "raw_max": out["raw_max"], "empty_map": out["raw_max"] <= EPS,
        "positive_mass": out["positive_mass"], "negative_mass": out["negative_mass"],
        "crop_box": list(result["crop_box"]), "input_size": list(result["input_size"]),
        "consistency_max_abs_logit_diff": diff,
    }
    return Explanation(result, attribution, out["cam"], crop)
