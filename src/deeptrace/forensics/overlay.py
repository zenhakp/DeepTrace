"""Overlay figure: the prepared crop with its map and regions, and the original image with the crop box.

The banner always carries the evidence_strength message, so a bright area is never shown without saying
how much it means. Weak evidence is drawn fainter and in gray. Text is ASCII (OpenCV Hershey fonts).
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from deeptrace.forensics.attribution import Explanation
from deeptrace.forensics.localization import format_probability

COLORMAPS = {"inferno": cv2.COLORMAP_INFERNO, "viridis": cv2.COLORMAP_VIRIDIS, "jet": cv2.COLORMAP_JET}
STYLE = {   # strength -> (box colour RGB, banner colour RGB)
    "model_indicated": ((230, 40, 40), (255, 255, 255)),
    "weak": ((170, 170, 170), (255, 200, 80)),
    "none": ((170, 170, 170), (200, 200, 200)),
    "not_applicable": ((170, 170, 170), (200, 200, 200)),
}
FOOTER = "Model-indicated regions, not ground-truth manipulated pixels. Localization accuracy: not yet measured."
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_block(lines, width, colour, scale=0.45, pad=8):
    line_h = int(22 * scale / 0.45)
    block = np.zeros((pad * 2 + line_h * len(lines), width, 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(block, line, (pad, pad + line_h * (i + 1) - 6), FONT, scale, colour, 1, cv2.LINE_AA)
    return block


def render_overlay(explanation: Explanation, loc: Dict[str, Any], original: Image.Image,
                   alpha_indicated: float = 0.45, alpha_weak: float = 0.25, colormap: str = "inferno",
                   title: Optional[str] = None) -> Image.Image:
    strength = loc["evidence_strength"]
    box_colour, banner_colour = STYLE[strength]
    orig = np.asarray(original.convert("RGB")).copy()
    panels = []
    if explanation.cam is not None:
        crop = np.asarray(explanation.crop.convert("RGB")).copy()
        if strength in ("model_indicated", "weak"):
            heat = cv2.applyColorMap((explanation.cam * 255).astype(np.uint8), COLORMAPS[colormap])[:, :, ::-1]
            alpha = alpha_indicated if strength == "model_indicated" else alpha_weak
            crop = (crop * (1 - alpha) + heat * alpha).astype(np.uint8)
        for r in loc["regions"]:
            l, t, rr, b = r["bbox_crop"]
            cv2.rectangle(crop, (l, t), (rr - 1, b - 1), box_colour, 2)
            cv2.putText(crop, str(r["rank"]), (l + 4, t + 18), FONT, 0.6, box_colour, 2, cv2.LINE_AA)
            ol, ot, orr, ob = (int(round(v)) for v in r["bbox_original"])
            cv2.rectangle(orig, (ol, ot), (orr - 1, ob - 1), box_colour, 2)
        cl, ct, cr, cb = explanation.result["crop_box"]
        cv2.rectangle(orig, (cl, ct), (cr - 1, cb - 1), (255, 255, 255), 1)
        panels.append(crop)
    size = panels[0].shape[0] if panels else 380
    h, w = orig.shape[:2]
    panels.append(cv2.resize(orig, (max(1, round(w * size / h)), size), interpolation=cv2.INTER_AREA))
    body = np.concatenate(panels, axis=1)
    width = body.shape[1]
    chars = max(40, int(width / 8.2))
    res = explanation.result
    head = title or (f"{res['prediction']}  fake-class probability {format_probability(res['fake_probability'])}"
                     if res.get("prediction") else f"status {res['status']}: no prediction")
    banner = _text_block([head] + textwrap.wrap(loc["message"], chars), width, banner_colour)
    footer = _text_block(textwrap.wrap(FOOTER, chars), width, (200, 200, 200), scale=0.4)
    return Image.fromarray(np.concatenate([banner, body, footer], axis=0))
