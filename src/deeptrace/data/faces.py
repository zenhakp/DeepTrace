"""Face detection (OpenCV YuNet) and crop/colour helpers. Only numpy and Pillow are needed to test them."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

Face = Tuple[float, float, float, float, float]  # x, y, w, h, score


def ensure_model(path, url: str) -> str:
    """Download the detector once if missing. Returns its sha256 (recorded, not assumed)."""
    p = Path(path)
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face detector model to {p} ...")
        try:
            urllib.request.urlretrieve(url, p)
        except Exception as exc:
            if p.exists():
                p.unlink()
            raise RuntimeError(
                f"Could not download the face model from {url}: {exc}. "
                f"Download it manually and place it at {p}") from exc
    if p.stat().st_size < 50_000:
        raise RuntimeError(f"Face model file looks invalid (too small): {p}. Delete it and retry.")
    return hashlib.sha256(p.read_bytes()).hexdigest()


class YuNetDetector:
    """Returns the largest detected face as (x, y, w, h, score), or None."""

    def __init__(self, model_path, score_threshold: float = 0.6):
        import cv2
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def __call__(self, rgb: np.ndarray) -> Optional[Face]:
        h, w = rgb.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(np.ascontiguousarray(rgb[:, :, ::-1]))   # YuNet expects BGR
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda a: a[2] * a[3])
        return float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])


def square_crop_box(face: Face, img_w: int, img_h: int, margin: float) -> Tuple[int, int, int, int]:
    """Square box around the face that always lies inside the image (no padding artifacts)."""
    x, y, w, h, _ = face
    side = min(margin * max(w, h), img_w, img_h)
    cx, cy = x + w / 2, y + h / 2
    left = min(max(cx - side / 2, 0), img_w - side)
    top = min(max(cy - side / 2, 0), img_h - side)
    l, t = int(round(left)), int(round(top))
    s = min(int(round(side)), img_w - l, img_h - t)
    return l, t, l + s, t + s


def mean_chroma(rgb: np.ndarray) -> float:
    a = rgb.astype(np.int16)
    return float((np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean() / 2)


def is_gray_source(mode: str, rgb: np.ndarray, eps: float) -> bool:
    return mode in ("L", "LA", "1") or mean_chroma(rgb) < eps


def q_lum_mean(img) -> Optional[float]:
    """Mean of the JPEG luminance quantization table (a proxy for the original compression quality)."""
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return None
    return float(np.mean(list(q[0])))


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image: a simple blur / upsampling indicator."""
    g = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())
