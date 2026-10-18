"""Image-level inference for downstream components (Part 2 and Part 3).

image (path or PIL) -> face detection -> the same crop / resize / JPEG re-encode used for training ->
EfficientNet-B4 -> JSON-serialisable result. Detector confidence is a softmax probability, not forensic certainty.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps

from deeptrace.common.config import load_config
from deeptrace.common.device import resolve_device
from deeptrace.common.paths import get_paths
from deeptrace.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from deeptrace.data.faces import YuNetDetector, ensure_model, square_crop_box
from deeptrace.data.preprocess import params_from_cfg
from deeptrace.models.detector import MODEL_NAME, Detector, load_detector

ImageLike = Union[str, Path, Image.Image]
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "part1_default.yaml"

LIMITATIONS: List[str] = [
    "Trained only on DeepFakeFace (Stable Diffusion text-to-image and inpainting fakes, InsightFace face swaps).",
    "On a sample of held-out modern generators (OpenFake test) the detector did not separate real from manipulated "
    "images (ROC-AUC 0.42, see docs/experiment_log.md); results outside the training distribution are unreliable.",
    "Confidence is the softmax probability of the predicted class (uncalibrated). It is not forensic certainty and "
    "not the probability that the image is actually manipulated.",
    "Only the largest detected face is analysed; images without a detectable face receive no prediction.",
]


class DeepTraceDetector:
    def __init__(self, detector: Detector, meta: Dict[str, Any], face_detector: Callable, params: Dict[str, Any],
                 checkpoint: str = "", min_crop_side: int = 96, reencode_jpeg: bool = True):
        self.detector, self.meta, self.face_detector, self.params = detector, meta, face_detector, params
        self.checkpoint, self.min_crop_side, self.reencode_jpeg = checkpoint, min_crop_side, reencode_jpeg
        self.image_size = int(meta["image_size"])

    @classmethod
    def from_run(cls, run_name: str = "main", config: Optional[str] = None, root: Optional[str] = None,
                 device: Optional[str] = None, checkpoint: Optional[str] = None, min_crop_side: int = 96):
        cfg = load_config(config or DEFAULT_CONFIG)
        params = params_from_cfg(cfg)
        paths = get_paths(root, create=False)
        dev = resolve_device(device or cfg["run"]["device"])
        ckpt = Path(checkpoint) if checkpoint else paths.checkpoints / run_name / "best.pt"
        detector, meta = load_detector(ckpt, dev)
        face_model = paths.checkpoints / params["face_model_file"]
        ensure_model(face_model, params["face_model_url"])
        return cls(detector, meta, YuNetDetector(face_model, params["face_score_threshold"]), params,
                   str(ckpt), min_crop_side)

    @staticmethod
    def _load(image: ImageLike) -> Image.Image:
        if isinstance(image, Image.Image):
            img = image
        else:
            p = Path(image)
            if not p.is_file():
                raise FileNotFoundError(f"Image file not found: {p}")
            try:
                img = Image.open(p)
                img.load()
            except Exception as exc:
                raise RuntimeError(f"Unable to read image: {p}") from exc
        return ImageOps.exif_transpose(img).convert("RGB")

    def prepare(self, image: ImageLike) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Optional[Image.Image]]:
        """(model input tensor (1,3,S,S) or None, info dict, the crop as the model sees it or None)."""
        img = self._load(image)
        W, H = img.size
        face = self.face_detector(np.asarray(img))
        info: Dict[str, Any] = {"input_size": [W, H], "face_found": face is not None}
        if face is None:
            return None, info, None
        x, y, w, h, score = face
        box = square_crop_box(face, W, H, self.params["crop_margin"])
        S = self.image_size
        crop = img.crop(box).resize((S, S), Image.Resampling.BICUBIC)
        if self.reencode_jpeg:                         # training crops were stored as JPEG q95
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=self.params["jpeg_quality"])
            buf.seek(0)
            crop = Image.open(buf).convert("RGB")
        arr = (np.asarray(crop, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)
        info.update({"face_box": [float(x), float(y), float(w), float(h)], "face_score": float(score),
                     "crop_box": [int(v) for v in box], "native_crop_side": int(box[2] - box[0])})
        return tensor, info, crop

    def analyze_with_crop(self, image: ImageLike) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        tensor, info, crop = self.prepare(image)
        result: Dict[str, Any] = {
            "status": "OK", "model_name": MODEL_NAME, **info,
            "model": {"checkpoint": self.checkpoint, "checkpoint_epoch": self.meta.get("epoch"),
                      "image_size": self.image_size, "trained_on": "DeepFakeFace (OpenRL/DeepFakeFace)"},
            "warnings": [], "limitations": list(LIMITATIONS)}
        if tensor is None:
            result.update({"status": "NO_FACE", "prediction": None, "confidence": None,
                           "real_probability": None, "fake_probability": None, "logits": None})
            result["warnings"].append({"code": "no_face_detected",
                                       "message": "No face was detected; the detector only analyses face crops, so no "
                                                  "prediction was made."})
            return result, None
        result.update(self.detector.predict(tensor)[0])
        if info["native_crop_side"] < self.min_crop_side:
            result["warnings"].append({"code": "small_face",
                                       "message": f"The face crop is small ({info['native_crop_side']} px native); "
                                                  "accuracy was lower for small faces in testing. Treat the result "
                                                  "with extra caution."})
        return result, crop

    def analyze(self, image: ImageLike) -> Dict[str, Any]:
        return self.analyze_with_crop(image)[0]
