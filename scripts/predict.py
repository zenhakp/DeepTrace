"""Analyse one image: face crop -> EfficientNet-B4 -> structured JSON (the Part 1 contract).

  python scripts/predict.py --image photo.jpg
  python scripts/predict.py --image photo.jpg --save-crop crop.jpg
Needs <DEEPTRACE_ROOT>/checkpoints/<run-name>/best.pt (see docs/detector_api.md).
"""
import argparse
import json
from pathlib import Path

from deeptrace.models.inference import DeepTraceDetector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--run-name", default="main")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    p.add_argument("--save-crop", default=None, help="write the face crop the model sees to this file")
    a = p.parse_args()

    det = DeepTraceDetector.from_run(a.run_name, config=a.config, root=a.root, device=a.device,
                                     checkpoint=a.checkpoint)
    result, crop = det.analyze_with_crop(a.image)
    if a.save_crop:
        if crop is None:
            print("No face found: crop not saved")
        else:
            Path(a.save_crop).parent.mkdir(parents=True, exist_ok=True)
            crop.save(a.save_crop, "JPEG", quality=95)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
