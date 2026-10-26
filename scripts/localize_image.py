"""Localization for one or more images: detector result + Grad-CAM map -> regions JSON and overlay PNG.

  python scripts/localize_image.py --image "path/to/photo1.jpg" "path/to/photo2.jpg"
  python scripts/localize_image.py --image photo.jpg --out-dir ~/Downloads/deeptrace_outputs/localization
Writes <stem>_localization.json and <stem>_overlay.png. The output folder must be outside the repository.
Regions are model-indicated suspicious regions, not ground-truth manipulated pixels.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from deeptrace.forensics import DEFAULT_OUT_DIR, load_forensics_config, output_dir
from deeptrace.forensics.attribution import explain_image
from deeptrace.forensics.localization import localize
from deeptrace.forensics.overlay import render_overlay
from deeptrace.models.inference import DeepTraceDetector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "localization"))
    p.add_argument("--config", default=None, help="Part 2 config (default configs/part2_forensics.yaml)")
    p.add_argument("--run-name", default="mixed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_forensics_config(a.config)
    out = output_dir(a.out_dir)
    det = DeepTraceDetector.from_run(a.run_name, root=a.root, device=cfg["run"]["device"], checkpoint=a.checkpoint)
    for image in a.image:
        path = Path(image)
        e = explain_image(det, path, **cfg["attribution"])
        loc = localize(e, **cfg["localization"])
        original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        render_overlay(e, loc, original, **cfg["overlay"]).save(out / f"{path.stem}_overlay.png")
        record = {"image": path.name, "detector": e.result, "localization": loc,
                  "settings": {"seed": cfg["run"]["seed"], "attribution": cfg["attribution"],
                               "localization": cfg["localization"]}}
        (out / f"{path.stem}_localization.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        fp = e.result["fake_probability"]
        print(f"{path.name}: {e.result['status']} {e.result['prediction']} "
              f"fake_p={'n/a' if fp is None else f'{fp:.4f}'} strength={loc['evidence_strength']} "
              f"regions={len(loc['regions'])} positive_fraction={loc['positive_fraction']}")
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
