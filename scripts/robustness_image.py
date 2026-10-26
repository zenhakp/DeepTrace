"""Robustness for one or more images: controlled perturbations -> per-image JSON and a measured summary table.

  python scripts/robustness_image.py --image "path/to/photo1.jpg" "path/to/photo2.jpg"
Writes <stem>_robustness.json per image and robustness_summary.md / .json for the whole run into --out-dir
(default ~/Downloads/deeptrace_outputs/robustness; must be outside the repository). Quote paths with spaces.
A prediction flip or NO_FACE under a perturbation is recorded as an observation, not an error.
"""
import argparse
import json
import time
from pathlib import Path

from deeptrace.forensics import DEFAULT_OUT_DIR, load_forensics_config, output_dir
from deeptrace.forensics.robustness import NOTE, run_robustness, summarise
from deeptrace.models.inference import DeepTraceDetector


def _fmt(v):
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def markdown(table, n_images, settings) -> str:
    cols = ["mode", "label", "n", "flips", "no_face", "median_abs_delta_fake_probability",
            "max_abs_delta_fake_probability", "median_map_correlation", "median_region_iou", "median_crop_iou"]
    lines = [f"# Robustness summary (measured on {n_images} images: a small sample, not representative)", "",
             NOTE, "", f"Settings: `{json.dumps(settings)}`", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(row[c]) for c in cols) + " |" for row in table]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "robustness"))
    p.add_argument("--config", default=None, help="Part 2 config (default configs/part2_forensics.yaml)")
    p.add_argument("--run-name", default="mixed")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_forensics_config(a.config)
    out = output_dir(a.out_dir)
    det = DeepTraceDetector.from_run(a.run_name, root=a.root, device=cfg["run"]["device"], checkpoint=a.checkpoint)
    rob = cfg["robustness"]
    reports = []
    for image in a.image:
        path = Path(image)
        t = time.time()
        rep = run_robustness(det, path, cfg["attribution"], cfg["localization"], rob["perturbations"], rob["modes"],
                             cfg["run"]["seed"], rob["min_overlap_fraction"])
        rep["image"] = path.name
        reports.append(rep)
        (out / f"{path.stem}_robustness.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        b = rep["baseline"]
        print(f"{path.name}: baseline {b['status']} {b['prediction']} fake_p={_fmt(b['fake_probability'])} | "
              f"{len(rep['perturbations'])} runs, {len(rep['observations'])} observations | {time.time() - t:.1f}s")
        for o in rep["observations"]:
            print(f"    - {o['message']}")
    table = summarise(reports)
    settings = reports[0]["settings"] if reports else {}
    (out / "robustness_summary.json").write_text(json.dumps({"n_images": len(reports), "images": [r["image"] for r in reports],
                                                             "settings": settings, "note": NOTE, "table": table},
                                                            indent=2), encoding="utf-8")
    (out / "robustness_summary.md").write_text(markdown(table, len(reports), settings), encoding="utf-8")
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
