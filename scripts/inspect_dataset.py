"""Inspect a downloaded dataset's zips (read-only) and save a JSON report.

  python scripts/inspect_dataset.py --dataset deepfakeface
"""
import argparse
import json
from pathlib import Path

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths, require_dataset_dir
from deeptrace.data.download import dataset_entry
from deeptrace.data.inspection import list_images, name_overlap, summarize_zip

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    p.add_argument("--sample", type=int, default=200, help="images per zip to open and check")
    p.add_argument("--reference", default="wiki.zip", help="zip whose names the others are compared with")
    a = p.parse_args()

    cfg = load_config(a.config)
    entry = dataset_entry(cfg, a.dataset)
    paths = get_paths(a.root)
    folder = require_dataset_dir(paths.datasets / a.dataset)

    summaries, names = {}, {}
    for fname in entry["files"]:
        if not fname.endswith(".zip"):
            continue
        print(f"Inspecting {fname} ...")
        summaries[fname] = summarize_zip(folder / fname, sample_n=a.sample, seed=cfg["run"]["seed"])
        names[fname] = list_images(folder / fname)

    overlaps = {}
    if a.reference in names:
        for fname, imgs in names.items():
            if fname != a.reference:
                overlaps[f"{a.reference} vs {fname}"] = name_overlap(names[a.reference], imgs)

    report = {"dataset": a.dataset, "revision": entry["revision"], "zips": summaries, "name_overlap": overlaps}
    out = paths.results / f"inspection_{a.dataset}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
