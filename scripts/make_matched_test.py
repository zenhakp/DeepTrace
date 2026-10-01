"""Build a matched evaluation subset (equal feature distribution in both classes).

  python scripts/make_matched_test.py --dataset deepfakeface --split test --feature q_lum_mean
"""
import argparse
import json
from pathlib import Path

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths
from deeptrace.data.matching import matched_image_ids, matching_report
from deeptrace.data.preprocess import params_from_cfg, read_faces_csv

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--feature", default="q_lum_mean")
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    pr = params_from_cfg(cfg)
    paths = get_paths(a.root)
    tag = f"s{pr['output_size']}_m{pr['crop_margin']}_q{pr['jpeg_quality']}"
    faces_csv = paths.processed / a.dataset / f"faces_{tag}_manifest.csv"
    if not faces_csv.is_file():
        raise FileNotFoundError(f"Faces manifest not found: {faces_csv}")

    rows = [r for r in read_faces_csv(faces_csv) if r["split"] == a.split]
    ids = matched_image_ids(rows, a.feature, a.bins, cfg["run"]["seed"])
    report = matching_report(rows, ids, a.feature)
    report.update({"dataset": a.dataset, "split": a.split, "bins": a.bins, "seed": cfg["run"]["seed"]})

    out = paths.splits / f"{a.dataset}_matched_{a.split}_{a.feature}.json"
    out.write_text(json.dumps({"report": report, "image_ids": ids}), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
