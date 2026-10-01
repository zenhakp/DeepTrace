"""Build the image manifest and the leakage-safe split for a downloaded dataset.

  python scripts/build_manifest.py --dataset deepfakeface
"""
import argparse
import json
from pathlib import Path, PurePosixPath

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths, require_dataset_dir
from deeptrace.data.download import dataset_entry
from deeptrace.data.inspection import WIKI_STEM
from deeptrace.data.manifest import (REAL_METHOD, assign_splits, check_groups, scan_zip,
                                     shortcut_report, split_settings, summarize,
                                     verify_no_leakage, write_manifest)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    p.add_argument("--reference", default="wiki.zip", help="the zip holding the REAL images")
    a = p.parse_args()

    cfg = load_config(a.config)
    entry = dataset_entry(cfg, a.dataset)
    seed, fractions = split_settings(cfg)
    paths = get_paths(a.root)
    folder = require_dataset_dir(paths.datasets / a.dataset)

    records, methods = [], []
    for fname in entry["files"]:
        if not fname.endswith(".zip"):
            continue
        is_real = fname == a.reference
        method = REAL_METHOD if is_real else Path(fname).stem
        methods.append(method)
        print(f"Scanning {fname} as '{method}' (label {0 if is_real else 1}) ...")
        records += scan_zip(folder / fname, method, 0 if is_real else 1)
    if REAL_METHOD not in methods:
        raise ValueError(f"Reference zip {a.reference} not found among datasets.{a.dataset}.files")

    n_groups = check_groups(records, methods)
    assignment = assign_splits([r["group_id"] for r in records], fractions, seed)
    for r in records:
        r["split"] = assignment[r["group_id"]]
    verify_no_leakage(records)

    manifest_path = paths.splits / f"{a.dataset}_manifest.csv"
    sha = write_manifest(records, manifest_path)
    odd = sorted(PurePosixPath(r["member"]).stem for r in records
                 if r["method"] == REAL_METHOD and not WIKI_STEM.match(PurePosixPath(r["member"]).stem))

    report = {
        "dataset": a.dataset, "revision": entry["revision"],
        "split_unit": "group = one real photo + every fake derived from it",
        "seed": seed, "fractions": fractions, "n_groups": n_groups, "n_images": len(records),
        "manifest_file": manifest_path.name, "manifest_sha256": sha,
        "counts": summarize(records),
        "shortcut_check": shortcut_report(records),
        "nonstandard_real_stems": odd,
    }
    out = paths.splits / f"{a.dataset}_split.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {manifest_path}\nSaved: {out}")


if __name__ == "__main__":
    main()
