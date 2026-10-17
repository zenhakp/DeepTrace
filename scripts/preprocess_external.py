"""Create face crops + a manifest for an EXTERNAL dataset (cross-dataset test only). Resumable.

  folders:  python scripts/preprocess_external.py --name ffhq140k --source folder --real-dir D/real --fake-dir D/fake --fake-name stylegan --sample-rate 0.25
  parquet:  python scripts/preprocess_external.py --name openfake_test --source parquet --parquet "D/**/*.parquet" --sample-rate 0.3
Then evaluate:  python scripts/evaluate.py --dataset <name> --run-name main --split test
"""
import argparse
import glob
import itertools
import json
from pathlib import Path

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths
from deeptrace.data.external import folder_items, parquet_items, run_external, summarize_external
from deeptrace.data.faces import YuNetDetector, ensure_model
from deeptrace.data.preprocess import faces_locations, params_from_cfg, read_faces_csv

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="name of the external dataset (used for output folders)")
    p.add_argument("--source", required=True, choices=["folder", "parquet"])
    p.add_argument("--real-dir")
    p.add_argument("--fake-dir")
    p.add_argument("--fake-name", default="fake", help="method name for fake images in folder mode")
    p.add_argument("--parquet", help="glob for parquet files (columns: image, label, model)")
    p.add_argument("--sample-rate", type=float, default=1.0, help="deterministic random fraction of images to use")
    p.add_argument("--limit", type=int, default=None, help="stop after N newly processed images (not class-balanced)")
    p.add_argument("--min-crop-side", type=int, default=96, help="drop faces whose native crop is smaller (design choice)")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    pr = params_from_cfg(cfg)
    seed = cfg["run"]["seed"]
    paths = get_paths(a.root)
    crops_dir, faces_csv = faces_locations(paths, a.name, pr)

    if a.source == "folder":
        if not (a.real_dir and a.fake_dir):
            raise ValueError("Folder mode needs --real-dir and --fake-dir")
        items = itertools.chain(folder_items(a.real_dir, 0, "real"), folder_items(a.fake_dir, 1, a.fake_name))
    else:
        files = sorted(glob.glob(a.parquet or "", recursive=True))
        if not files:
            raise FileNotFoundError(f"No parquet files match: {a.parquet}")
        print(f"{len(files)} parquet file(s)")
        items = parquet_items(files)

    model_path = paths.checkpoints / pr["face_model_file"]
    model_sha = ensure_model(model_path, pr["face_model_url"])
    detector = YuNetDetector(model_path, pr["face_score_threshold"])

    n = run_external(items, detector, pr, crops_dir, faces_csv, sample_rate=a.sample_rate, seed=seed,
                     limit=a.limit, min_crop_side=a.min_crop_side)
    print(f"Processed {n} images in this run")

    rows = read_faces_csv(faces_csv)
    report = {"name": a.name, "source": a.source, "sample_rate": a.sample_rate, "seed": seed,
              "min_crop_side": a.min_crop_side, "params": pr, "face_model_sha256": model_sha,
              "n_rows": len(rows), "summary": summarize_external(rows)}
    out = faces_csv.with_name(faces_csv.name.replace("_manifest.csv", "_report.json"))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nCrops: {crops_dir}\nManifest: {faces_csv}\nReport: {out}")


if __name__ == "__main__":
    main()
