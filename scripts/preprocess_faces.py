"""Detect faces, write uniform face crops, and audit metadata shortcuts. Resumable.

  python scripts/preprocess_faces.py --dataset deepfakeface --limit-groups 300   # quick check
  python scripts/preprocess_faces.py --dataset deepfakeface                      # everything (resumes)
"""
import argparse
import json
import time
from pathlib import Path

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths, require_dataset_dir
from deeptrace.data.faces import YuNetDetector, ensure_model
from deeptrace.data.preprocess import (load_manifest, params_from_cfg, read_faces_csv, run_preprocess,
                                       shortcut_audit, summarize_faces)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    p.add_argument("--limit-groups", type=int, default=None, help="process only a random subset of groups")
    a = p.parse_args()

    cfg = load_config(a.config)
    params = params_from_cfg(cfg)
    paths = get_paths(a.root)
    folder = require_dataset_dir(paths.datasets / a.dataset)
    rows = load_manifest(paths.splits / f"{a.dataset}_manifest.csv")

    model_path = paths.checkpoints / params["face_model_file"]
    model_sha = ensure_model(model_path, params["face_model_url"])
    detector = YuNetDetector(model_path, params["face_score_threshold"])

    tag = f"s{params['output_size']}_m{params['crop_margin']}_q{params['jpeg_quality']}"
    out_dir = paths.processed / a.dataset / f"faces_{tag}"
    faces_csv = paths.processed / a.dataset / f"faces_{tag}_manifest.csv"

    t0 = time.time()
    n = run_preprocess(rows, folder, out_dir, faces_csv, detector, params,
                       limit_groups=a.limit_groups, seed=cfg["run"]["seed"])
    print(f"Processed {n} groups this run in {time.time() - t0:.0f}s")

    all_rows = read_faces_csv(faces_csv)
    report = {"dataset": a.dataset, "params": params, "face_model_sha256": model_sha,
              "rows_in_faces_manifest": len(all_rows),
              "summary": summarize_faces(all_rows), "shortcut_audit": shortcut_audit(all_rows)}
    out = paths.processed / a.dataset / f"faces_{tag}_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nCrops: {out_dir}\nManifest: {faces_csv}\nReport: {out}")


if __name__ == "__main__":
    main()
