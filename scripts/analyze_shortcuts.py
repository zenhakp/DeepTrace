"""Within-class shortcut analysis from a predictions CSV (no GPU needed).

  python scripts/analyze_shortcuts.py --dataset deepfakeface --predictions deeptrace_data/experiments/main/predictions_test.csv
For each class and feature: 5 equal-count bins, with the share predicted MANIPULATED per bin.
Observational: a trend suggests dependence on the feature, it does not prove causation.
"""
import argparse
import csv
import json
from pathlib import Path

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths
from deeptrace.data.preprocess import faces_locations, params_from_cfg, read_faces_csv
from deeptrace.evaluation.analysis import binned_rates

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--features", nargs="+", default=["q_lum_mean", "bytes_per_px", "native_crop_side"])
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=None)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    paths = get_paths(a.root)
    _, faces_csv = faces_locations(paths, a.dataset, params_from_cfg(cfg))
    if not faces_csv.is_file():
        raise FileNotFoundError(f"Faces manifest not found: {faces_csv}")
    if not Path(a.predictions).is_file():
        raise FileNotFoundError(f"Predictions file not found: {a.predictions}")
    rows = {r["image_id"]: r for r in read_faces_csv(faces_csv)}
    with open(a.predictions, newline="", encoding="utf-8") as f:
        preds = list(csv.DictReader(f))

    report = {}
    for feat in a.features:
        for label, name in ((0, "real"), (1, "manipulated")):
            table = binned_rates(rows, preds, feat, label, a.bins, a.threshold)
            report[f"{feat}/{name}"] = table
            print(f"\n{feat}  |  class: {name}  (share predicted MANIPULATED per bin; real = false-positive rate)")
            for t in table:
                print(f"  bin {t['bin']}  n={t['n']:5d}  {t['feature_min']:10.3f} .. {t['feature_max']:10.3f}  "
                      f"mean_prob={t['mean_fake_prob']:.4f}  share={t['share_predicted_manipulated']:.4f}")
    out = Path(a.out) if a.out else paths.results / "shortcut_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
