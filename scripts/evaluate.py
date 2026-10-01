"""Evaluate a trained detector on a split (optionally on a matched subset). Writes JSON + predictions CSV.

  python scripts/evaluate.py --dataset deepfakeface --run-name main --split val
  python scripts/evaluate.py --dataset deepfakeface --run-name main --split test
  python scripts/evaluate.py --dataset deepfakeface --run-name main --split test --matched-ids <matched json>
Use the test split only for the final model; do not tune on it.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from deeptrace.common.config import load_config
from deeptrace.common.device import resolve_device
from deeptrace.common.paths import get_paths
from deeptrace.data.dataset import FaceCropDataset, class_counts, load_split_rows, make_loader
from deeptrace.data.download import dataset_entry
from deeptrace.data.preprocess import faces_locations, params_from_cfg
from deeptrace.evaluation.metrics import classification_metrics, per_method_report
from deeptrace.evaluation.predict import predict_rows
from deeptrace.models.detector import load_detector

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--checkpoint", default=None, help="default: <checkpoints>/<run-name>/best.pt")
    p.add_argument("--matched-ids", default=None, help="JSON from make_matched_test.py")
    p.add_argument("--limit", type=int, default=None, help="debug: evaluate only N images")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    device = resolve_device(cfg["run"]["device"])
    entry = dataset_entry(cfg, a.dataset)
    pr = params_from_cfg(cfg)
    paths = get_paths(a.root)
    crops_dir, faces_csv = faces_locations(paths, a.dataset, pr)
    ckpt = Path(a.checkpoint) if a.checkpoint else paths.checkpoints / a.run_name / "best.pt"
    detector, meta = load_detector(ckpt, device)

    rows = load_split_rows(faces_csv, a.split)
    if a.matched_ids:
        ids = set(json.loads(Path(a.matched_ids).read_text(encoding="utf-8"))["image_ids"])
        rows = [r for r in rows if r["image_id"] in ids]
        if not rows:
            raise ValueError(f"No images of split '{a.split}' match the ids in {a.matched_ids}")
    if a.limit:
        rows = sorted(rows, key=lambda r: hashlib.sha256(f"limit:{r['image_id']}".encode()).hexdigest())[:a.limit]

    ds = FaceCropDataset(rows, crops_dir, image_size=meta["image_size"], train=False)
    loader = make_loader(ds, a.batch, shuffle=False, num_workers=a.workers, seed=0)
    records = predict_rows(detector.model, loader, device, amp=(device == "cuda"))

    metrics = classification_metrics([r["label"] for r in records], [r["fake_prob"] for r in records], a.threshold)
    report = {
        "dataset": a.dataset, "dataset_revision": entry["revision"], "split": a.split,
        "matched_subset": a.matched_ids, "limit": a.limit, "n_images": len(rows),
        "class_distribution": class_counts(rows), "threshold": a.threshold,
        "checkpoint": str(ckpt), "checkpoint_epoch": meta["epoch"], "checkpoint_global_step": meta["global_step"],
        "model": {"name": meta["model_name"], "labels": meta["labels"], "image_size": meta["image_size"],
                  "train_settings": meta["train"], "seed": meta["seed"]},
        "preprocessing": pr, "metrics": metrics, "per_method": per_method_report(records, a.threshold),
        "note": "Threshold 0.5 is a design choice; ROC-AUC does not depend on it. per_method: for 'real' the share "
                "predicted MANIPULATED is the false-positive rate; for fake methods it is recall.",
    }
    suffix = f"{a.split}" + ("_matched" if a.matched_ids else "") + (f"_limit{a.limit}" if a.limit else "")
    out_dir = paths.experiments / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out_dir / f"predictions_{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "label", "fake_prob"])
        w.writeheader()
        w.writerows(records)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out_dir / f'eval_{suffix}.json'}")


if __name__ == "__main__":
    main()
