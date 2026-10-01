"""Train the EfficientNet-B4 detector (resumable).

  CPU smoke test:  python scripts/train.py --dataset deepfakeface --no-pretrained --subset 40 --epochs 1 --size 96 --batch 8 --workers 0 --run-name smoke
  Colab GPU:       python scripts/train.py --dataset deepfakeface --run-name main
"""
import argparse
import json
import subprocess
from pathlib import Path

from deeptrace.common.config import load_config, save_config_snapshot
from deeptrace.common.device import describe_environment, resolve_device
from deeptrace.common.paths import get_paths
from deeptrace.common.seed import seed_everything
from deeptrace.data.dataset import class_counts, load_split_rows
from deeptrace.data.download import dataset_entry
from deeptrace.data.preprocess import faces_locations, params_from_cfg, subset_groups
from deeptrace.models.detector import build_model
from deeptrace.training.trainer import run_training, train_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subset(rows, n, seed):
    keep = set(subset_groups([r["group_id"] for r in rows], n, seed))
    return [r for r in rows if r["group_id"] in keep]


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke tests); resumable")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--subset", type=int, default=None, help="debug: use only N train groups (N//2 for validation)")
    p.add_argument("--size", type=int, default=None, help="override loading.image_size (debug)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    a = p.parse_args()

    overrides = {}
    if a.epochs:
        overrides["train.epochs"] = a.epochs
    if a.size:
        overrides["loading.image_size"] = a.size
    if a.batch:
        overrides["loading.batch_size"] = a.batch
    if a.workers is not None:
        overrides["loading.num_workers"] = a.workers
    cfg = load_config(a.config, overrides)
    if a.no_pretrained:
        cfg["train"]["pretrained"] = False
    ts = train_settings(cfg)
    seed = cfg["run"]["seed"]
    seed_everything(seed)
    device = resolve_device(cfg["run"]["device"])

    entry = dataset_entry(cfg, a.dataset)
    pr = params_from_cfg(cfg)
    paths = get_paths(a.root)
    crops_dir, faces_csv = faces_locations(paths, a.dataset, pr)

    train_rows = load_split_rows(faces_csv, "train")
    val_rows = load_split_rows(faces_csv, "val")
    if a.subset:
        train_rows, val_rows = _subset(train_rows, a.subset, seed), _subset(val_rows, max(a.subset // 2, 2), seed)
    print(f"train {class_counts(train_rows)}  val {class_counts(val_rows)}  (0=real, 1=manipulated)")

    run_name = a.run_name or f"{a.dataset}_effb4_seed{seed}"
    run_dir, ckpt_dir = paths.experiments / run_name, paths.checkpoints / run_name
    save_config_snapshot(cfg, run_dir)
    (run_dir / "environment.json").write_text(json.dumps({
        **describe_environment(), "git_commit": _git_commit(), "dataset": a.dataset,
        "dataset_revision": entry["revision"], "preprocess": pr, "subset_groups": a.subset,
        "cli_overrides": overrides, "no_pretrained_flag": a.no_pretrained}, indent=2), encoding="utf-8")

    model = build_model(pretrained=ts["pretrained"], drop_rate=ts["drop_rate"])
    result = run_training(model, train_rows, val_rows, crops_dir, cfg, run_dir, ckpt_dir, device,
                          seed=seed, max_steps=a.max_steps, resume=not a.no_resume)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    print(f"Checkpoints: {ckpt_dir}\nLogs: {run_dir}")


if __name__ == "__main__":
    main()
