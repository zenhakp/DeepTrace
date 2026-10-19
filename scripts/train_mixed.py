"""Train on several face-crop sources at once (class-balanced within each source, per-source validation). Resumable.

  Colab: python scripts/train_mixed.py --sources deepfakeface openfake_train_faces --run-name mixed
  Smoke: python scripts/train_mixed.py --run-name mixed_smoke --max-steps 30
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from deeptrace.common.config import load_config, save_config_snapshot
from deeptrace.common.device import describe_environment, resolve_device
from deeptrace.common.paths import get_paths
from deeptrace.common.seed import seed_everything
from deeptrace.data.mixed import load_mixed_rows
from deeptrace.data.preprocess import faces_locations, params_from_cfg
from deeptrace.models.detector import build_model
from deeptrace.training.trainer import run_training, train_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subset(rows, n_per_class_per_source, seed):
    out = []
    for src in sorted({r["source"] for r in rows}):
        for label in (0, 1):
            pool = sorted((r for r in rows if r["source"] == src and r["label"] == label),
                          key=lambda r: hashlib.sha256(f"subset:{seed}:{r['image_id']}".encode()).hexdigest())
            out += pool[:n_per_class_per_source]
    return out


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=["deepfakeface", "openfake_train_faces"])
    p.add_argument("--run-name", default="mixed")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--subset", type=int, default=None, help="debug: N rows per class per source and split")
    p.add_argument("--size", type=int, default=None)
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
    ts, seed = train_settings(cfg), cfg["run"]["seed"]
    seed_everything(seed)
    device = resolve_device(cfg["run"]["device"])
    pr = params_from_cfg(cfg)
    paths = get_paths(a.root)

    train_rows = load_mixed_rows(paths, pr, a.sources, "train")
    val_rows = load_mixed_rows(paths, pr, a.sources, "val")
    if a.subset:
        train_rows, val_rows = _subset(train_rows, a.subset, seed), _subset(val_rows, max(a.subset // 2, 2), seed)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        c = Counter((r["source"], r["label"]) for r in rows)
        print(name, {s: {"real": c[(s, 0)], "manipulated": c[(s, 1)]} for s in a.sources})

    run_dir, ckpt_dir = paths.experiments / a.run_name, paths.checkpoints / a.run_name
    save_config_snapshot(cfg, run_dir)
    sources_info = {}
    for s in a.sources:
        entry = (cfg.get("datasets", {}).get(s) or cfg.get("datasets", {}).get("openfake" if s.startswith("openfake") else s, {}))
        info = {"revision": entry.get("revision")}
        plan = faces_locations(paths, s, pr)[1].parent / "plan.json"
        if plan.is_file():
            info["plan"] = json.loads(plan.read_text(encoding="utf-8"))
        sources_info[s] = info
    (run_dir / "environment.json").write_text(json.dumps({
        **describe_environment(), "git_commit": _git_commit(), "sources": sources_info, "preprocess": pr,
        "subset": a.subset, "cli_overrides": overrides, "no_pretrained_flag": a.no_pretrained}, indent=2),
        encoding="utf-8")

    model = build_model(pretrained=ts["pretrained"], drop_rate=ts["drop_rate"])
    result = run_training(model, train_rows, val_rows, paths.processed, cfg, run_dir, ckpt_dir, device,
                          seed=seed, max_steps=a.max_steps, resume=not a.no_resume)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    print(f"Checkpoints: {ckpt_dir}\nLogs: {run_dir}")


if __name__ == "__main__":
    main()
