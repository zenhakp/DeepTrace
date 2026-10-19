"""Build a face-cropped subset of OpenFake TRAIN shards (training data for the generalization experiment).

Resumable: finished shards are recorded and skipped; an interrupted shard resumes image by image.
Each shard (about 5.7 GB) is downloaded, processed and deleted. Splits are by whole shard.
  python scripts/build_openfake_subset.py --n-train-shards 6 --n-val-shards 1 [--max-new-shards 1]
"""
import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from deeptrace.common.config import load_config
from deeptrace.common.paths import get_paths
from deeptrace.data.external import choose_shards, parquet_items, run_external, summarize_external
from deeptrace.data.faces import YuNetDetector, ensure_model
from deeptrace.data.preprocess import faces_locations, params_from_cfg, read_faces_csv

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="openfake_train_faces")
    p.add_argument("--n-train-shards", type=int, default=6)
    p.add_argument("--n-val-shards", type=int, default=1)
    p.add_argument("--max-new-shards", type=int, default=None, help="process at most N unfinished shards, then stop")
    p.add_argument("--min-crop-side", type=int, default=96)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    pr, seed = params_from_cfg(cfg), cfg["run"]["seed"]
    repo, rev = cfg["datasets"]["openfake"]["hf_repo"], cfg["datasets"]["openfake"]["revision"]
    paths = get_paths(a.root)
    crops_dir, faces_csv = faces_locations(paths, a.name, pr)
    base = faces_csv.parent
    shard_dir = base / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    info = HfApi().dataset_info(repo, revision=rev, files_metadata=True)
    sizes = {s.rfilename: int(s.size or 0) for s in info.siblings
             if s.rfilename.endswith(".parquet") and s.rfilename.startswith("core/train")}
    plan = choose_shards(list(sizes), a.n_train_shards, a.n_val_shards)
    plan_doc = {"repo": repo, "revision": rev, "plan": plan, "min_crop_side": a.min_crop_side}
    plan_path = base / "plan.json"
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != json.loads(json.dumps(plan_doc)):
            raise ValueError(f"A different plan already exists in {plan_path}. Use a new --name, or delete it deliberately.")
    else:
        plan_path.write_text(json.dumps(plan_doc, indent=2), encoding="utf-8")
    print(f"{len(plan)} shards, total {sum(sizes[n] for n, _ in plan) / 1e9:.1f} GB (each deleted after processing)")
    for rel, split in plan:
        print(f"  {split:5} {sizes[rel] / 1e9:5.2f} GB  {rel}")

    model_path = paths.checkpoints / pr["face_model_file"]
    ensure_model(model_path, pr["face_model_url"])
    detector = YuNetDetector(model_path, pr["face_score_threshold"])
    dl_dir = Path(tempfile.gettempdir()) / "openfake_dl"

    done_now = 0
    for k, (rel, split) in enumerate(plan, 1):
        marker = shard_dir / (Path(rel).name + ".done.json")
        if marker.is_file():
            print(f"[{k}/{len(plan)}] already done: {rel}")
            continue
        if a.max_new_shards is not None and done_now >= a.max_new_shards:
            print("Stopping: --max-new-shards reached (rerun to continue)")
            break
        print(f"[{k}/{len(plan)}] {rel} split={split}")
        t0 = time.time()
        hf_hub_download(repo, rel, repo_type="dataset", revision=rev, local_dir=str(dl_dir))
        t1 = time.time()
        n = run_external(parquet_items([dl_dir / rel]), detector, pr, crops_dir, faces_csv, sample_rate=1.0,
                         seed=seed, min_crop_side=a.min_crop_side, split=split)
        t2 = time.time()
        marker.write_text(json.dumps({"shard": rel, "split": split, "processed_now": n,
                                      "download_s": round(t1 - t0, 1), "process_s": round(t2 - t1, 1)}),
                          encoding="utf-8")
        (dl_dir / rel).unlink()
        done_now += 1
        print(f"  done: {n} images this run; download {t1 - t0:.0f} s, face-crop {t2 - t1:.0f} s")

    rows = read_faces_csv(faces_csv) if faces_csv.is_file() else []
    table = Counter((r["split"], "real" if r["label"] == 0 else "manipulated") for r in rows if r["kept"] == 1)
    done = len(list(shard_dir.glob("*.done.json")))
    print(f"\nshards finished: {done}/{len(plan)}   rows in manifest: {len(rows)}")
    for key in sorted(table):
        print(f"  kept {key[0]:5} {key[1]:12} {table[key]}")
    if rows:
        s = summarize_external(rows)
        (base / "summary.json").write_text(json.dumps({"shards_done": done, "kept": {f"{a}/{b}": n for (a, b), n in table.items()},
                                                       "summary": s}, indent=2), encoding="utf-8")
        print("generators seen:", len([m for m in s["by_method"] if m != "real"]),
              "| median native crop side:", s["median_native_crop_side"])


if __name__ == "__main__":
    main()
