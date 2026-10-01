"""CPU wiring check: can the pipeline overfit a handful of crops? This is NOT a research result.

  python scripts/sanity_overfit.py --dataset deepfakeface --size 224 --steps 40
"""
import argparse
import hashlib
from itertools import cycle
from pathlib import Path

import torch

from deeptrace.common.config import load_config
from deeptrace.common.device import resolve_device
from deeptrace.common.paths import get_paths
from deeptrace.common.seed import seed_everything
from deeptrace.data.dataset import FaceCropDataset, load_split_rows, make_loader
from deeptrace.data.preprocess import params_from_cfg
from deeptrace.models.detector import Detector, build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=32, help="images to overfit (half real, half manipulated)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--size", type=int, default=224, help="input size (smaller = faster on CPU)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pretrained", action="store_true", help="download ImageNet weights (needs internet)")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "part1_default.yaml"))
    p.add_argument("--root", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    seed_everything(cfg["run"]["seed"])
    device = resolve_device(cfg["run"]["device"])
    pr = params_from_cfg(cfg)
    paths = get_paths(a.root)
    tag = f"s{pr['output_size']}_m{pr['crop_margin']}_q{pr['jpeg_quality']}"
    crops_dir = paths.processed / a.dataset / f"faces_{tag}"
    faces_csv = paths.processed / a.dataset / f"faces_{tag}_manifest.csv"

    def pick(label: int, k: int):
        rows = [r for r in load_split_rows(faces_csv, "train") if r["label"] == label]
        rows.sort(key=lambda r: hashlib.sha256(f"sanity:{r['image_id']}".encode()).hexdigest())
        return rows[:k]

    sel = pick(0, a.n // 2) + pick(1, a.n // 2)
    ds = FaceCropDataset(sel, crops_dir, image_size=a.size, train=False)
    loader = make_loader(ds, a.batch, shuffle=True, num_workers=0, seed=cfg["run"]["seed"])

    model = build_model(pretrained=a.pretrained, drop_rate=0.0).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    print(f"device={device} images={len(ds)} size={a.size} batch={a.batch} pretrained={a.pretrained}")

    losses, batches = [], cycle(loader)
    for step in range(1, a.steps + 1):
        x, y, _ = next(batches)
        loss = lossf(model(x.to(device)), y.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if step % 5 == 0:
            print(f"step {step:3d}  mean loss of last 5 = {sum(losses[-5:]) / 5:.4f}")

    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    det = Detector(model, str(device))
    correct = total = 0
    for x, y, _ in make_loader(ds, a.batch, shuffle=False, seed=0):
        for o, t in zip(det.predict(x), y.tolist()):
            correct += int(o["prediction"] == ("MANIPULATED" if t == 1 else "REAL"))
            total += 1
    print(f"loss: first-5 mean {first:.4f} -> last-5 mean {last:.4f}")
    print(f"accuracy on these {total} training images: {correct}/{total}  (training data only, not a result)")
    print("WIRING CHECK PASSED" if last < first else "WIRING CHECK: loss did not decrease, investigate")


if __name__ == "__main__":
    main()
