"""Training loop: transfer learning, class-balanced epochs, resumable checkpoints, per-epoch validation."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from deeptrace.data.dataset import FaceCropDataset, loading_from_cfg, make_loader
from deeptrace.evaluation.metrics import classification_metrics
from deeptrace.evaluation.predict import predict_rows
from deeptrace.models.detector import LABELS, MODEL_NAME


def train_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = cfg.get("train")
    if not isinstance(t, dict):
        raise ValueError("Invalid configuration: missing 'train' section")

    def integer(key: str, lo: int) -> int:
        v = t.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise ValueError(f"Invalid configuration: train.{key} must be an integer >= {lo}, got {v!r}")
        return v

    def number(key: str, lo: float, hi: float, lo_open: bool = False) -> float:
        v = t.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v >= hi or (v <= lo if lo_open else v < lo):
            raise ValueError(f"Invalid configuration: train.{key} must be in {'(' if lo_open else '['}{lo}, {hi}), got {v!r}")
        return float(v)

    def boolean(key: str) -> bool:
        v = t.get(key)
        if not isinstance(v, bool):
            raise ValueError(f"Invalid configuration: train.{key} must be true or false, got {v!r}")
        return v

    return {
        "epochs": integer("epochs", 1), "lr": number("lr", 0, 10, lo_open=True),
        "weight_decay": number("weight_decay", 0, 10), "warmup_steps": integer("warmup_steps", 0),
        "amp": boolean("amp"), "fakes_per_real": integer("fakes_per_real", 1),
        "ckpt_every_steps": integer("ckpt_every_steps", 1), "log_every_steps": integer("log_every_steps", 1),
        "pretrained": boolean("pretrained"), "drop_rate": number("drop_rate", 0, 1),
    }


def epoch_plan(rows: Sequence[Dict[str, Any]], epoch: int, seed: int, fakes_per_real: int = 1) -> List[int]:
    """Indices into rows for one epoch: all real images + fakes_per_real x as many fakes, spread evenly over
    the fake methods and re-drawn every epoch. Deterministic in (seed, epoch), so training can resume."""
    real = [i for i, r in enumerate(rows) if r["label"] == 0]
    by_method: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["label"] == 1:
            by_method[r["method"]].append(i)
    if not real or not by_method:
        raise ValueError("epoch_plan needs both real and manipulated rows")

    def key(salt: str, i: int) -> str:
        return hashlib.sha256(f"{salt}:{seed}:{epoch}:{rows[i]['image_id']}".encode("utf-8")).hexdigest()

    methods = sorted(by_method)
    base, extra = divmod(len(real) * fakes_per_real, len(methods))
    chosen = list(real)
    for j, m in enumerate(methods):
        pool = sorted(by_method[m], key=lambda i: key("pick", i))
        chosen += pool[: base + (1 if j < extra else 0)]
    return sorted(chosen, key=lambda i: key("order", i))


def lr_factor(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, payload: Dict[str, Any]) -> None:
    """Atomic: write a temp file, then replace, so a disconnect cannot leave a half-written checkpoint."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, p)


def run_training(model: torch.nn.Module, train_rows: Sequence[Dict[str, Any]], val_rows: Sequence[Dict[str, Any]],
                 crops_dir, cfg: Dict[str, Any], run_dir, ckpt_dir, device: str, seed: int = 42,
                 max_steps: Optional[int] = None, resume: bool = True,
                 log: Callable[[str], None] = print) -> Dict[str, Any]:
    ts, ld = train_settings(cfg), loading_from_cfg(cfg)
    run_dir, ckpt_dir = Path(run_dir), Path(ckpt_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    model.to(dev)
    use_amp = ts["amp"] and dev.type == "cuda"
    bs, nw = ld["batch_size"], ld["num_workers"]

    train_ds = FaceCropDataset(train_rows, crops_dir, ld["image_size"], train=True,
                               degrade_cfg=ld["degrade"], hflip=ld["hflip"])
    val_ds = FaceCropDataset(val_rows, crops_dir, ld["image_size"], train=False)
    plan_len = len(epoch_plan(train_rows, 0, seed, ts["fakes_per_real"]))
    steps_per_epoch = math.ceil(plan_len / bs)
    total_steps = steps_per_epoch * ts["epochs"]
    fingerprint = {"batch_size": bs, "image_size": ld["image_size"], "epochs": ts["epochs"],
                   "fakes_per_real": ts["fakes_per_real"], "n_train_rows": len(train_rows)}

    opt = torch.optim.AdamW(model.parameters(), lr=ts["lr"], weight_decay=ts["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, ts["warmup_steps"], total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = {"epoch": 0, "step_in_epoch": 0, "global_step": 0, "best_val_auc": -1.0}

    last_path = ckpt_dir / "last.pt"
    if resume and last_path.is_file():
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        if ck["fingerprint"] != fingerprint:
            raise ValueError(f"Cannot resume: settings differ from the checkpoint ({ck['fingerprint']} vs "
                             f"{fingerprint}). Use a new --run-name or --no-resume.")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        log(f"Resumed from {last_path}: epoch {state['epoch'] + 1}, step {state['step_in_epoch']}, "
            f"global step {state['global_step']}")

    log_path = run_dir / "train_log.jsonl"

    def jlog(entry: Dict[str, Any]) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def save_last() -> None:
        save_checkpoint(last_path, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                                    "state": dict(state), "fingerprint": fingerprint})

    history: List[Dict[str, Any]] = []
    stopped = False
    log(f"device={dev} amp={use_amp} train_rows={len(train_rows)} plan/epoch={plan_len} "
        f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    for epoch in range(state["epoch"], ts["epochs"]):
        idx = epoch_plan(train_rows, epoch, seed, ts["fakes_per_real"])[state["step_in_epoch"] * bs:]
        loader = make_loader(Subset(train_ds, idx), bs, shuffle=False, num_workers=nw, seed=seed + epoch)
        model.train()
        window: List[float] = []
        t_last = time.time()
        for x, y, _ in loader:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss ({float(loss)}) at global step {state['global_step'] + 1}. "
                                   f"Lower the learning rate or inspect the data.")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            state["step_in_epoch"] += 1
            state["global_step"] += 1
            window.append(float(loss))
            if state["global_step"] % ts["log_every_steps"] == 0:
                now = time.time()
                entry = {"type": "train", "epoch": epoch + 1, "global_step": state["global_step"],
                         "loss": sum(window) / len(window), "lr": opt.param_groups[0]["lr"],
                         "it_per_s": len(window) / max(now - t_last, 1e-9)}
                jlog(entry)
                log(f"epoch {epoch + 1}/{ts['epochs']} step {state['step_in_epoch']}/{steps_per_epoch} "
                    f"loss {entry['loss']:.4f} lr {entry['lr']:.2e} {entry['it_per_s']:.2f} it/s")
                window, t_last = [], now
            if state["global_step"] % ts["ckpt_every_steps"] == 0:
                save_last()
            if max_steps and state["global_step"] >= max_steps:
                save_last()
                stopped = True
                break
        if stopped:
            break

        state["epoch"], state["step_in_epoch"] = epoch + 1, 0
        records = predict_rows(model, make_loader(val_ds, bs, shuffle=False, num_workers=nw, seed=0), dev, use_amp)
        vm = classification_metrics([r["label"] for r in records], [r["fake_prob"] for r in records])
        entry = {"type": "val", "epoch": epoch + 1, "global_step": state["global_step"], **vm}
        jlog(entry)
        history.append(entry)
        log(f"VALIDATION epoch {epoch + 1}: ROC-AUC {vm['roc_auc']} balanced acc {vm['balanced_accuracy']} "
            f"acc {vm['accuracy']:.4f} (n={vm['n']})")
        if vm["roc_auc"] is not None and vm["roc_auc"] > state["best_val_auc"]:
            state["best_val_auc"] = vm["roc_auc"]
            save_checkpoint(ckpt_dir / "best.pt", {
                "model": model.state_dict(),
                "meta": {"model_name": MODEL_NAME, "labels": list(LABELS), "epoch": epoch + 1,
                         "global_step": state["global_step"], "val_metrics": vm,
                         "image_size": ld["image_size"], "train": ts, "seed": seed}})
            log(f"  new best validation ROC-AUC -> {ckpt_dir / 'best.pt'}")
        save_last()

    result = {"global_step": state["global_step"], "epoch": state["epoch"], "stopped_early": stopped,
              "best_val_auc": state["best_val_auc"] if state["best_val_auc"] >= 0 else None,
              "steps_per_epoch": steps_per_epoch, "history": history}
    (run_dir / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
