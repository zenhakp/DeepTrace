"""Batch prediction over a loader: one record per image with the fake-class probability."""
from __future__ import annotations

from typing import Any, Dict, List

import torch


@torch.no_grad()
def predict_rows(model: torch.nn.Module, loader, device, amp: bool = False) -> List[Dict[str, Any]]:
    model.eval()
    dev = torch.device(device)
    out: List[Dict[str, Any]] = []
    for x, y, ids in loader:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            logits = model(x.to(dev))
        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
        out += [{"image_id": i, "label": int(l), "fake_prob": float(p)} for i, l, p in zip(ids, y.tolist(), probs)]
    return out
