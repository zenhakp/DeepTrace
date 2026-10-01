"""Classification metrics in pure Python. Positive class = MANIPULATED (label 1)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from deeptrace.data.preprocess import auc as roc_auc   # rank-based AUC, ties count half


def confusion(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for label, p in zip(labels, probs):
        pred = p >= threshold
        if label == 1 and pred:
            tp += 1
        elif label == 1:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else a / b


def classification_metrics(labels: Sequence[int], probs: Sequence[float], threshold: float = 0.5) -> Dict[str, Any]:
    if len(labels) != len(probs) or not len(labels):
        raise ValueError("labels and probs must be non-empty and of equal length")
    c = confusion(labels, probs, threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = len(labels)
    precision, recall, spec = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    f1 = None if precision is None or recall is None or precision + recall == 0 \
        else 2 * precision * recall / (precision + recall)
    return {
        "n": n, "n_real": tn + fp, "n_manipulated": tp + fn, "threshold": threshold,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": None if recall is None or spec is None else (recall + spec) / 2,
        "precision": precision, "recall": recall, "specificity": spec, "f1": f1,
        "roc_auc": roc_auc(list(labels), list(probs)),
        "confusion": c,
    }


def per_method_report(records: Sequence[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, Any]:
    """Share predicted MANIPULATED per image source. For 'real' this is the false-positive rate;
    for each fake method it is the recall on that method."""
    by: Dict[str, List[bool]] = defaultdict(list)
    for r in records:
        by[r["image_id"].split("/")[0]].append(r["fake_prob"] >= threshold)
    return {m: {"n": len(v), "share_predicted_manipulated": round(sum(v) / len(v), 4)}
            for m, v in sorted(by.items())}
