"""Generator-family analysis. Family = the first word of the generator name (flux-2-klein-9b -> flux).
A family is 'seen' if any generator of the training data shares it. Crude by design: it is simple and reproducible."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from .metrics import roc_auc


def family_of(method: str) -> str:
    return str(method).split("-")[0]


def seen_families(train_methods: Iterable[str]) -> set:
    return {family_of(m) for m in train_methods}


def _rate(flags) -> Any:
    return round(sum(flags) / len(flags), 4) if flags else None


def family_report(records: Sequence[Dict[str, Any]], seen: set, threshold: float = 0.5) -> Dict[str, Any]:
    real_probs = [r["fake_prob"] for r in records if r["label"] == 0]
    per_gen: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, list] = {"seen_family": [], "unseen_family": []}
    for r in records:
        if r["label"] != 1:
            continue
        m = r["image_id"].split("/")[0]
        fam = family_of(m)
        d = per_gen.setdefault(m, {"family": fam, "family_seen_in_training": fam in seen, "flags": []})
        d["flags"].append(r["fake_prob"] >= threshold)
        groups["seen_family" if fam in seen else "unseen_family"].append(r["fake_prob"])

    def auc_vs_real(fake_probs):
        if not fake_probs or not real_probs:
            return None
        a = roc_auc([0] * len(real_probs) + [1] * len(fake_probs), real_probs + fake_probs)
        return None if a is None else round(a, 4)

    return {
        "per_generator": {m: {"n": len(d["flags"]), "recall": _rate(d["flags"]), "family": d["family"],
                              "family_seen_in_training": d["family_seen_in_training"]}
                          for m, d in sorted(per_gen.items())},
        "by_group": {g: {"n": len(p), "recall": _rate([x >= threshold for x in p]),
                         "roc_auc_vs_all_real": auc_vs_real(p)} for g, p in groups.items()},
    }
