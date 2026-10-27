"""Real LLM answers: known failures must be rejected, known good answers must pass (no over-strict validator).

tests/data/llm_known_failures.json: wrong (or, where its "problem" says so, rejected by design) section texts.
tests/data/llm_known_good.json: accurate section texts from the step 3 live run, checked by reading them.
Both hold the exact text the models wrote; add every new real failure here before changing a rule.
"""
import json
from pathlib import Path

import pytest

from deeptrace.app.digest import build_digest
from deeptrace.app.evidence import REPO_ROOT, load_evidence
from deeptrace.app.validate import check_section

DATA = Path(__file__).parent / "data"
FAILURES = json.loads((DATA / "llm_known_failures.json").read_text(encoding="utf-8"))
GOOD = json.loads((DATA / "llm_known_good.json").read_text(encoding="utf-8"))
EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"


def _digest(name):
    return build_digest(load_evidence(EXAMPLES / f"{name}.json"))


def test_cases_are_well_formed():
    for cases in (FAILURES, GOOD):
        assert len({c["id"] for c in cases}) == len(cases)
        for c in cases:
            assert c["section"] in [s.key for s in _digest(c["evidence"]).sections] and c["text"] and c["source"]


@pytest.mark.parametrize("case", FAILURES, ids=[c["id"] for c in FAILURES])
def test_known_failure_is_rejected(case):
    assert check_section(_digest(case["evidence"]), case["section"], case["text"]), case["problem"]


@pytest.mark.parametrize("case", GOOD, ids=[c["id"] for c in GOOD])
def test_known_good_answer_passes(case):
    assert check_section(_digest(case["evidence"]), case["section"], case["text"]) == []
