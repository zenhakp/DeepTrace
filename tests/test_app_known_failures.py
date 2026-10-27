"""Real LLM failures (tests/data/llm_known_failures.json) that the step 4 validator must reject.

Until step 4 adds deeptrace.app.validate, these are expected failures (strict xfail on ImportError). When the module
exists they run for real: a case the validator lets through is then a test failure, and the xfail marker must be
removed (strict=True turns an unexpected pass into a failure, so it cannot be forgotten).
"""
import json
from pathlib import Path

import pytest

from deeptrace.app.digest import build_digest
from deeptrace.app.evidence import REPO_ROOT, load_evidence

CASES = json.loads((Path(__file__).parent / "data" / "llm_known_failures.json").read_text(encoding="utf-8"))
EXAMPLES = REPO_ROOT / "docs" / "part2_evidence_examples"


def test_known_failure_cases_are_well_formed():
    assert len({c["id"] for c in CASES}) == len(CASES)
    for c in CASES:
        dg = build_digest(load_evidence(EXAMPLES / f"{c['evidence']}.json"))
        assert c["section"] in [s.key for s in dg.sections]
        assert c["text"] and c["problem"] and c["source"]


@pytest.mark.xfail(strict=True, raises=ImportError, reason="step 4 adds deeptrace.app.validate")
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_step4_validator_rejects_known_failure(case):
    from deeptrace.app.validate import check_section  # step 4: returns a list of problems, empty if acceptable
    dg = build_digest(load_evidence(EXAMPLES / f"{case['evidence']}.json"))
    assert check_section(dg, case["section"], case["text"]), case["problem"]
