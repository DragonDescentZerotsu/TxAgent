"""Golden regression baseline for the unit normalizer.

`fixtures/units_golden.json` freezes the normalized output for the top-frequency real
Starling unit forms (top ~150 per source), reviewed once via the audit
(`units_audit.py`). This test fails if any change alters a verified mapping -- guarding
against silent drift or regressions in `clean_unit` / `canonicalize_unit`.

These entries are produced by `canonicalize_unit(raw)` with no task, so `unknown_tokens`
here reflects the **shared** qualifier vocabulary only. A token scoped to one task in
`common/qualifier_vocabulary_policy.json` (`skin`, `applied`, `brain`, ...) stays flagged in
this fixture by design -- that is the scoping working, not under-normalization. Assertions
about task-scoped vocabulary belong in `test_units.py` or the regression corpus, which call
with an explicit task.

To regenerate after an intentional change (and re-review):
    conda run --no-capture-output -n txagent-glm \
        python tests/chembl_tool/common/units_audit.py --write-golden
"""

import json
from pathlib import Path

from tools.chembl_tool.common.units import canonicalize_unit

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "units_golden.json"


def _load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_fixture_is_present_and_substantial():
    golden = _load_golden()
    assert len(golden) >= 300  # top real forms across six sources


def test_normalizer_matches_golden_baseline():
    golden = _load_golden()
    mismatches = []
    for raw, expected in golden.items():
        result = canonicalize_unit(raw)
        actual = {
            "cleaned": result.cleaned,
            "canonical": result.canonical,
            # JSON has no tuples: compare dimension/unknowns as lists.
            "dimension": [list(pair) for pair in result.dimension],
            "scale": result.scale,
            "transform": result.transform,
            "unknown_tokens": list(result.unknown_tokens),
        }
        expected_norm = {
            "cleaned": expected["cleaned"],
            "canonical": expected["canonical"],
            "dimension": [list(pair) for pair in expected["dimension"]],
            "scale": expected["scale"],
            "transform": expected["transform"],
            "unknown_tokens": list(expected["unknown_tokens"]),
        }
        if actual != expected_norm:
            mismatches.append((raw, expected_norm, actual))
    assert not mismatches, (
        f"{len(mismatches)} golden mismatch(es); first: "
        f"{mismatches[0] if mismatches else ''}"
    )
