from __future__ import annotations

import pytest

from tools.chembl_tool.common.starling.review_exact_measurement_units import (
    _remove_exact_keys,
    clean_whitespace,
    find_whitespace_alias,
    validate_decisions,
)


def test_whitespace_alias_requires_one_existing_mapped_rule() -> None:
    rules = {
        ("task", "endpoint", "µg /  cm²"): {
            "action": "map",
            "canonical_unit": "µg/cm²",
            "scale": "1",
            "domain": "nonnegative",
        }
    }
    assert clean_whitespace("  µg\t/  cm² ") == "µg / cm²"
    assert find_whitespace_alias(("task", "endpoint", "µg / cm²"), rules) == next(
        iter(rules.values())
    )


def test_review_decisions_preserve_or_use_the_existing_pool() -> None:
    expected = {
        ("task", "endpoint", "new literal"),
        ("task", "endpoint", "ug/ml"),
    }
    decisions = [
        {
            "task": "task",
            "canonical_endpoint": "endpoint",
            "input_unit": "new literal",
            "decision": "preserve_new",
            "canonical_unit": "new literal",
            "scale": "1",
            "rationale": "The literal reference basis is distinct.",
        },
        {
            "task": "task",
            "canonical_endpoint": "endpoint",
            "input_unit": "ug/ml",
            "decision": "integrate",
            "canonical_unit": "µg/mL",
            "scale": "1",
            "rationale": "ASCII spelling of the existing unit.",
        },
    ]
    assert len(validate_decisions(expected, decisions, {"µg/mL"})) == 2

    decisions[0]["canonical_unit"] = "different"
    with pytest.raises(ValueError, match="invalid preserve_new"):
        validate_decisions(expected, decisions, {"µg/mL"})


def test_replacing_one_grouped_exclusion_keeps_the_other_endpoint() -> None:
    payload = {
        "version": "starling_exact_measurement_units.v2",
        "entries": [
            {
                "task": "task",
                "canonical_endpoints": ["first", "second"],
                "input_unit": "%",
                "action": "exclude",
                "canonical_unit": None,
                "scale": None,
            }
        ],
    }
    result = _remove_exact_keys(payload, {("task", "first", "%")})
    assert result["entries"][0]["canonical_endpoints"] == ["second"]
