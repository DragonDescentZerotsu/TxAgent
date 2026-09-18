"""Focused checks for the two-pass Skin unit review."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing import (
    review_unit_reconciliation_v4 as review,
)


def _response(model: str, decisions: list[dict[str, str]]) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": json.dumps({"decisions": decisions})}}],
    }


def test_first_pass_packets_are_fifty_plus_one_tail() -> None:
    packets = review._packets([{"item_id": str(i)} for i in range(6321)], "first")
    assert len(packets) == 127
    assert all(len(items) == 50 for _, items in packets[:-1])
    assert len(packets[-1][1]) == 21


def test_explicit_denominator_keeps_qualifier_and_normalizes_scale() -> None:
    item = {"item_id": "u1", "input_unit": "# positive reactions out of 3"}
    response = _response(
        review.MODEL,
        [
            {
                "item_id": "u1",
                "action": "map",
                "canonical_unit": "positive fraction",
                "scale": "0.3333333333333333",
                "rationale": "Explicit positive count denominator.",
            }
        ],
    )
    decision = review._parse_decisions(response, [item], "first")[0]
    assert decision["canonical_unit"] == "positive fraction"
    assert decision["scale"] == "0.3333333333333333333333333333"
    assert review._validate_scale("events out of 10", "1/10") == "0.1"
    assert review._ratio_denominator("positive/10") == 10
    assert review._ratio_denominator("lesions per 8 mice") == 8
    assert review._ratio_denominator("patients (total N=10)") == 10
    assert review._ratio_denominator("reaction in 50 subjects") == 50
    assert review._ratio_denominator("primary irritation index out of 8.00") is None
    assert review._ratio_denominator("molecules/10^4 nucleotides") is None
    with pytest.raises(ValueError, match="neutral fraction"):
        review._validate_alias("# observations out of 10", "positive fraction")

    corrected = review._parse_decisions(
        _response(
            review.MODEL,
            [
                {
                    "item_id": "u2",
                    "action": "map",
                    "canonical_unit": "positive fraction",
                    "scale": "1/10",
                    "rationale": "Over-qualified proposal.",
                }
            ],
        ),
        [{"item_id": "u2", "input_unit": "# observations out of 10"}],
        "first",
    )[0]
    assert corrected["canonical_unit"] == "neutral fraction"
    assert corrected["guard_adjustment"] == (
        "ratio_qualifier_corrected+scale_normalized"
    )


def test_known_physical_scale_change_is_rejected() -> None:
    with pytest.raises(ValueError, match="coefficient or meaning"):
        review._validate_alias("nM", "mM")
    review._validate_alias("nmol/L", "nM")
    review._validate_alias("ng/cm2", "ng/cm²")


def test_unsafe_model_alias_is_preserved_as_identity() -> None:
    item = {"item_id": "u1", "input_unit": "nM"}
    response = _response(
        review.MODEL,
        [
            {
                "item_id": "u1",
                "action": "map",
                "canonical_unit": "mM",
                "scale": "1",
                "rationale": "Unsafe proposal.",
            }
        ],
    )
    decision = review._parse_decisions(response, [item], "first")[0]
    assert decision["canonical_unit"] == "nM"
    assert decision["guard_adjustment"] == "unsafe_alias_to_identity"


def test_second_retry_uses_gpt_54_mini(tmp_path: Path) -> None:
    requests: list[dict] = []

    class Completions:
        def __init__(self, fallback: bool) -> None:
            self.fallback = fallback

        def create(self, **request):
            requests.append(request)
            if not self.fallback:
                return {"model": review.MODEL, "choices": []}
            return _response(
                review.FALLBACK_MODEL,
                [
                    {
                        "item_id": "u1",
                        "action": "map",
                        "canonical_unit": "mg/mL",
                        "scale": "1",
                        "rationale": "Identity unit.",
                    }
                ],
            )

    class Client:
        def __init__(self, fallback: bool) -> None:
            self.chat = type("Chat", (), {"completions": Completions(fallback)})()

    item = {"item_id": "u1", "input_unit": "mg/mL"}
    decisions = review._call_packet(
        Client(False),
        Client(True),
        tmp_path / "cache.jsonl",
        "manifest-hash",
        "first",
        "first-00000",
        [item],
        {},
    )
    assert decisions[0]["canonical_unit"] == "mg/mL"
    assert [request["model"] for request in requests] == [
        review.MODEL,
        review.MODEL,
        review.FALLBACK_MODEL,
    ]
    assert all(request["reasoning_effort"] == "high" for request in requests)
    assert requests[-1]["max_completion_tokens"] == 128_000
    assert "max_tokens" not in requests[-1]


def test_v4_wildcard_mapping_is_loadable(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps(
            {
                "version": review.MAPPING_VERSION,
                "entries": [
                    {
                        "task": "skin_reaction",
                        "canonical_endpoints": ["*"],
                        "input_unit": "nmol/L",
                        "action": "map",
                        "canonical_unit": "nM",
                        "scale": "1",
                        "domain": "any",
                    }
                ],
            }
        )
    )
    mapping = load_exact_unit_mapping(path)
    assert mapping[("skin_reaction", "*", "nmol/L")]["canonical_unit"] == "nM"
