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
    packets = review._packets([{"item_id": str(i)} for i in range(6322)], "first")
    assert len(packets) == 127
    assert all(len(items) == 50 for _, items in packets[:-1])
    assert len(packets[-1][1]) == 22


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
    assert review._ratio_denominator("positive/17 patients") == 17
    assert review._ratio_denominator("positive/18 tests") == 18
    assert review._ratio_denominator("proportion (1/52)") == 52
    assert review._ratio_denominator("lesions per 8 mice") == 8
    assert review._ratio_denominator("patients (total N=10)") == 10
    assert review._ratio_denominator("reaction in 50 subjects") == 50
    assert review._ratio_denominator("/10 animals") == 10
    assert review._ratio_denominator("primary irritation index out of 8.00") is None
    assert review._ratio_denominator("molecules/10^4 nucleotides") is None
    assert review._ratio_denominator("µg/cm^2/h^(1/2)") is None
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


@pytest.mark.parametrize(
    ("unit", "scale"),
    [
        ("positive/17 patients", "0.05882352941176470588235294118"),
        ("positive/18 tests", "0.05555555555555555555555555556"),
        ("positive/25 subjects", "0.04"),
        ("positive/62 patients", "0.01612903225806451612903225806"),
        ("proportion (1/52)", "0.01923076923076923076923076923"),
    ],
)
def test_v5_suffix_and_parenthesized_denominators(unit: str, scale: str) -> None:
    assert review._validate_scale(unit, "1") == scale


def test_known_physical_scale_change_is_rejected() -> None:
    with pytest.raises(ValueError, match="coefficient or meaning"):
        review._validate_alias("nM", "mM")
    review._validate_alias("nmol/L", "nM")
    review._validate_alias("ng/cm2", "ng/cm²")


def test_skin_exposure_kg_spelling_means_body_weight() -> None:
    review._validate_alias("mg/kg KG", "mg/kg body weight")
    with pytest.raises(ValueError, match="coefficient or meaning"):
        review._validate_alias("mg/kg KG", "mg/kg/day")


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


def test_terminal_fallback_failure_resumes_at_gpt_retry(tmp_path: Path) -> None:
    cache = tmp_path / "cache.jsonl"
    cache.write_text(
        "\n".join(
            json.dumps(
                {
                    "inventory_manifest_sha256": "manifest",
                    "phase": "first",
                    "packet_id": "first-00000",
                    "attempt": attempt,
                    "status": "rejected",
                    "reason": f"failure {attempt}",
                }
            )
            for attempt in (1, 2, 3)
        )
        + "\n"
    )
    assert review._resume_attempt(cache, "manifest", "first", "first-00000") == (
        4,
        "failure 3",
    )
    with cache.open("a", encoding="utf-8") as stream:
        for attempt in (4, 5):
            stream.write(
                json.dumps(
                    {
                        "inventory_manifest_sha256": "manifest",
                        "phase": "first",
                        "packet_id": "first-00000",
                        "attempt": attempt,
                        "status": "rejected",
                        "reason": "RateLimitError: exhausted",
                    }
                )
                + "\n"
            )
    assert review._resume_attempt(cache, "manifest", "first", "first-00000") == (
        6,
        "failure 3",
    )


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


def test_v5_map_contains_only_current_wildcard_units(tmp_path: Path, monkeypatch) -> None:
    inventory = {
        "items": [{"item_id": "u1", "input_unit": "nmol/L"}],
        "scope": {},
        "inputs": {},
        "counts": {},
    }
    manifest = tmp_path / "inventory.manifest.json"
    manifest.write_text(json.dumps({"inventory": inventory}))
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        json.dumps(
            {
                "item_id": "u1",
                "input_unit": "nmol/L",
                "action": "map",
                "canonical_unit": "nmol/L",
                "scale": "1",
            }
        )
        + "\n"
    )
    receipt = {
        "version": review.RECEIPT_VERSION,
        "review_sha256": review.file_sha256(review_path),
        "route": {},
        "counts": {},
    }
    review_path.with_suffix(".manifest.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(review, "load_inventory", lambda _: ({}, inventory))
    payload, report = review.build_mapping(manifest, review_path)
    assert [row["input_unit"] for row in payload["entries"]] == ["nmol/L"]
    assert payload["entries"][0]["canonical_endpoints"] == ["*"]
    assert payload["skin_v10_unit_review"]["validations"]["no_predecessor_entries"] is True


def test_v5_applies_validated_canonical_merges_without_changing_scale(
    tmp_path: Path, monkeypatch
) -> None:
    inventory = {
        "items": [
            {"item_id": "u1", "input_unit": "nmol/L"},
            {"item_id": "u2", "input_unit": "nM"},
        ],
        "scope": {},
        "inputs": {},
        "counts": {},
    }
    manifest = tmp_path / "inventory.manifest.json"
    manifest.write_text(json.dumps({"inventory": inventory}))
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "item_id": "u1",
                    "input_unit": "nmol/L",
                    "action": "map",
                    "canonical_unit": "nmol/L",
                    "scale": "1",
                },
                {
                    "item_id": "u2",
                    "input_unit": "nM",
                    "action": "map",
                    "canonical_unit": "nM",
                    "scale": "1",
                },
            )
        )
        + "\n"
    )
    review_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "version": review.RECEIPT_VERSION,
                "review_sha256": review.file_sha256(review_path),
                "route": {},
                "counts": {},
            }
        )
    )
    merges = tmp_path / "merges.json"
    merges.write_text(
        json.dumps(
            {
                "version": review.MERGE_REVIEW_VERSION,
                "task": "skin_reaction",
                "accepted_merges": [
                    {
                        "source_canonical_units": ["nM", "nmol/L"],
                        "target_canonical_unit": "nM",
                        "rationale": "Equivalent amount concentration units.",
                    }
                ],
                "result_counts": {
                    "source_labels_in_decisions": 2,
                    "canonical_labels_before": 2,
                    "canonical_labels_after_reviewed_merges": 1,
                    "canonical_label_reduction": 1,
                },
                "second_global_pass": {"status": "complete"},
                "validations": {
                    "all_3359_labels_reviewed": True,
                    "no_endpoint_specific_decisions": True,
                    "no_source_in_multiple_merge_decisions": True,
                    "targets_are_existing_reviewed_labels": True,
                    "second_global_pass_complete": True,
                    "merge_decisions_do_not_change_scale": True,
                    "post_review_denominator_guard_complete": True,
                },
            }
        )
    )
    monkeypatch.setattr(review, "load_inventory", lambda _: ({}, inventory))
    payload, report = review.build_mapping(manifest, review_path, merges)
    assert {entry["canonical_unit"] for entry in payload["entries"]} == {"nM"}
    assert {entry["scale"] for entry in payload["entries"]} == {"1"}
    assert report["counts"]["canonical_units_after_reviewed_merge"] == 1


def test_v5_rejects_duplicate_merge_sources(tmp_path: Path) -> None:
    path = tmp_path / "merges.json"
    path.write_text(
        json.dumps(
            {
                "version": review.MERGE_REVIEW_VERSION,
                "task": "skin_reaction",
                "accepted_merges": [
                    {
                        "source_canonical_units": ["nM", "nmol/L"],
                        "target_canonical_unit": "nM",
                        "rationale": "Equivalent.",
                    },
                    {
                        "source_canonical_units": ["nM", "nmol/l"],
                        "target_canonical_unit": "nM",
                        "rationale": "Equivalent.",
                    },
                ],
                "result_counts": {"canonical_labels_before": 3},
                "second_global_pass": {"status": "complete"},
                "validations": {
                    "all_3359_labels_reviewed": True,
                    "no_endpoint_specific_decisions": True,
                    "no_source_in_multiple_merge_decisions": True,
                    "targets_are_existing_reviewed_labels": True,
                    "second_global_pass_complete": True,
                    "merge_decisions_do_not_change_scale": True,
                    "post_review_denominator_guard_complete": True,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate.*source"):
        review._load_canonical_merge_review(path, {"nM", "nmol/L", "nmol/l"})
