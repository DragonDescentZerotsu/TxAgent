"""Executable coverage and value-invariance checks for DILI V10 units."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.dili.data_processing import (
    run_unit_reconciliation_review as review_runner,
)
from data.processing.evidence_library.versions.v10.tasks.dili.data_processing.build_unit_reconciliation import (
    build_inventory,
    build_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.dili.data_processing.run_unit_reconciliation_review import (
    _parse_decisions,
    _validate_equivalent,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    cleaned = tmp_path / "cleaned.parquet"
    rows = [
        {
            "cleaned_record_id": "a",
            "source_row_uid": "uid-a",
            "source_id": "dili_v5",
            "endpoint_name": "potency",
            "measurement_text": "EC50: 2",
            "unit_text": "uM",
            "support_text": "EC50 was 2 uM.",
            "measurement_resolution_route": "accept",
            "measurement_resolution_exact_measurement": "2",
            "measurement_resolution_exact_unit": "uM",
        },
        {
            "cleaned_record_id": "b",
            "source_row_uid": "uid-b",
            "source_id": "dili_v1",
            "endpoint_name": "potency",
            "measurement_text": "EC50: 2.42E-6",
            "unit_text": "mol/L",
            "support_text": "EC50 was reported.",
            "measurement_resolution_route": "extract",
            "measurement_resolution_exact_measurement": None,
            "measurement_resolution_exact_unit": None,
        },
        {
            "cleaned_record_id": "c",
            "source_row_uid": "uid-c",
            "source_id": "dili_v2",
            "endpoint_name": "signal",
            "measurement_text": "7 pg/ml",
            "unit_text": None,
            "support_text": "The signal was 7 pg/ml.",
            "measurement_resolution_route": "extract",
            "measurement_resolution_exact_measurement": None,
            "measurement_resolution_exact_unit": None,
        },
        {
            "cleaned_record_id": "d",
            "source_row_uid": "uid-d",
            "source_id": "dili_v3",
            "endpoint_name": "invalid",
            "measurement_text": "5",
            "unit_text": "not a unit",
            "support_text": "Malformed source field.",
            "measurement_resolution_route": "reject",
            "measurement_resolution_exact_measurement": None,
            "measurement_resolution_exact_unit": None,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), cleaned)
    resolution = tmp_path / "resolution.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cleaned_record_id": "b",
                    "source_row_uid": "uid-b",
                    "source_id": "dili_v1",
                    "status": "ok",
                    "measurements_json": json.dumps(
                        [{"measurement": "2.42", "unit": "10^-6 mol/L"}]
                    ),
                },
                {
                    "cleaned_record_id": "c",
                    "source_row_uid": "uid-c",
                    "source_id": "dili_v2",
                    "status": "ok",
                    "measurements_json": json.dumps(
                        [{"measurement": "7", "unit": "pg/ml"}]
                    ),
                },
            ]
        ),
        resolution,
    )
    return cleaned, resolution


def test_inventory_unions_source_exact_and_llm_units(tmp_path: Path) -> None:
    cleaned, resolution = _inputs(tmp_path)
    inventory = build_inventory(cleaned, resolution, prior_paths={})
    by_unit = {row["input_unit"]: row for row in inventory["items"]}
    assert set(by_unit) == {"uM", "mol/L", "10^-6 mol/L", "pg/ml", "not a unit"}
    assert inventory["counts"] == {
        "units": 5,
        "source_occurrences": 3,
        "source_exact_occurrences": 1,
        "llm_ok_occurrences": 2,
    }
    assert by_unit["uM"]["origin_counts"] == {"source": 1, "source_exact": 1}
    assert by_unit["10^-6 mol/L"]["origin_counts"] == {"llm": 1}


def test_inventory_fails_on_uid_drift(tmp_path: Path) -> None:
    cleaned, resolution = _inputs(tmp_path)
    rows = pq.read_table(resolution).to_pylist()
    rows[0]["source_row_uid"] = "wrong"
    pq.write_table(pa.Table.from_pylist(rows), resolution)
    with pytest.raises(ValueError, match="source mismatch"):
        build_inventory(cleaned, resolution, prior_paths={})


def test_mapping_is_complete_wildcard_and_coefficient_preserving(tmp_path: Path) -> None:
    cleaned, resolution = _inputs(tmp_path)
    inventory = build_inventory(cleaned, resolution, prior_paths={})
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    manifest_path = tmp_path / "inventory.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "dili_unit_review_packets.v1",
                "task": "dili",
                "inventory": {
                    "path": str(inventory_path),
                    "sha256": file_sha256(inventory_path),
                },
                "inputs": inventory["inputs"],
                "item_count": 5,
                "packet_size": 5,
                "packet_count": 1,
            }
        )
    )
    targets = {
        "uM": ("map", "µM"),
        "mol/L": ("map", "M"),
        "10^-6 mol/L": ("map", "µM"),
        "pg/ml": ("map", "pg/mL"),
        "not a unit": ("exclude", ""),
    }
    review_path = tmp_path / "review.jsonl"
    review_path.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item["item_id"],
                    "input_unit": item["input_unit"],
                    "action": targets[item["input_unit"]][0],
                    "canonical_unit": targets[item["input_unit"]][1],
                    "rationale": "Reviewed fixture decision",
                    "review_phase": "primary_and_adjudication",
                }
            )
            + "\n"
            for item in inventory["items"]
        )
    )
    receipt_path = tmp_path / "review.manifest.json"
    receipt_path.write_text(
        json.dumps(
            {
                "version": "dili_unit_review_run.v1",
                "review_sha256": file_sha256(review_path),
                "inventory_manifest_sha256": file_sha256(manifest_path),
                "route": {"model": "fixture"},
            }
        )
    )
    mapping, report = build_mapping(manifest_path, review_path, receipt_path)
    output = tmp_path / "mapping.json"
    output.write_text(json.dumps(mapping))
    rules = load_exact_unit_mapping(output)
    assert len(rules) == 5
    assert rules[("dili", "*", "10^-6 mol/L")]["canonical_unit"] == "µM"
    assert rules[("dili", "*", "10^-6 mol/L")]["scale"] == "1"
    assert rules[("dili", "*", "not a unit")]["action"] == "exclude"
    assert report["counts"]["units"] == 5


def test_equivalent_display_validation_preserves_scale_and_semantics() -> None:
    for source, target in (
        ("uM", "µM"),
        ("umol_per_l", "µM"),
        ("10^-6 mol/L", "µM"),
        ("10^-3 umol_per_l", "nM"),
        ("pg/ml", "pg/mL"),
    ):
        _validate_equivalent(source, target)
    with pytest.raises(ValueError, match="coefficient or meaning"):
        _validate_equivalent("mM", "µM")
    with pytest.raises(ValueError, match="semantic markers"):
        _validate_equivalent("percent_of_control", "%")
    with pytest.raises(ValueError, match="semantic markers"):
        _validate_equivalent("fraction", "%")


def test_unsafe_alias_is_preserved_as_source_identity() -> None:
    item = {"item_id": "u_test", "input_unit": "mM"}
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "decisions": [
                                {
                                    "item_id": "u_test",
                                    "action": "map",
                                    "canonical_unit": "µM",
                                    "rationale": "Incorrect model proposal",
                                }
                            ]
                        }
                    )
                }
            }
        ]
    }
    decision = _parse_decisions(response, [item])[0]
    assert decision["canonical_unit"] == "mM"
    assert decision["guard_adjustment"] == "unsafe_alias_to_identity"


def test_failed_review_packet_splits_without_losing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [
        {"item_id": "u_a", "input_unit": "µM"},
        {"item_id": "u_b", "input_unit": "nM"},
    ]

    def call_packet(*args, **kwargs):
        packet_items = args[5]
        if len(packet_items) > 1:
            raise ValueError("packet-level response mismatch")
        item = packet_items[0]
        return {}, [
            {
                "item_id": item["item_id"],
                "input_unit": item["input_unit"],
                "action": "map",
                "canonical_unit": item["input_unit"],
                "rationale": "Reviewed singleton",
                "guard_adjustment": "",
            }
        ]

    monkeypatch.setattr(review_runner, "_call_packet", call_packet)
    decisions = review_runner._resolve_packet(
        phase="primary",
        packet_id="primary-00000",
        items=items,
        client=object(),
        cache_path=tmp_path / "cache.jsonl",
        manifest_sha256="fixture",
        cached={},
        rejected=set(),
        max_tokens=100,
    )
    assert [row["item_id"] for row in decisions] == ["u_a", "u_b"]


def test_failed_adjudication_singleton_uses_no_reasoning_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = {"item_id": "u_a", "input_unit": "µM"}
    phases = []

    def call_packet(*args, **kwargs):
        phases.append(args[3])
        return {}, [
            {
                "item_id": "u_a",
                "input_unit": "µM",
                "action": "map",
                "canonical_unit": "µM",
                "rationale": "Reviewed singleton",
                "guard_adjustment": "",
            }
        ]

    monkeypatch.setattr(review_runner, "_call_packet", call_packet)
    decision = review_runner._resolve_packet(
        phase="adjudication",
        packet_id="adjudication-00000.a.a.a.a",
        items=[item],
        client=object(),
        cache_path=tmp_path / "cache.jsonl",
        manifest_sha256="fixture",
        cached={},
        rejected={("adjudication", "adjudication-00000.a.a.a.a")},
        max_tokens=100,
    )[0]
    assert phases == ["adjudication_fallback"]
    assert decision["item_id"] == "u_a"
    request = review_runner._request(
        "adjudication_fallback", "packet", [item], max_tokens=100
    )
    assert request["reasoning_effort"] == "none"
