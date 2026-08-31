"""The frozen extraction reaches Stage 02 as input, and only where it resolved one value."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.starling.normalization.measurement_resolution import (
    apply_measurement_resolution,
    load_exact_unit_mapping,
)
from tools.chembl_tool.common.starling.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
)


def _mapping(tmp_path, rows):
    path = tmp_path / "measurement_resolution.parquet"
    table = (
        pa.Table.from_pylist(rows)
        if rows
        else pa.table(
            {
                "cleaned_record_id": pa.array([], type=pa.string()),
                "status": pa.array([], type=pa.string()),
                "measurements_json": pa.array([], type=pa.string()),
            }
        )
    )
    pq.write_table(table, path)
    return path


def _units(
    tmp_path,
    entries,
    version="starling_exact_measurement_units.v1",
):
    path = tmp_path / "exact_units.json"
    path.write_text(
        json.dumps({"version": version, "entries": entries})
    )
    return path


def _unit(endpoint, input_unit, canonical_unit, scale="1", domain="any"):
    return {
        "task": "test_task",
        "canonical_endpoint": endpoint,
        "input_unit": input_unit,
        "action": "map",
        "canonical_unit": canonical_unit,
        "scale": scale,
        "domain": domain,
    }


def test_grouped_v2_rules_expand_to_exact_keys(tmp_path) -> None:
    path = _units(
        tmp_path,
        [
            {
                "task": "test_task",
                "canonical_endpoints": ["pampa", "permeability"],
                "input_unit": "10^-6 cm/s",
                "action": "map",
                "canonical_unit": "cm/s",
                "scale": "0.000001",
                "domain": "nonnegative",
            }
        ],
        version="starling_exact_measurement_units.v2",
    )
    mapping = load_exact_unit_mapping(path)
    assert set(mapping) == {
        ("test_task", "pampa", "10^-6 cm/s"),
        ("test_task", "permeability", "10^-6 cm/s"),
    }
    assert all(rule["canonical_unit"] == "cm/s" for rule in mapping.values())


def _row(record_id, measurement, unit="raw unit"):
    return {
        "cleaned_record_id": record_id,
        "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
        "measurement_resolution_route": "extract",
        "measurement_text": measurement,
        "unit_text": unit,
        "canonical_endpoint_name": "permeability",
    }


def _resolution(record_id, status, measurements):
    return {
        "cleaned_record_id": record_id,
        "status": status,
        "measurements_json": json.dumps(measurements),
    }


def test_a_single_resolved_quantity_becomes_the_stage2_input(tmp_path) -> None:
    records = [_row("rec-1", "2.72 ± 0.03 × 10^-6")]
    audit = apply_measurement_resolution(
        records,
        mapping_path=_mapping(
            tmp_path,
            [_resolution("rec-1", "ok", [{"measurement": "2.72e-6", "unit": "cm/s"}])],
        ),
        task="test_task",
        unit_mapping_path=_units(
            tmp_path, [_unit("permeability", "cm/s", "cm/s")]
        ),
    )
    assert records[0]["measurement_text"] == "2.72 ± 0.03 × 10^-6"
    assert records[0]["unit_text"] == "raw unit"
    assert records[0]["resolved_measurement_text"] == "0.00000272"
    assert records[0]["resolved_unit_text"] == "cm/s"
    assert records[0]["pre_resolution_measurement_text"] == "2.72 ± 0.03 × 10^-6"
    assert records[0]["pre_resolution_unit_text"] == "raw unit"
    assert records[0]["measurement_resolution_status"] == "ok"
    assert audit["substituted_rows"] == 1


def test_source_exact_route_ignores_a_stale_llm_resolution(tmp_path) -> None:
    record = _row("rec-1", "5", "cm/h")
    record["measurement_resolution_route"] = "accept"
    records = [record]
    apply_measurement_resolution(
        records,
        mapping_path=_mapping(
            tmp_path,
            [_resolution("rec-1", "ok", [{"measurement": "99", "unit": "cm/s"}])],
        ),
        task="test_task",
        unit_mapping_path=_units(
            tmp_path,
            [_unit("permeability", "cm/h", "cm/s", "0.0002777777777777778")],
        ),
    )
    assert records[0]["measurement_resolution_origin"] == "source_exact"
    assert records[0]["measurement_resolution_input_measurement"] == "5"
    assert records[0]["resolved_measurement_text"].startswith("0.0013888")


def test_a_declared_canonical_source_exact_unit_bypasses_the_unit_map(tmp_path) -> None:
    record = _row("rec-1", "4/10 positive", None)
    record.update(
        {
            "measurement_resolution_route": "accept",
            "measurement_resolution_exact_measurement": "0.4",
            "measurement_resolution_exact_unit": "fraction",
            "measurement_resolution_exact_unit_is_canonical": True,
        }
    )
    records = [record]
    apply_measurement_resolution(
        records,
        mapping_path=_mapping(tmp_path, []),
        task="test_task",
        unit_mapping_path=_units(tmp_path, []),
    )
    assert records[0]["measurement_resolution_origin"] == "source_exact"
    assert records[0]["measurement_unit_mapping_status"] == "mapped"
    assert records[0]["resolved_scalar_value"] == pytest.approx(0.4)
    assert records[0]["resolved_unit_text"] == "fraction"


@pytest.mark.parametrize("route", ["categorical", "reject"])
def test_nonextract_routes_ignore_a_stale_llm_resolution(tmp_path, route) -> None:
    record = _row("rec-1", "substrate", "")
    record["measurement_resolution_route"] = route
    records = [record]
    audit = apply_measurement_resolution(
        records,
        mapping_path=_mapping(
            tmp_path,
            [_resolution("rec-1", "ok", [{"measurement": "99", "unit": "cm/s"}])],
        ),
        task="test_task",
        unit_mapping_path=_units(tmp_path, []),
    )
    assert records[0]["measurement_resolution_status"] == "not_extracted"
    assert records[0]["measurement_resolution_origin"] is None
    assert records[0].get("resolved_scalar_value") is None
    assert audit["substituted_rows"] == 0


def test_several_quantities_are_always_exploded(tmp_path) -> None:
    records = [_row("rec-1", "0.87%, 1.2%, and 804 ug/cm2")]
    entries = [
        {"measurement": "0.87", "unit": "%"},
        {"measurement": "1.2", "unit": "%"},
        {"measurement": "804", "unit": "ug/cm^2"},
    ]
    apply_measurement_resolution(
        records,
        mapping_path=_mapping(tmp_path, [_resolution("rec-1", "ok", entries)]),
        task="test_task",
        unit_mapping_path=_units(
            tmp_path,
            [
                _unit("permeability", "%", "%"),
                _unit("permeability", "ug/cm^2", "µg/cm^2"),
            ],
        ),
    )
    assert len(records) == 3
    assert [row["resolved_measurement_text"] for row in records] == [
        "0.87",
        "1.2",
        "804",
    ]
    assert len({row["cleaned_record_id"] for row in records}) == 3
    assert all(
        row["measurement_resolution_parent_cleaned_record_id"] == "rec-1"
        for row in records
    )
    assert [row["measurement_resolution_entry_index"] for row in records] == [0, 1, 2]
    assert all(
        row["measurement_text"] == "0.87%, 1.2%, and 804 ug/cm2"
        for row in records
    )


@pytest.mark.parametrize("status", ["unsure", "relative", "unavailable"])
def test_an_unresolved_status_leaves_the_source_columns_alone(tmp_path, status) -> None:
    records = [_row("rec-1", "up to 7")]
    apply_measurement_resolution(
        records,
        mapping_path=_mapping(tmp_path, [_resolution("rec-1", status, [])]),
        task="test_task",
        unit_mapping_path=_units(tmp_path, []),
    )
    assert records[0]["measurement_text"] == "up to 7"
    assert records[0]["measurement_resolution_status"] == status
    assert records[0]["pre_resolution_measurement_text"] == "up to 7"


def test_a_partial_extraction_is_refused_unless_waived(tmp_path) -> None:
    mapping = _mapping(
        tmp_path, [_resolution("rec-1", "ok", [{"measurement": "1", "unit": "%"}])]
    )
    records = [_row("rec-1", "1%"), _row("rec-2", "2%")]
    with pytest.raises(ValueError, match="no frozen resolution"):
        apply_measurement_resolution(
            records,
            mapping_path=mapping,
            task="test_task",
            unit_mapping_path=_units(
                tmp_path, [_unit("permeability", "%", "%")]
            ),
        )
    audit = apply_measurement_resolution(
        records,
        mapping_path=mapping,
        task="test_task",
        unit_mapping_path=_units(
            tmp_path, [_unit("permeability", "%", "%")]
        ),
        allow_partial=True,
    )
    assert audit["extract_rows_without_a_resolution"] == 1
    assert records[1]["measurement_resolution_status"] == "not_extracted"


def test_no_mapping_at_all_is_a_deliberate_source_only_build(tmp_path) -> None:
    records = [_row("rec-1", "45%")]
    audit = apply_measurement_resolution(records, mapping_path=None)
    assert audit["substituted_rows"] == 0
    assert audit["extract_rows_without_a_resolution"] == 0
    assert records[0]["measurement_text"] == "45%"
    assert records[0]["measurement_resolution_active"] is True


def test_an_explicitly_empty_mapping_does_not_bypass_completeness(tmp_path) -> None:
    records = [_row("rec-1", "45%")]
    with pytest.raises(ValueError, match="no frozen resolution"):
        apply_measurement_resolution(
            records,
            mapping_path=_mapping(tmp_path, []),
            task="test_task",
            unit_mapping_path=_units(tmp_path, []),
        )


def test_mapping_contract_rejects_duplicates_unknown_status_and_orphans(tmp_path) -> None:
    units = _units(tmp_path, [_unit("permeability", "%", "%")])
    duplicate = _resolution("rec-1", "ok", [{"measurement": "1", "unit": "%"}])
    with pytest.raises(ValueError, match="duplicate"):
        apply_measurement_resolution(
            [_row("rec-1", "1%")],
            mapping_path=_mapping(tmp_path, [duplicate, duplicate]),
            task="test_task",
            unit_mapping_path=units,
        )
    with pytest.raises(ValueError, match="unsupported.*status"):
        apply_measurement_resolution(
            [_row("rec-1", "1%")],
            mapping_path=_mapping(
                tmp_path, [_resolution("rec-1", "broken", [])]
            ),
            task="test_task",
            unit_mapping_path=units,
        )
    with pytest.raises(ValueError, match="absent from Stage 01"):
        apply_measurement_resolution(
            [_row("rec-1", "1%")],
            mapping_path=_mapping(
                tmp_path,
                [
                    duplicate,
                    _resolution(
                        "orphan", "ok", [{"measurement": "2", "unit": "%"}]
                    ),
                ],
            ),
            task="test_task",
            unit_mapping_path=units,
        )
    audit = apply_measurement_resolution(
        [_row("rec-1", "1%")],
        mapping_path=_mapping(
            tmp_path,
            [
                duplicate,
                _resolution("rejected", "ok", [{"measurement": "2", "unit": "%"}]),
            ],
        ),
        task="test_task",
        unit_mapping_path=units,
        ignored_record_ids={"rejected"},
    )
    assert audit["ignored_structure_rejection_rows"] == 1


def test_mapping_contract_rejects_routing_drift_and_float_overflow(tmp_path) -> None:
    drifted = _row("rec-1", "1%")
    drifted["measurement_routing_version"] = "starling_measurement_routing.v2"
    with pytest.raises(ValueError, match="routing version drift"):
        apply_measurement_resolution(
            [drifted],
            mapping_path=_mapping(
                tmp_path,
                [_resolution("rec-1", "ok", [{"measurement": "1", "unit": "%"}])],
            ),
            task="test_task",
            unit_mapping_path=_units(tmp_path, [_unit("permeability", "%", "%")]),
        )
    with pytest.raises(ValueError, match="overflows float"):
        apply_measurement_resolution(
            [_row("rec-1", "large")],
            mapping_path=_mapping(
                tmp_path,
                [
                    _resolution(
                        "rec-1", "ok", [{"measurement": "1e9999", "unit": "%"}]
                    )
                ],
            ),
            task="test_task",
            unit_mapping_path=_units(tmp_path, [_unit("permeability", "%", "%")]),
        )


def test_exact_scale_uses_plain_decimal_and_domain_fails_closed(tmp_path) -> None:
    mapping = _mapping(
        tmp_path,
        [
            _resolution(
                "rec-1", "ok", [{"measurement": "5.6", "unit": "10^-3 cm/h"}]
            ),
            _resolution(
                "rec-2", "ok", [{"measurement": "-6", "unit": "cm/h"}]
            ),
        ],
    )
    units = _units(
        tmp_path,
        [
            _unit(
                "permeability",
                "10^-3 cm/h",
                "cm/s",
                "0.0000002777777777777777777778",
                "nonnegative",
            ),
            _unit("permeability", "cm/h", "cm/s", "0.0002777777777777778", "nonnegative"),
        ],
    )
    records = [_row("rec-1", "source"), _row("rec-2", "source")]
    apply_measurement_resolution(
        records,
        mapping_path=mapping,
        task="test_task",
        unit_mapping_path=units,
    )
    assert records[0]["resolved_measurement_text"].startswith("0.0000015555")
    assert records[0]["resolved_unit_text"] == "cm/s"
    assert records[1]["measurement_unit_mapping_status"] == "mapped"
    assert records[1]["measurement_numeric_domain_status"] == "outside_declared_domain"
    assert records[1]["resolved_scalar_value"] < 0


def test_missing_exact_unit_rule_is_a_hard_error(tmp_path) -> None:
    records = [_row("rec-1", "1")]
    with pytest.raises(ValueError, match="has no rule"):
        apply_measurement_resolution(
            records,
            mapping_path=_mapping(
                tmp_path,
                [_resolution("rec-1", "ok", [{"measurement": "1", "unit": "odd"}])],
            ),
            task="test_task",
            unit_mapping_path=_units(tmp_path, []),
        )


def test_legacy_unit_exclusion_cannot_discard_a_resolved_scalar(tmp_path) -> None:
    excluded = {
        "task": "test_task",
        "canonical_endpoint": "permeability",
        "input_unit": "ratio",
        "action": "exclude",
    }
    with pytest.raises(ValueError, match="still excludes a resolved scalar"):
        apply_measurement_resolution(
            [_row("rec-1", "0.4")],
            mapping_path=_mapping(
                tmp_path,
                [
                    _resolution(
                        "rec-1",
                        "ok",
                        [{"measurement": "0.4", "unit": "ratio"}],
                    )
                ],
            ),
            task="test_task",
            unit_mapping_path=_units(tmp_path, [excluded]),
        )
