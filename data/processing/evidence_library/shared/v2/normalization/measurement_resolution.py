"""Join frozen measurement extraction and exact unit decisions into Stage 02."""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
    stable_id,
)

RESOLUTION_APPLY_VERSION = "starling_measurement_resolution_apply.v6"
EXACT_UNIT_MAPPING_VERSION = "starling_exact_measurement_units.v2"
GROUPED_EXACT_UNIT_MAPPING_VERSIONS = frozenset(
    {
        EXACT_UNIT_MAPPING_VERSION,
        "starling_exact_measurement_units.v3",
        "starling_exact_measurement_units.v4",
        "starling_exact_measurement_units.v5",
    }
)
_LEGACY_EXACT_UNIT_MAPPING_VERSION = "starling_exact_measurement_units.v1"
DEFAULT_EXPECTED_ROUTING_VERSION = "starling_measurement_routing.v5"
DEFAULT_EXACT_UNIT_MAPPING = (
    Path(__file__).resolve().parent.parent / "exact_measurement_unit_map.v2.json"
)
NOT_EXTRACTED = "not_extracted"
MAPPING_COLUMNS = ("cleaned_record_id", "status", "measurements_json")
RESOLUTION_STATUSES = frozenset({"ok", "relative", "unsure", "unavailable"})


def load_measurement_resolution(
    mapping_path: Path,
) -> dict[str, tuple[str, str | None]]:
    table = pq.read_table(mapping_path, columns=list(MAPPING_COLUMNS))
    columns = [table.column(name).to_pylist() for name in MAPPING_COLUMNS]
    mapping: dict[str, tuple[str, str | None]] = {}
    for record_id, status, measurements_json in zip(*columns, strict=True):
        key = str(record_id or "")
        resolved_status = str(status or "")
        if not key or key in mapping:
            raise ValueError(f"empty or duplicate measurement-resolution ID: {key!r}")
        if resolved_status not in RESOLUTION_STATUSES:
            raise ValueError(
                f"unsupported measurement-resolution status for {key}: "
                f"{resolved_status!r}"
            )
        mapping[key] = (resolved_status, measurements_json)
    return mapping


def load_exact_unit_mapping(
    path: str | Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version not in {_LEGACY_EXACT_UNIT_MAPPING_VERSION, *GROUPED_EXACT_UNIT_MAPPING_VERSIONS}:
        raise ValueError(f"unsupported exact unit mapping: {target}")
    mapping: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in payload.get("entries", []):
        task = str(entry.get("task") or "")
        input_unit = str(entry.get("input_unit") or "")
        endpoints = (
            entry.get("canonical_endpoints")
            if version in GROUPED_EXACT_UNIT_MAPPING_VERSIONS
            else [entry.get("canonical_endpoint")]
        )
        if not isinstance(endpoints, list):
            message = f"invalid exact unit endpoint group: {entry!r}"
            raise ValueError(message)  # noqa: TRY004 - preserve the schema contract
        endpoints = [str(endpoint or "") for endpoint in endpoints]
        if (
            not task
            or not input_unit
            or not endpoints
            or not all(endpoints)
            or endpoints != sorted(set(endpoints))
        ):
            raise ValueError(f"invalid exact unit endpoint group: {entry!r}")
        context = (task, input_unit, endpoints)
        action = entry.get("action")
        domain = entry.get("domain") or "any"
        if action not in {"map", "exclude"} or domain not in {
            "any",
            "nonnegative",
            "positive",
        }:
            raise ValueError(f"invalid exact unit rule for {context!r}")
        if action == "map":
            if not entry.get("canonical_unit"):
                raise ValueError(
                    f"mapped exact unit rule has no output unit: {context!r}"
                )
            try:
                scale = Decimal(str(entry.get("scale")))
            except InvalidOperation as error:
                raise ValueError(f"invalid exact unit scale for {entry!r}") from error
            if not scale.is_finite() or scale <= 0:
                raise ValueError(f"invalid exact unit scale for {entry!r}")
        rule = dict(entry)
        rule.pop("canonical_endpoints", None)
        for endpoint in endpoints:
            key = (task, endpoint, input_unit)
            if key in mapping:
                raise ValueError(f"invalid or duplicate exact unit key: {key}")
            mapping[key] = {**rule, "canonical_endpoint": endpoint}
    return mapping


def _decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"resolved measurement is not a decimal: {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"resolved measurement is not finite: {value!r}")
    return number


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _identity_unit_rule(input_unit: str) -> dict[str, Any]:
    return {
        "action": "map",
        "canonical_unit": input_unit,
        "scale": "1",
        "domain": "any",
    }


def _unit_rule(
    record: dict[str, Any],
    *,
    task: str,
    unit: Any,
    unit_mapping: dict[tuple[str, str, str], dict[str, Any]],
    unit_is_canonical: bool = False,
    allow_identity_fallback: bool = False,
) -> tuple[tuple[str, str, str], str, dict[str, Any], str]:
    input_unit = str(unit or "").strip()
    endpoint = str(record.get("canonical_endpoint_name") or "")
    key = (task, endpoint, input_unit)
    if unit_is_canonical:
        return (
            key,
            input_unit,
            _identity_unit_rule(input_unit),
            "source_declared_canonical",
        )
    rule = unit_mapping.get(key) or unit_mapping.get((task, "*", input_unit))
    if rule is not None:
        return key, input_unit, rule, "frozen_exact_map"
    if allow_identity_fallback:
        return (
            key,
            input_unit,
            _identity_unit_rule(input_unit),
            "source_exact_identity",
        )
    raise ValueError(f"exact unit mapping has no rule for {key}")


def _mapped_value(
    measurement: Any,
    rule: dict[str, Any],
    key: tuple[str, str, str],
) -> tuple[Decimal, float, str]:
    scalar = _decimal(measurement) * Decimal(str(rule["scale"]))
    domain = rule.get("domain", "any")
    outside_domain = (domain == "positive" and scalar <= 0) or (
        domain == "nonnegative" and scalar < 0
    )
    scalar_float = float(scalar)
    if not math.isfinite(scalar_float):
        raise ValueError(f"resolved measurement overflows float for {key}")
    status = "outside_declared_domain" if outside_domain else "within_declared_domain"
    return scalar, scalar_float, status


def _apply_unit_rule(
    record: dict[str, Any],
    *,
    task: str,
    measurement: Any,
    unit: Any,
    unit_mapping: dict[tuple[str, str, str], dict[str, Any]],
    unit_is_canonical: bool = False,
    allow_identity_fallback: bool = False,
) -> None:
    key, input_unit, rule, rule_source = _unit_rule(
        record,
        task=task,
        unit=unit,
        unit_mapping=unit_mapping,
        unit_is_canonical=unit_is_canonical,
        allow_identity_fallback=allow_identity_fallback,
    )
    record.update(
        {
            "measurement_resolution_input_measurement": str(measurement),
            "measurement_resolution_input_unit": input_unit,
            "measurement_unit_mapping_action": rule["action"],
            "measurement_unit_mapping_scale": rule.get("scale"),
            "measurement_unit_mapping_rule_source": rule_source,
            "measurement_numeric_domain": rule.get("domain", "any"),
            "resolved_measurement_text": None,
            "resolved_unit_text": None,
            "resolved_scalar_value": None,
        }
    )
    if rule["action"] == "exclude":
        record.update(
            {
                "measurement_unit_mapping_status": "excluded",
                "measurement_numeric_domain_status": "not_applicable",
            }
        )
        return
    scalar, scalar_float, domain_status = _mapped_value(measurement, rule, key)
    record.update(
        {
            "measurement_unit_mapping_status": "mapped",
            "measurement_numeric_domain_status": domain_status,
            "resolved_measurement_text": _decimal_text(scalar),
            "resolved_unit_text": str(rule["canonical_unit"]),
            "resolved_scalar_value": scalar_float,
        }
    )


def _source_resolution(
    source_record: dict[str, Any],
    resolution: tuple[str, str | None] | None,
    mapping_active: bool,
) -> tuple[str, str | None, list[dict[str, Any]], str | None]:
    route = str(source_record.get("measurement_resolution_route") or "")
    if route == "accept" and mapping_active:
        entries = [
            {
                "measurement": source_record.get(
                    "measurement_resolution_exact_measurement",
                    source_record.get("measurement_text"),
                ),
                "unit": source_record.get(
                    "measurement_resolution_exact_unit",
                    source_record.get("unit_text"),
                ),
            }
        ]
        return "ok", "source_exact", entries, None
    if route == "extract":
        status = NOT_EXTRACTED if resolution is None else resolution[0]
        entries = json.loads(resolution[1] or "[]") if resolution is not None else []
        resolution_json = resolution[1] if resolution is not None else None
        return status, "llm", entries, resolution_json
    return NOT_EXTRACTED, None, [], None


def _preflight_records(
    records: list[dict[str, Any]],
    *,
    mapping: dict[str, tuple[str, str | None]],
    mapping_active: bool,
    mapping_required: bool,
    task: str,
    unit_mapping: dict[tuple[str, str, str], dict[str, Any]],
    expected_routing_version: str,
    allow_identity_fallback: bool,
    allow_extracted_identity_fallback: bool,
) -> list[str]:
    uncovered: list[str] = []
    for source_record in records:
        record_id = str(source_record.get("cleaned_record_id") or "")
        routing_version = str(source_record.get("measurement_routing_version") or "")
        if routing_version != expected_routing_version:
            raise ValueError(
                f"measurement routing version drift for {record_id}: "
                f"expected {expected_routing_version or '<missing>'}, found "
                f"{routing_version or '<missing>'}"
            )
        resolution = mapping.get(record_id)
        route = str(source_record.get("measurement_resolution_route") or "")
        status, _, entries, _ = _source_resolution(
            source_record, resolution, mapping_active
        )
        if route == "extract" and resolution is None and mapping_required:
            uncovered.append(record_id)
        if status == "ok" and not entries:
            raise ValueError(f"resolved row {record_id!r} has no measurements")
        for entry in entries if status == "ok" else ():
            _, _, rule, _ = _unit_rule(
                source_record,
                task=task,
                unit=entry.get("unit"),
                unit_mapping=unit_mapping,
                unit_is_canonical=bool(
                    route == "accept"
                    and source_record.get(
                        "measurement_resolution_exact_unit_is_canonical"
                    )
                ),
                allow_identity_fallback=(
                    (route == "accept" and allow_identity_fallback)
                    or (route == "extract" and allow_extracted_identity_fallback)
                ),
            )
            if rule["action"] == "map":
                _decimal(entry.get("measurement"))
    return uncovered


def _resolution_metadata(
    source_record: dict[str, Any],
    record_id: str,
    status: str,
    origin: str | None,
    entries: list[dict[str, Any]],
    resolution_json: str | None,
    index: int,
) -> dict[str, Any]:
    quantity_count = len(entries) if status == "ok" else 0
    return {
        "cleaned_record_id": (
            record_id
            if quantity_count <= 1
            else stable_id("measurement_resolution_child", record_id, index)
        ),
        "measurement_resolution_parent_cleaned_record_id": record_id,
        "measurement_resolution_entry_index": index if status == "ok" else None,
        "measurement_resolution_quantity_count": quantity_count,
        "measurement_resolution_origin": origin if status == "ok" else None,
        "measurement_resolution_status": status,
        "measurement_resolution_active": True,
        "measurement_resolution_entries_json": (
            resolution_json
            if resolution_json is not None
            else json.dumps(entries)
            if entries
            else None
        ),
        "pre_resolution_measurement_text": source_record.get("measurement_text"),
        "pre_resolution_unit_text": source_record.get("unit_text"),
    }


def _resolved_records(
    source_record: dict[str, Any],
    resolved: tuple[str, str | None, list[dict[str, Any]], str | None],
    *,
    task: str,
    unit_mapping: dict[tuple[str, str, str], dict[str, Any]],
    allow_identity_fallback: bool,
    allow_extracted_identity_fallback: bool,
) -> Iterator[dict[str, Any]]:
    status, origin, entries, resolution_json = resolved
    record_id = str(source_record.get("cleaned_record_id") or "")
    route = str(source_record.get("measurement_resolution_route") or "")
    child_entries = entries if status == "ok" else [None]
    for index, entry in enumerate(child_entries):
        record = source_record if len(child_entries) == 1 else dict(source_record)
        record.update(
            _resolution_metadata(
                source_record,
                record_id,
                status,
                origin,
                entries,
                resolution_json,
                index,
            )
        )
        if entry is not None:
            _apply_unit_rule(
                record,
                task=task,
                measurement=entry["measurement"],
                unit=entry["unit"],
                unit_mapping=unit_mapping,
                unit_is_canonical=bool(
                    route == "accept"
                    and source_record.get(
                        "measurement_resolution_exact_unit_is_canonical"
                    )
                ),
                allow_identity_fallback=(
                    (route == "accept" and allow_identity_fallback)
                    or (route == "extract" and allow_extracted_identity_fallback)
                ),
            )
        yield record


def _apply_resolutions(
    records: list[dict[str, Any]],
    mapping: dict[str, tuple[str, str | None]],
    *,
    mapping_active: bool,
    task: str,
    unit_mapping: dict[tuple[str, str, str], dict[str, Any]],
    allow_identity_fallback: bool,
    allow_extracted_identity_fallback: bool,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    pending = deque(records)
    records.clear()
    while pending:
        source_record = pending.popleft()
        record_id = str(source_record.get("cleaned_record_id") or "")
        resolved = _source_resolution(
            source_record, mapping.pop(record_id, None), mapping_active
        )
        status, origin, _, _ = resolved
        for record in _resolved_records(
            source_record,
            resolved,
            task=task,
            unit_mapping=unit_mapping,
            allow_identity_fallback=allow_identity_fallback,
            allow_extracted_identity_fallback=allow_extracted_identity_fallback,
        ):
            if status == "ok":
                unit_counts[str(record["measurement_unit_mapping_status"])] += 1
            records.append(record)
        counts[status] += 1
        if status == "ok":
            origin_counts[origin] += 1
    return counts, origin_counts, unit_counts


def _consume_extra_mappings(
    mapping: dict[str, tuple[str, str | None]],
    ignored_record_ids: set[str] | None,
    allow_partial: bool,
    allow_out_of_scope: bool,
) -> tuple[int, int]:
    ignored_mapping_rows = sorted(set(mapping) & set(ignored_record_ids or ()))
    for record_id in ignored_mapping_rows:
        mapping.pop(record_id)
    ignored_out_of_scope_mapping_rows = (
        len(mapping) if allow_partial or allow_out_of_scope else 0
    )
    if allow_partial or allow_out_of_scope:
        mapping.clear()
    if mapping:
        first = next(iter(mapping))
        raise ValueError(
            f"{len(mapping)} measurement-resolution row(s) are absent from Stage 01; "
            f"first={first}"
        )
    return len(ignored_mapping_rows), ignored_out_of_scope_mapping_rows


def _resolution_audit(
    records: list[dict[str, Any]],
    *,
    mapping_path: Path | None,
    unit_mapping_path: str | Path,
    mapping_rows: int,
    ignored_mapping_rows: int,
    ignored_out_of_scope_mapping_rows: int,
    counts: Counter[str],
    origin_counts: Counter[str],
    unit_counts: Counter[str],
    uncovered: list[str],
) -> dict[str, Any]:
    return {
        "apply_version": RESOLUTION_APPLY_VERSION,
        "mapping_path": str(mapping_path) if mapping_path else None,
        "mapping_sha256": file_sha256(mapping_path) if mapping_path else None,
        "unit_mapping_path": str(unit_mapping_path) if mapping_path else None,
        "unit_mapping_sha256": (
            file_sha256(Path(unit_mapping_path)) if mapping_path else None
        ),
        "mapping_rows": mapping_rows,
        "ignored_structure_rejection_rows": ignored_mapping_rows,
        "ignored_out_of_scope_mapping_rows": ignored_out_of_scope_mapping_rows,
        "input_rows": sum(counts.values()),
        "output_rows": len(records),
        "status_counts": dict(sorted(counts.items())),
        "ok_origin_counts": dict(sorted(origin_counts.items())),
        "unit_mapping_status_counts": dict(sorted(unit_counts.items())),
        "source_exact_identity_unit_rows": sum(
            row.get("measurement_unit_mapping_rule_source") == "source_exact_identity"
            for row in records
        ),
        "substituted_rows": unit_counts["mapped"],
        "exploded_child_rows": len(records) - sum(counts.values()),
        "extract_rows_without_a_resolution": len(uncovered),
    }


def apply_measurement_resolution(
    records: list[dict[str, Any]],
    *,
    mapping_path: Path | None,
    task: str = "",
    unit_mapping_path: str | Path = DEFAULT_EXACT_UNIT_MAPPING,
    allow_partial: bool = False,
    allow_out_of_scope_mapping_rows: bool = False,
    ignored_record_ids: set[str] | None = None,
    expected_routing_version: str = DEFAULT_EXPECTED_ROUTING_VERSION,
    allow_unmapped_source_exact_units: bool = False,
    allow_unmapped_extracted_units: bool = False,
) -> dict[str, Any]:
    """Join resolved quantities, explode them, and exact-map their units."""
    mapping_active = bool(mapping_path)
    mapping = load_measurement_resolution(mapping_path) if mapping_active else {}
    unit_mapping = load_exact_unit_mapping(unit_mapping_path) if mapping_active else {}
    mapping_rows = len(mapping)
    uncovered = _preflight_records(
        records,
        mapping=mapping,
        mapping_active=mapping_active,
        mapping_required=mapping_path is not None,
        task=task,
        unit_mapping=unit_mapping,
        expected_routing_version=expected_routing_version,
        allow_identity_fallback=allow_unmapped_source_exact_units,
        allow_extracted_identity_fallback=allow_unmapped_extracted_units,
    )
    if uncovered and not allow_partial:
        raise ValueError(
            f"{len(uncovered)} extract-routed row(s) have no frozen resolution; "
            f"first={uncovered[0]}"
        )
    counts, origin_counts, unit_counts = _apply_resolutions(
        records,
        mapping,
        mapping_active=mapping_active,
        task=task,
        unit_mapping=unit_mapping,
        allow_identity_fallback=allow_unmapped_source_exact_units,
        allow_extracted_identity_fallback=allow_unmapped_extracted_units,
    )
    ignored_rows, out_of_scope_rows = _consume_extra_mappings(
        mapping,
        ignored_record_ids,
        allow_partial,
        allow_out_of_scope_mapping_rows,
    )
    return _resolution_audit(
        records,
        mapping_path=mapping_path,
        unit_mapping_path=unit_mapping_path,
        mapping_rows=mapping_rows,
        ignored_mapping_rows=ignored_rows,
        ignored_out_of_scope_mapping_rows=out_of_scope_rows,
        counts=counts,
        origin_counts=origin_counts,
        unit_counts=unit_counts,
        uncovered=uncovered,
    )


__all__ = [
    "DEFAULT_EXACT_UNIT_MAPPING",
    "EXACT_UNIT_MAPPING_VERSION",
    "NOT_EXTRACTED",
    "RESOLUTION_APPLY_VERSION",
    "apply_measurement_resolution",
    "load_exact_unit_mapping",
    "load_measurement_resolution",
]
