"""Descriptive inputs for auditable evidence-compatibility policies.

This module deliberately stops before task-specific compatibility decisions.
It exposes normalized endpoint, context, measurement, support, and uncertainty
signals that a task adapter may later classify.  The output is an internal
audit/selection sidecar and must not be treated as a label or LLM evidence.
"""

from __future__ import annotations

from typing import Any, Mapping

from tools.chembl_tool.common.evidence_contract import evidence_for_llm


COMPATIBILITY_INPUT_VERSION = "evidence_compatibility_inputs.v1"


def compatibility_inputs_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return source-backed compatibility inputs without scoring relevance."""
    record = evidence_for_llm(row)
    endpoint = record.get("endpoint") or {}
    measurement = endpoint.get("measurement") or {}
    text = record.get("text") or {}
    annotations = record.get("annotations") or {}
    provenance = record.get("provenance") or {}
    scope = annotations.get("scope") or {}
    uncertainty = [
        str(value).strip()
        for value in annotations.get("uncertainty") or []
        if str(value).strip()
    ]
    endpoint_name = _normalize_text(endpoint.get("name"))
    unit = _normalize_text(measurement.get("unit"))
    relation = _normalize_text(measurement.get("relation"))
    value = measurement.get("value")
    has_value = value not in (None, "")
    pmids = _unique_text(
        row.get("source_pmids")
        or row.get("pmids")
        or row.get("source_pmid")
        or row.get("pmid")
    )
    source_record_count = _first_int(
        provenance.get("source_record_count"),
        row.get("source_record_count"),
    )
    numeric_record_count = _first_int(
        row.get("source_numeric_record_count"),
        row.get("numeric_record_count"),
    )
    example_measurements = [
        {
            "endpoint": _normalize_text(example.get("endpoint_type")),
            "value": str(example.get("reported_value") or "").strip(),
            "unit": _normalize_text(example.get("reported_units")),
        }
        for example in record.get("examples") or []
        if isinstance(example, Mapping)
        and str(example.get("reported_value") or "").strip()
    ]
    has_measurement = has_value or bool(example_measurements)
    n_measurements = max(
        int(numeric_record_count or 0),
        len(example_measurements),
        int(has_value),
    )
    return {
        "contract_version": COMPATIBILITY_INPUT_VERSION,
        "endpoint": {
            "name": endpoint_name,
            "group_id": _normalize_text((record.get("group") or {}).get("id")),
            "endpoint_group": _normalize_text(
                (record.get("group") or {}).get("endpoint_group")
            ),
            "available": bool(endpoint_name),
        },
        "measurement": {
            "relation": relation,
            "value": value if has_value else "",
            "unit": unit,
            "has_value": has_value,
            "has_unit": bool(unit),
            "has_any_reported_measurement": has_measurement,
            "n_reported_measurements": n_measurements,
            "example_measurements": example_measurements,
            "raw_comparability_key": (
                f"{endpoint_name}|{unit}" if endpoint_name and unit else ""
            ),
            "requires_task_unit_normalization": bool(has_value),
        },
        "context": {
            "scope": dict(scope) if isinstance(scope, Mapping) else {},
            "scope_keys": sorted(str(key) for key in scope) if isinstance(scope, Mapping) else [],
            "context_text_available": bool(_normalize_text(text.get("context"))),
            "evidence_text_available": bool(_normalize_text(text.get("evidence"))),
        },
        "role": _normalize_text(annotations.get("evidence_role")) or "unspecified",
        "support": {
            "source_record_count": source_record_count,
            "numeric_record_count": numeric_record_count,
            "independent_pmid_count": len(pmids),
            "has_multi_record_support": bool(source_record_count and source_record_count > 1),
            "has_independent_pmid_support": len(pmids) > 1,
        },
        "uncertainty": {
            "flags": uncertainty,
            "n_flags": len(uncertainty),
            "has_explicit_uncertainty": bool(uncertainty),
            "confidence": (record.get("quality") or {}).get("confidence", ""),
        },
        "readiness": {
            "endpoint_match": bool(endpoint_name),
            "context_compatibility": bool(scope) or bool(_normalize_text(text.get("context"))),
            "measurement_available": bool(endpoint_name and has_measurement),
            "measurement_consistency": bool(endpoint_name and n_measurements >= 2),
            "record_agreement": bool(
                (source_record_count and source_record_count >= 2) or len(pmids) >= 2
            ),
            "evidence_scope": bool(scope) or (
                _normalize_text(annotations.get("evidence_role")) not in {"", "unspecified"}
            ),
        },
    }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _unique_text(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in values
            if str(item).strip()
        )
    )


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
