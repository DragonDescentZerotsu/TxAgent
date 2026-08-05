"""Build molecule-level evidence from profile-mapped Starling parquet files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import statistics
import time
from typing import Any

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
    standardize_smiles,
)


@dataclass(frozen=True)
class StarlingSourceProfile:
    """Declarative mapping from one parquet schema to the common evidence row."""

    source_id: str
    path: str
    group_id: str
    assay_tier: str
    endpoint_group: str
    evidence_source: str
    endpoint_field: str
    smiles_field: str = "smiles"
    value_field: str = ""
    unit_field: str = ""
    context_fields: tuple[str, ...] = ()
    scope_fields: tuple[str, ...] = ()
    name_fields: tuple[str, ...] = ("molecule_name", "global_identifier")
    support_text_field: str = "support_text"
    confidence_field: str = "confidence"
    record_id_field: str = "extraction_id"
    target_pref_name: str = ""
    evidence_role: str = "unspecified"
    allow_numeric_proxy_summary: bool = False
    standard_type_prefix: str = ""
    include_endpoint_values: tuple[str, ...] = ()
    exclude_endpoint_values: tuple[str, ...] = ()
    context_filter_fields: tuple[str, ...] = ()
    required_context_patterns_by_endpoint: tuple[tuple[str, tuple[str, ...]], ...] = ()
    max_rows: int = 0
    extra_example_fields: tuple[str, ...] = field(default_factory=tuple)


def build_starling_parquet_evidence_rows(
    profiles: list[StarlingSourceProfile],
    *,
    max_record_examples: int = 6,
    min_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load profiles and aggregate raw rows to one evidence row per source molecule."""
    rows: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {}
    canonical_smiles_cache: dict[str, str] = {}
    for profile in profiles:
        records, stats = _load_profile_records(
            profile,
            min_confidence=min_confidence,
            canonical_smiles_cache=canonical_smiles_cache,
        )
        profile_rows = _aggregate_profile_records(profile, records, max_record_examples=max_record_examples)
        rows.extend(profile_rows)
        source_stats[profile.source_id] = {**stats, "n_evidence_rows": len(profile_rows)}
    return rows, {"n_sources": len(profiles), "n_evidence_rows": len(rows), "sources": source_stats}


def build_and_write_starling_index(
    evidence_rows: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    index_version: str,
    source: dict[str, Any],
    source_stats: dict[str, Any],
    evidence_filename: str,
    index_filename: str,
    meta_filename: str,
    workers: int = 1,
    progress_every: int = 10000,
) -> dict[str, Any]:
    """Build and persist a profile-backed Starling index with common metadata."""
    started = time.monotonic()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    index = build_neighbor_index(
        evidence_rows,
        index_version=index_version,
        workers=workers,
        progress_every=progress_every,
    )
    index["source"] = source

    evidence_path = out_path / evidence_filename
    index_path = out_path / index_filename
    meta_path = out_path / meta_filename
    write_jsonl(evidence_path, evidence_rows)
    with index_path.open("wb") as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "index_version": index_version,
        "n_evidence_rows": len(evidence_rows),
        "n_index_molecules": len(index["molecules"]),
        "groups": sorted(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "source": source,
        "source_stats": source_stats,
        "elapsed_s": round(time.monotonic() - started, 3),
        "paths": {
            "evidence_jsonl": str(evidence_path),
            "index_pkl": str(index_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return meta


def _load_profile_records(
    profile: StarlingSourceProfile,
    *,
    min_confidence: float,
    canonical_smiles_cache: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("pandas with parquet support is required for Starling parquet ingestion") from error

    path = Path(profile.path)
    frame = pd.read_parquet(path)
    if profile.max_rows:
        frame = frame.head(profile.max_rows)

    records: list[dict[str, Any]] = []
    n_missing_smiles = 0
    n_invalid_smiles = 0
    n_low_confidence = 0
    n_filtered_endpoint = 0
    n_filtered_context = 0
    required_context = {
        endpoint.lower(): patterns for endpoint, patterns in profile.required_context_patterns_by_endpoint
    }
    for row_number, raw in enumerate(frame.to_dict(orient="records"), start=1):
        row = {key: _clean_scalar(value) for key, value in raw.items()}
        input_smiles = _text(row.get(profile.smiles_field))
        if not input_smiles:
            n_missing_smiles += 1
            continue
        if input_smiles not in canonical_smiles_cache:
            canonical_smiles_cache[input_smiles] = standardize_smiles(input_smiles)[0]
        canonical_smiles = canonical_smiles_cache[input_smiles]
        if not canonical_smiles:
            n_invalid_smiles += 1
            continue
        endpoint_value = _text(row.get(profile.endpoint_field)).lower()
        include_values = {value.lower() for value in profile.include_endpoint_values}
        exclude_values = {value.lower() for value in profile.exclude_endpoint_values}
        if include_values and endpoint_value not in include_values:
            n_filtered_endpoint += 1
            continue
        if exclude_values and endpoint_value in exclude_values:
            n_filtered_endpoint += 1
            continue
        required_patterns = required_context.get(endpoint_value, ())
        if required_patterns:
            context_text = " ".join(_text(row.get(field_name)) for field_name in profile.context_filter_fields)
            if not any(re.search(pattern, context_text, flags=re.IGNORECASE) for pattern in required_patterns):
                n_filtered_context += 1
                continue
        confidence = _float_or_none(row.get(profile.confidence_field))
        if confidence is not None and confidence < min_confidence:
            n_low_confidence += 1
            continue
        row["_source_row_number"] = row_number
        row["_canonical_smiles"] = canonical_smiles
        records.append(row)

    return records, {
        "path": str(path),
        "n_input_rows": len(frame),
        "n_loaded_rows": len(records),
        "n_missing_smiles": n_missing_smiles,
        "n_invalid_smiles": n_invalid_smiles,
        "n_low_confidence": n_low_confidence,
        "n_filtered_endpoint": n_filtered_endpoint,
        "n_filtered_context": n_filtered_context,
        "n_unique_smiles": len({_text(row.get("_canonical_smiles")) for row in records}),
        "endpoint_field": profile.endpoint_field,
        "endpoint_counts": dict(
            Counter(_text(row.get(profile.endpoint_field)) or "<missing>" for row in records).most_common(50)
        ),
    }


def _aggregate_profile_records(
    profile: StarlingSourceProfile,
    records: list[dict[str, Any]],
    *,
    max_record_examples: int,
) -> list[dict[str, Any]]:
    records_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_smiles[_text(record.get("_canonical_smiles"))].append(record)
    return [
        _summarize_profile_molecule(profile, smiles, source_records, limit=max_record_examples)
        for smiles, source_records in sorted(records_by_smiles.items())
    ]


def _summarize_profile_molecule(
    profile: StarlingSourceProfile,
    smiles: str,
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    endpoint_counts = Counter(_text(record.get(profile.endpoint_field)) or "unspecified" for record in records)
    unit_counts = Counter(_text(record.get(profile.unit_field)) or "unspecified" for record in records) if profile.unit_field else Counter()
    confidence_values = [
        value
        for value in (_float_or_none(record.get(profile.confidence_field)) for record in records)
        if value is not None
    ]
    examples = _representative_examples(profile, records, limit=limit)
    endpoint_summary = _compact_distribution(endpoint_counts)
    numeric_values = [
        value
        for value in (_float_or_none(record.get(profile.value_field)) for record in records)
        if value is not None
    ] if profile.value_field else []
    direct_summary_allowed = (
        (
            profile.evidence_role == "direct_outcome"
            or profile.allow_numeric_proxy_summary
        )
        and len(endpoint_counts) == 1
        and len([unit for unit in unit_counts if unit and unit != "unspecified"]) <= 1
        and bool(numeric_values)
    )
    summary_unit = _single_value_or_empty(unit_counts)
    standard_type = profile.standard_type_prefix or endpoint_summary
    if profile.standard_type_prefix and endpoint_summary:
        standard_type = f"{profile.standard_type_prefix}: {endpoint_summary}"
    scope = {
        field_name: _unique_values(record.get(field_name) for record in records)[:10]
        for field_name in profile.scope_fields
        if _unique_values(record.get(field_name) for record in records)
    }
    uncertainty = []
    if any(not _text(record.get(profile.support_text_field)) for record in records):
        uncertainty.append("some_source_rows_missing_support_text")
    if profile.value_field and not any(_text(record.get(profile.value_field)) for record in records):
        uncertainty.append("source_rows_missing_reported_value")
    if direct_summary_allowed and not summary_unit:
        uncertainty.append("direct_outcome_numeric_value_missing_unit")

    row = {
        "molecule_chembl_id": starling_molecule_id(smiles),
        "canonical_smiles": smiles,
        "assay_chembl_id": f"STARLING_{profile.source_id.upper()}",
        "assay_tier": profile.assay_tier,
        "endpoint_group": profile.endpoint_group,
        "group_id": profile.group_id,
        "standard_type": standard_type,
        "standard_relation": "=" if direct_summary_allowed else "",
        "standard_value": round(statistics.median(numeric_values), 6) if direct_summary_allowed else "",
        "standard_units": summary_unit,
        "pchembl_value": "",
        "activity_comment": f"Starling {profile.source_id} summary over {len(records)} records",
        "data_validity_comment": "",
        "assay_description": _examples_text(examples, max_chars=5000),
        "target_pref_name": profile.target_pref_name or profile.endpoint_group,
        "target_genes": "",
        "organism": _organism_summary(profile, records),
        "confidence_score": round(statistics.median(confidence_values), 4) if confidence_values else "",
        "relationship_type": "",
        "evidence_source": profile.evidence_source,
        "evidence_role": profile.evidence_role,
        "evidence_scope": scope,
        "transferability": "not_assessed",
        "uncertainty": uncertainty,
        "source_molecule_names": _unique_values(
            _first_nonempty(record.get(field_name) for field_name in profile.name_fields)
            for record in records
        )[:10],
        "source_record_count": len(records),
        "source_numeric_record_count": _count_numeric_records(profile, records),
        "source_qualitative_record_count": len(records) - _count_numeric_records(profile, records),
        "source_value_min": min(numeric_values) if numeric_values else "",
        "source_value_median": statistics.median(numeric_values) if numeric_values else "",
        "source_value_max": max(numeric_values) if numeric_values else "",
        "source_endpoint_counts": dict(endpoint_counts.most_common(20)),
        "source_unit_counts": dict(unit_counts.most_common(20)),
        "source_support_texts": _unique_values(record.get(profile.support_text_field) for record in records)[:limit],
        "source_record_examples": examples,
        "source_qualitative_examples": [],
    }
    return attach_minimal_evidence(row)


def _representative_examples(
    profile: StarlingSourceProfile,
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: (
            _text(record.get(profile.endpoint_field)),
            -(_float_or_none(record.get(profile.confidence_field)) or 0.0),
            _text(record.get(profile.value_field)) if profile.value_field else "",
            int(record.get("_source_row_number") or 0),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_endpoints: set[str] = set()
    for record in ranked:
        endpoint = _text(record.get(profile.endpoint_field))
        if endpoint not in seen_endpoints:
            selected.append(record)
            seen_endpoints.add(endpoint)
        if len(selected) >= limit:
            break
    for record in ranked:
        if len(selected) >= limit:
            break
        if record not in selected:
            selected.append(record)
    return [_example_payload(profile, record) for record in selected]


def _example_payload(profile: StarlingSourceProfile, record: dict[str, Any]) -> dict[str, Any]:
    context = {
        field_name: _clean_scalar(record.get(field_name))
        for field_name in profile.context_fields
        if _clean_scalar(record.get(field_name)) != ""
    }
    payload = {
        "source_id": profile.source_id,
        "source_index": record.get("_source_row_number", ""),
        "source_record_id": record.get(profile.record_id_field, "") if profile.record_id_field else "",
        "molecule_name": _first_nonempty(record.get(field_name) for field_name in profile.name_fields),
        "endpoint_type": _text(record.get(profile.endpoint_field)),
        "reported_value": _text(record.get(profile.value_field)) if profile.value_field else "",
        "reported_units": _text(record.get(profile.unit_field)) if profile.unit_field else "",
        "context": context,
        "support_text": _text(record.get(profile.support_text_field)),
        "source_confidence": _clean_scalar(record.get(profile.confidence_field)),
    }
    for field_name in profile.extra_example_fields:
        value = _clean_scalar(record.get(field_name))
        if value != "":
            payload[field_name] = value
    return payload


def _examples_text(examples: list[dict[str, Any]], *, max_chars: int) -> str:
    chunks = []
    for example in examples:
        parts = []
        for key in ("endpoint_type", "reported_value", "reported_units", "context", "support_text"):
            value = example.get(key)
            if value not in (None, "", {}):
                parts.append(f"{key}: {json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, dict) else value}")
        if parts:
            chunks.append("\n".join(parts))
    return "\n\n---\n\n".join(chunks)[:max_chars]


def _organism_summary(profile: StarlingSourceProfile, records: list[dict[str, Any]]) -> str:
    for field_name in ("species", *profile.scope_fields, *profile.context_fields):
        if any(token in field_name.lower() for token in ("species", "population", "organism", "biological_context")):
            values = _unique_values(record.get(field_name) for record in records)
            if values:
                return "; ".join(values[:5])
    return ""


def _count_numeric_records(profile: StarlingSourceProfile, records: list[dict[str, Any]]) -> int:
    if not profile.value_field:
        return 0
    return sum(_float_or_none(record.get(profile.value_field)) is not None for record in records)


def _compact_distribution(counter: Counter[str]) -> str:
    return "; ".join(f"{key}={count}" for key, count in counter.most_common(8) if key)


def _single_value_or_empty(counter: Counter[str]) -> str:
    values = [value for value in counter if value and value != "unspecified"]
    return values[0] if len(values) == 1 else ""


def starling_molecule_id(smiles: str) -> str:
    """Return the stable cross-profile molecule identifier used by Starling indices."""
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16].upper()
    return f"STARLING_{digest}"


def _unique_values(values: Any) -> list[str]:
    output: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in output:
            output.append(text)
    return output


def _first_nonempty(values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _text(value: Any) -> str:
    value = _clean_scalar(value)
    return "" if value == "" else str(value).strip()


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
