"""Layers 4-5: record deduplication and molecule-level organization."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id

from .cleaning import stable_id
from .contracts import ORGANIZATION_STAGE_VERSION, NormalizationResult


def deduplicate_within_source(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse exact semantic duplicates within, never across, one source."""
    kept_by_key: dict[str, dict[str, Any]] = {}
    duplicate_ids: dict[str, list[str]] = defaultdict(list)
    duplicate_rows: list[dict[str, Any]] = []
    order_key = lambda item: (  # noqa: E731 - reused for the monotonic fast path
        str(item.get("source_id") or ""),
        int(item.get("source_row_number") or 0),
    )
    already_ordered = all(
        order_key(records[index - 1]) <= order_key(records[index])
        for index in range(1, len(records))
    )
    ordered: Sequence[Mapping[str, Any]] = (
        records if already_ordered else sorted(records, key=order_key)
    )
    for source_view in ordered:
        source = source_view
        context_identity = source.get("deduplication_context_id")
        if not context_identity:
            context_identity = stable_id(
                "source_context",
                source.get("source_id"),
                str(source.get("evidence_context_json") or "{}"),
            )
        key = stable_id(
            "duplicate",
            source.get("source_id"),
            source.get("source_smiles"),
            source.get("canonical_smiles"),
            source.get("group_id"),
            source.get("endpoint_name"),
            # Assay-transfer base-unit and tail transforms happen before this
            # organization step. Retrieval membership must instead retain the
            # exact dedup identity that existed at the source-canonical layer.
            source.get("assay_transfer_prebase_measurement_text")
            if "assay_transfer_prebase_measurement_text" in source
            else source.get("canonical_measurement"),
            source.get("assay_transfer_prebase_unit_text")
            if "assay_transfer_prebase_unit_text" in source
            else source.get("canonical_unit"),
            context_identity,
            source.get("support_text"),
        )
        existing = kept_by_key.get(key)
        source_record_id = str(source.get("source_record_id") or "")
        if existing is None:
            kept_by_key[key] = dict(source)
            duplicate_ids[key].append(source_record_id)
            continue
        duplicate_ids[key].append(source_record_id)
        duplicate_rows.append(
            {
                "duplicate_group_id": key,
                "kept_normalized_record_id": existing.get("normalized_record_id"),
                "removed_normalized_record_id": source.get("normalized_record_id"),
                "source_id": source.get("source_id"),
                "source_record_id": source_record_id,
            }
        )
    output: list[dict[str, Any]] = []
    for key, record in kept_by_key.items():
        ids = duplicate_ids[key]
        retrieval_eligible = bool(record.get("canonical_smiles") and record.get("group_id"))
        record.update(
            {
                "organization_version": ORGANIZATION_STAGE_VERSION,
                "duplicate_group_id": key,
                "duplicate_group_size": len(ids),
                "duplicate_source_record_ids": json.dumps(ids, ensure_ascii=False),
                "retrieval_eligible": retrieval_eligible,
                "organization_status": (
                    "retrieval_eligible"
                    if retrieval_eligible
                    else (
                        str(record.get("structure_status") or "unresolved")
                        if not record.get("canonical_smiles")
                        else "unresolved_mechanism_family"
                    )
                ),
            }
        )
        output.append(record)
    return output, duplicate_rows


def organize_normalized_records(
    normalized_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records, duplicates = deduplicate_within_source(normalized_records)
    exclusions = [
        {
            "normalized_record_id": record.get("normalized_record_id"),
            "cleaned_record_id": record.get("cleaned_record_id"),
            "source_id": record.get("source_id"),
            "source_row_number": record.get("source_row_number"),
            "source_record_id": record.get("source_record_id"),
            "structure_status": record.get("structure_status"),
            "organization_status": record.get("organization_status"),
        }
        for record in records
        if not record.get("retrieval_eligible")
    ]
    stats = {
        "n_normalized_records_before_deduplication": len(normalized_records),
        "n_records": len(records),
        "n_duplicates_removed": len(duplicates),
        "n_retrieval_eligible": sum(bool(row.get("retrieval_eligible")) for row in records),
        "n_organization_exclusions": len(exclusions),
        "n_absolute_and_continuous": sum(
            bool(row.get("is_absolute_and_continuous")) for row in records
        ),
    }
    return records, duplicates, exclusions, stats


def aggregate_molecule_family_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_record_examples: int = 6,
) -> list[dict[str, Any]]:
    """Build one evidence row per retrievable molecule and mechanism family."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("retrieval_eligible"):
            continue
        grouped[(str(record["canonical_smiles"]), str(record["group_id"]))].append(record)

    evidence_rows: list[dict[str, Any]] = []
    for (smiles, group_id), group_records in sorted(grouped.items()):
        first = group_records[0]
        examples = representative_examples(group_records, limit=max_record_examples)
        scalar_records = [
            record
            for record in group_records
            if record.get("finite_scalar_value") is not None
        ]
        confidence_values = [
            float(record["confidence"])
            for record in group_records
            if record.get("confidence") is not None
        ]
        source_names = sorted({str(record["source_name"]) for record in group_records})
        endpoint_counts = Counter(
            str(record.get("endpoint_name") or "unspecified")
            for record in group_records
        )
        row = {
            "molecule_chembl_id": starling_molecule_id(smiles),
            "canonical_smiles": smiles,
            "assay_chembl_id": f"STARLING_NORMALIZED_{_slug(group_id).upper()}",
            "assay_tier": first["assay_tier"],
            "endpoint_group": first["endpoint_group"],
            "group_id": group_id,
            "standard_type": "; ".join(
                f"{name}={count}" for name, count in endpoint_counts.most_common(8)
            ),
            "standard_relation": "",
            "standard_value": "",
            "standard_units": "",
            "pchembl_value": "",
            "activity_comment": (
                f"Normalized Starling summary over {len(group_records)} retained records "
                f"from {len(source_names)} source(s)"
            ),
            "data_validity_comment": "",
            "assay_description": examples_text(examples, max_chars=5000),
            "target_pref_name": first["target_pref_name"],
            "target_genes": "",
            "organism": "",
            "confidence_score": (
                round(statistics.median(confidence_values), 4) if confidence_values else ""
            ),
            "relationship_type": "",
            "evidence_source": "Starling normalized oral bioavailability",
            "evidence_role": first["evidence_role"],
            "evidence_scope": {},
            "transferability": "not_assessed",
            "uncertainty": (
                ["qualitative_or_non_scalar_records_present"]
                if len(scalar_records) != len(group_records)
                else []
            ),
            "source_molecule_names": sorted(
                {
                    str(record["molecule_name"])
                    for record in group_records
                    if record.get("molecule_name")
                }
            )[:10],
            "source_record_count": len(group_records),
            "source_numeric_record_count": len(scalar_records),
            "source_qualitative_record_count": len(group_records) - len(scalar_records),
            "source_record_examples": examples,
            "source_qualitative_examples": [],
            "source_names": source_names,
            "normalized_records": [_compact_record(record) for record in group_records],
        }
        evidence_rows.append(attach_minimal_evidence(row))
    return evidence_rows


def representative_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            str(item.get("canonical_endpoint") or item.get("endpoint_name") or ""),
            -(float(item.get("confidence")) if item.get("confidence") is not None else 0.0),
            str(item.get("source_id") or ""),
            int(item.get("source_row_number") or 0),
        ),
    )
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for record in ranked:
        identity = str(
            record.get("canonical_endpoint") or record.get("endpoint_name") or ""
        )
        if identity not in seen:
            selected.append(record)
            seen.add(identity)
        if len(selected) >= limit:
            break
    for record in ranked:
        if len(selected) >= limit:
            break
        if record not in selected:
            selected.append(record)
    return [_llm_example(record) for record in selected]


def examples_text(examples: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    chunks: list[str] = []
    for example in examples:
        parts: list[str] = []
        for key in (
            "endpoint_type",
            "reported_value",
            "reported_units",
            "context",
            "support_text",
        ):
            value = example.get(key)
            if value not in (None, "", {}):
                rendered = (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, Mapping)
                    else str(value)
                )
                parts.append(f"{key}: {rendered}")
        if parts:
            chunks.append("\n".join(parts))
    return "\n\n---\n\n".join(chunks)[:max_chars]


def combine_normalization_results(results: Sequence[NormalizationResult]) -> NormalizationResult:
    records = [record for result in results for record in result.records]
    rejections = [record for result in results for record in result.rejections]
    duplicates = [record for result in results for record in result.duplicates]
    return NormalizationResult(
        records,
        rejections,
        duplicates,
        {
            "n_sources": len(results),
            "n_input_rows": sum(int(result.stats.get("n_input_rows", 0)) for result in results),
            "n_records": len(records),
            "n_rejections": len(rejections),
            "n_duplicates_removed": len(duplicates),
            "n_absolute_and_continuous": sum(
                bool(record.get("is_absolute_and_continuous")) for record in records
            ),
            "sources": {
                str(result.stats.get("source_id")): result.stats for result in results
            },
        },
        cleaned_records=[
            row for result in results for row in (result.cleaned_records or [])
        ],
        normalized_records=[
            row for result in results for row in (result.normalized_records or [])
        ],
    )


def _compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "normalized_record_id",
        "cleaned_record_id",
        "source_id",
        "source_name",
        "source_row_number",
        "source_record_id",
        "source_smiles",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "spacing_and_spelling_endpoint",
        "spacing_and_spelling_status",
        "spacing_and_spelling_reason",
        "spacing_and_spelling_version",
        "canonical_endpoint",
        "canonical_measurement",
        "canonical_unit",
        "canonical_bioavailability_report_type",
        "global_context",
        "global_species_context",
        "auxiliary_mapping_status",
        "auxiliary_attachment_version",
        "normalization_validity_status",
        "finite_scalar_value",
        "is_absolute_and_continuous",
        "absolute_and_continuous_value",
        "variation_value",
        "measurement_unit_status",
        "source_scalar_rule_version",
        "source_scalar_rule_id",
        "source_scalar_rule_reason",
        "scalar_semantic_label",
        "source_column_contract_version",
        "llm_source_contract_json",
        "llm_source_fields_json",
        "confidence",
        "support_text",
        "evidence_context_json",
        "duplicate_group_size",
        "duplicate_source_record_ids",
    )
    return {field: record.get(field) for field in fields}


def _llm_example(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        context = json.loads(str(record.get("evidence_context_json") or "{}"))
    except json.JSONDecodeError:
        context = {}
    try:
        source_contract = json.loads(
            str(record.get("llm_source_contract_json") or "{}")
        )
        source_fields = json.loads(
            str(record.get("llm_source_fields_json") or "{}")
        )
    except json.JSONDecodeError as exc:
        raise ValueError("invalid persisted LLM source projection") from exc
    if source_contract or source_fields:
        if not source_contract or not isinstance(source_fields, Mapping):
            raise ValueError("incomplete persisted LLM source projection")
        return {
            "source_contract": source_contract,
            "source_fields": source_fields,
        }
    return {
        "source_id": record.get("source_id"),
        "source_record_id": record.get("source_record_id"),
        # Named to match the minimal_evidence example contract that the
        # morganfingerprint/legacy field policies read (see evidence_contract.py).
        "endpoint_type": record.get("endpoint_name"),
        "reported_value": record.get("measurement_text"),
        "reported_units": record.get("unit_text"),
        "context": context,
        "support_text": record.get("support_text"),
        "source_confidence": record.get("confidence"),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


__all__ = [
    "aggregate_molecule_family_records",
    "combine_normalization_results",
    "deduplicate_within_source",
    "examples_text",
    "organize_normalized_records",
    "representative_examples",
]
