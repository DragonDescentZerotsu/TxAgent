"""Minimal source-agnostic evidence records for molecular reasoning tasks.

The contract is intentionally descriptive. It normalizes heterogeneous source
rows for retrieval and prompting, but it never decides a task label or assigns
query-specific analog transferability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


CONTRACT_VERSION = "minimal_evidence.v1"
EVIDENCE_ROLES = {
    "direct_outcome",
    "surrogate_proxy",
    "mechanistic_factor",
    "context_modifier",
    "unspecified",
}
TRANSFERABILITY_STATES = {"not_assessed", "high", "moderate", "low", "not_applicable"}

_PRIVATE_EXAMPLE_FIELDS = {
    "pmid",
    "pmids",
    "source_pmids",
    "doi",
    "source_doi",
    "source_url",
}


def minimal_evidence_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a legacy ChEMBL-like or Starling row to the common contract."""
    existing = row.get("minimal_evidence")
    if isinstance(existing, Mapping):
        record = deepcopy(dict(existing))
        record["contract_version"] = CONTRACT_VERSION
        return _sanitize_record(record)

    evidence_source = _first_text(
        row.get("evidence_source"),
        row.get("source_dataset"),
        row.get("source"),
        "unknown",
    )
    molecule_id = _first_text(
        row.get("source_molecule_id"),
        row.get("molecule_chembl_id"),
        row.get("molecule_id"),
    )
    names = _string_list(row.get("source_molecule_names") or row.get("molecule_names"))
    endpoint_name = _first_text(row.get("standard_type"), row.get("endpoint_group"), row.get("endpoint"))
    evidence_text = _evidence_text(row)
    context_text = _context_text(row)
    uncertainty = _string_list(row.get("uncertainty"))
    validity_comment = _text(row.get("data_validity_comment"))
    if validity_comment and validity_comment not in uncertainty:
        uncertainty.append(validity_comment)

    evidence_role = _text(row.get("evidence_role")) or "unspecified"
    if evidence_role not in EVIDENCE_ROLES:
        evidence_role = "unspecified"
    transferability = _text(row.get("transferability")) or "not_assessed"
    if transferability not in TRANSFERABILITY_STATES:
        transferability = "not_assessed"

    record = {
        "contract_version": CONTRACT_VERSION,
        "source": {
            "name": evidence_source,
            "record_id": _first_text(row.get("source_record_id"), row.get("assay_chembl_id")),
        },
        "molecule": {
            "id": molecule_id,
            "canonical_smiles": _text(row.get("canonical_smiles")),
            "names": names,
        },
        "group": {
            "id": _text(row.get("group_id")),
            "tier": _text(row.get("assay_tier") or row.get("tier")),
            "endpoint_group": _text(row.get("endpoint_group")),
        },
        "endpoint": {
            "name": endpoint_name,
            "measurement": {
                "relation": _text(row.get("standard_relation") or row.get("relation")),
                "value": _json_scalar(row.get("standard_value", row.get("value", ""))),
                "unit": _text(row.get("standard_units") or row.get("unit")),
            },
        },
        "text": {
            "evidence": evidence_text,
            "context": context_text,
        },
        "annotations": {
            "evidence_role": evidence_role,
            "scope": _scope(row.get("scope") or row.get("evidence_scope")),
            "transferability": transferability,
            "uncertainty": uncertainty,
        },
        "quality": {
            "confidence": _json_scalar(row.get("confidence_score", row.get("confidence", ""))),
        },
        "provenance": {
            "assay_id": _text(row.get("assay_chembl_id")),
            "source_record_count": _int_or_empty(row.get("source_record_count")),
        },
        "examples": _safe_examples(row),
    }
    return _sanitize_record(record)


def attach_minimal_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Attach a normalized contract while preserving the source row for audit."""
    row.pop("minimal_evidence", None)
    row["minimal_evidence"] = minimal_evidence_from_row(row)
    return row


def evidence_for_llm(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the compact, source-agnostic contract for an LLM prompt."""
    return minimal_evidence_from_row(row)


def validate_minimal_evidence(record: Mapping[str, Any]) -> list[str]:
    """Return validation errors without imposing task-specific semantics."""
    errors: list[str] = []
    if record.get("contract_version") != CONTRACT_VERSION:
        errors.append("invalid_contract_version")
    source = record.get("source")
    if not isinstance(source, Mapping) or not _text(source.get("name")):
        errors.append("missing_source_name")
    molecule = record.get("molecule")
    if not isinstance(molecule, Mapping) or not (
        _text(molecule.get("id")) or _text(molecule.get("canonical_smiles"))
    ):
        errors.append("missing_molecule_identity")
    group = record.get("group")
    if not isinstance(group, Mapping) or not _text(group.get("id")):
        errors.append("missing_group_id")
    endpoint = record.get("endpoint")
    text = record.get("text")
    has_endpoint = isinstance(endpoint, Mapping) and bool(_text(endpoint.get("name")))
    has_evidence_text = isinstance(text, Mapping) and bool(_text(text.get("evidence")))
    if not has_endpoint and not has_evidence_text:
        errors.append("missing_endpoint_and_evidence_text")
    return errors


def _sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    examples = record.get("examples")
    if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes)):
        record["examples"] = [_sanitize_mapping(item) for item in examples if isinstance(item, Mapping)]
    annotations = record.get("annotations")
    if not isinstance(annotations, Mapping):
        record["annotations"] = {
            "evidence_role": "unspecified",
            "scope": {},
            "transferability": "not_assessed",
            "uncertainty": [],
        }
    return record


def _safe_examples(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    examples = [
        *(row.get("source_record_examples") or []),
        *(row.get("source_qualitative_examples") or []),
    ]
    return [_sanitize_mapping(item) for item in examples if isinstance(item, Mapping)][:6]


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text.lower() in _PRIVATE_EXAMPLE_FIELDS:
            continue
        if isinstance(item, Mapping):
            output[key_text] = _sanitize_mapping(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            output[key_text] = [
                _sanitize_mapping(entry) if isinstance(entry, Mapping) else _json_scalar(entry)
                for entry in item
            ]
        else:
            output[key_text] = _json_scalar(item)
    return output


def _evidence_text(row: Mapping[str, Any]) -> str:
    support_texts = _string_list(row.get("source_support_texts"))
    return _first_text(
        row.get("evidence_text"),
        row.get("assay_description"),
        "\n".join(support_texts),
        row.get("activity_comment"),
    )


def _context_text(row: Mapping[str, Any]) -> str:
    parts = []
    for label, value in (
        ("organism", row.get("organism")),
        ("target", row.get("target_pref_name")),
        ("target_genes", row.get("target_genes")),
        ("activity_comment", row.get("activity_comment")),
        ("relationship_type", row.get("relationship_type")),
        ("context", row.get("context_text")),
    ):
        text = _text(value)
        if text:
            parts.append(f"{label}: {text}")
    return "; ".join(parts)


def _scope(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    text = _text(value)
    return {"description": text} if text else {}


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else [value]
    output = []
    for item in values:
        text = _text(item)
        if text and text not in output:
            output.append(text)
    return output


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _json_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _int_or_empty(value: Any) -> int | str:
    try:
        if value in (None, ""):
            return ""
        return int(value)
    except (TypeError, ValueError):
        return ""
