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
ASSAY_COMPACT_PROMPT_PROFILE = "assay_compact.v1"
ASSAY_COMPACT_V2_PROMPT_PROFILE = "assay_compact.v2"
ASSAY_RAW_CARD_PROMPT_PROFILE = "assay_compact.raw_v3"
_LEGACY_ASSAY_FLAT_GROUP_ID = "Flat.assay_ranked_evidence"
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
    endpoint_name = _first_text(
        row.get("standard_type"), row.get("endpoint_group"), row.get("endpoint")
    )
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
            "record_id": _first_text(
                row.get("source_record_id"), row.get("assay_chembl_id")
            ),
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
            "confidence": _json_scalar(
                row.get("confidence_score", row.get("confidence", ""))
            ),
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


def evidence_for_group_llm(
    row: Mapping[str, Any],
    group: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the prompt view declared by a retrieval group.

    Ordinary group-level retrieval keeps the full minimal-evidence contract.
    Dense assay-level replays use a bounded view.  The group-id fallback keeps
    already materialized v1 assay replay artifacts readable.
    """
    profile = str(group.get("evidence_prompt_profile") or "")
    if profile == ASSAY_COMPACT_V2_PROMPT_PROFILE:
        return assay_evidence_for_llm_v2(row)
    if profile == ASSAY_RAW_CARD_PROMPT_PROFILE:
        return assay_evidence_for_llm_raw_cards(row)
    if profile == ASSAY_COMPACT_PROMPT_PROFILE or (
        not profile and group.get("group_id") == _LEGACY_ASSAY_FLAT_GROUP_ID
    ):
        return assay_evidence_for_llm(row)
    if profile:
        raise ValueError(f"Unsupported evidence prompt profile: {profile}")
    return evidence_for_llm(row)


def assay_evidence_for_llm(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded minimal-evidence view for dense flat assay prompts.

    Every assay row remains present.  The bounded view removes redundant
    molecule/group fields already carried by the outer neighbor and flat group,
    and keeps a short source excerpt instead of repeating long normalized text.
    """
    evidence = minimal_evidence_from_row(row)
    endpoint = evidence.get("endpoint") or {}
    measurement = endpoint.get("measurement") or {}
    annotations = evidence.get("annotations") or {}
    scope = annotations.get("scope") or {}
    provenance = evidence.get("provenance") or {}
    compact = {
        "contract_version": CONTRACT_VERSION,
        "source": {
            "name": _bounded_text((evidence.get("source") or {}).get("name"), 80)
        },
        "assay_id": _bounded_text(provenance.get("assay_id"), 120),
        "endpoint": {
            "name": _bounded_text(endpoint.get("name"), 200),
            "measurement": {
                "relation": _bounded_text(measurement.get("relation"), 24),
                "value": _bounded_text(measurement.get("value"), 200),
                "unit": _bounded_text(measurement.get("unit"), 80),
            },
        },
        "text": {
            "evidence_excerpt": _bounded_text(
                (evidence.get("text") or {}).get("evidence"),
                160,
            ),
        },
        "annotations": {
            "evidence_role": _bounded_text(annotations.get("evidence_role"), 40),
            "scope": {
                "assay_context": _bounded_list(scope.get("assay_context"), 1, 240),
                "species_context": _bounded_list(scope.get("species_context"), 4, 80),
                "qualifying_conditions": _bounded_list(
                    scope.get("qualifying_conditions"), 4, 120
                ),
            },
            "uncertainty": _bounded_list(annotations.get("uncertainty"), 4, 120),
        },
        "quality": {
            "confidence": (evidence.get("quality") or {}).get("confidence", ""),
        },
        "provenance": {
            "source_record_count": provenance.get("source_record_count", ""),
        },
    }
    return _drop_empty(compact)


def assay_evidence_for_llm_v2(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return historical summary-backed record cards for dense assay prompts."""
    return _assay_record_cards(row, use_summary=True)


def assay_evidence_for_llm_raw_cards(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return semantic record cards with complete representative support text."""
    return _assay_record_cards(row, use_summary=False)


def _assay_record_cards(
    row: Mapping[str, Any],
    *,
    use_summary: bool,
) -> dict[str, Any]:
    """Build the shared v2/v3 assay card shape.

    Unlike ``assay_compact.v1``, this view does not repeat contract boilerplate,
    opaque assay identifiers, default role/transferability fields, confidence,
    or extraction diagnostics.  It preserves the endpoint/value/unit/species/
    condition/support pairing for each of at most three representative records.
    The historical v2 profile requires frozen summaries for long support; the
    current raw-v3 profile keeps every selected card field complete without
    rewriting or field-level truncation.
    """
    examples = [
        item
        for item in row.get("source_record_examples") or []
        if isinstance(item, Mapping)
    ][:3]
    cards = []
    for example in examples:
        raw_support = _text(example.get("support_text"))
        summary = _text(example.get("support_summary"))
        support = summary if use_summary and summary else raw_support
        if use_summary and len(raw_support) > 320 and not summary:
            raise ValueError(
                "assay_compact.v2 requires a frozen support_summary when "
                "support_text exceeds 320 characters"
            )
        cards.append(
            _drop_empty(
                {
                    "endpoint": _assay_card_text(
                        example.get("endpoint_type"), use_summary, 200
                    ),
                    "value": _assay_card_text(
                        example.get("reported_value"), use_summary, 200
                    ),
                    "unit": _assay_card_text(
                        example.get("reported_units"), use_summary, 80
                    ),
                    "species": _assay_card_text(
                        example.get("species_context"), use_summary, 120
                    ),
                    "conditions": _assay_card_text(
                        example.get("qualifying_conditions"), use_summary, 240
                    ),
                    "support": _bounded_text(support, 480) if use_summary else support,
                }
            )
        )
    if not cards:
        evidence = minimal_evidence_from_row(row)
        text = _text((evidence.get("text") or {}).get("evidence"))
        if use_summary and len(text) > 320:
            raise ValueError(
                "assay_compact.v2 evidence without representative records "
                "must not exceed 320 characters"
            )
        endpoint = evidence.get("endpoint") or {}
        measurement = endpoint.get("measurement") or {}
        cards = [
            _drop_empty(
                {
                    "endpoint": _assay_card_text(
                        endpoint.get("name"), use_summary, 200
                    ),
                    "value": _assay_card_text(
                        measurement.get("value"), use_summary, 200
                    ),
                    "unit": _assay_card_text(
                        measurement.get("unit"), use_summary, 80
                    ),
                    "support": text,
                }
            )
        ]

    scope = row.get("evidence_scope") or row.get("scope") or {}
    assay_context = ""
    if isinstance(scope, Mapping):
        contexts = _string_list(scope.get("assay_context"))
        assay_context = contexts[0] if contexts else ""
    assay_context = assay_context or _text(row.get("target_pref_name"))
    return _drop_empty(
        {
            "assay_context": _assay_card_text(assay_context, use_summary, 280),
            "records": cards,
            "source_record_count": _int_or_empty(row.get("source_record_count")),
        }
    )


def _assay_card_text(value: Any, use_summary: bool, legacy_limit: int) -> str:
    """Keep raw-v3 fields whole while preserving the frozen v2 bounds."""
    return _bounded_text(value, legacy_limit) if use_summary else _text(value)


def numeric_only_evidence_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a scalar-only evidence row, or None when no numeric value exists."""
    value = row.get("standard_value", row.get("value", ""))
    try:
        float(value)
    except (TypeError, ValueError):
        return None

    output = deepcopy(dict(row))
    output.pop("minimal_evidence", None)
    output.update(
        {
            "assay_description": "",
            "activity_comment": "",
            "target_pref_name": "",
            "target_genes": "",
            "organism": "",
            "relationship_type": "",
            "evidence_scope": {},
            "scope": {},
            "source_molecule_names": [],
            "source_support_texts": [],
            "source_qualitative_examples": [],
            "source_record_examples": _numeric_examples(row),
        }
    )
    return attach_minimal_evidence(output)


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
        record["examples"] = [
            _sanitize_mapping(item) for item in examples if isinstance(item, Mapping)
        ]
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
    return [_sanitize_mapping(item) for item in examples if isinstance(item, Mapping)][
        :6
    ]


def _numeric_examples(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed_fields = {
        "endpoint_type",
        "exposure_measure",
        "metric_type",
        "parameter_name",
        "oral_bioavailability_value_percent",
        "parameter_value",
        "reported_value",
        "measured_value",
        "reported_units",
        "parameter_units",
        "quant_value",
        "quant_units",
    }
    examples = [
        *(row.get("source_record_examples") or []),
        *(row.get("source_qualitative_examples") or []),
    ]
    return [
        {
            str(key): _json_scalar(value)
            for key, value in example.items()
            if str(key) in allowed_fields
        }
        for example in examples
        if isinstance(example, Mapping)
    ][:6]


def _sanitize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    output = _validated_source_projection(value)
    for key, item in value.items():
        key_text = str(key)
        if key_text == "source_fields" or key_text == "source_contract":
            continue
        if key_text.lower() in _PRIVATE_EXAMPLE_FIELDS:
            continue
        if isinstance(item, Mapping):
            output[key_text] = _sanitize_mapping(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            output[key_text] = [
                _sanitize_mapping(entry)
                if isinstance(entry, Mapping)
                else _json_scalar(entry)
                for entry in item
            ]
        else:
            output[key_text] = _json_scalar(item)
    return output


def _validated_source_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the allowlist before preserving otherwise-private source fields."""
    contract = value.get("source_contract")
    fields = value.get("source_fields")
    if contract is None and fields is None:
        return {}
    if not isinstance(contract, Mapping) or not isinstance(fields, Mapping):
        raise ValueError("source projection requires both source_contract and source_fields")
    if contract.get("contract_version") not in {
        "source_column_contract.v1",
        "source_column_contract.v2",
    }:
        raise ValueError("unsupported source-column contract version")
    allowed = contract.get("source_or_simply_cleaned")
    if not isinstance(allowed, Mapping) or any(item is not True for item in allowed.values()):
        raise ValueError("source-column contract must contain a Boolean-true allowlist")
    if set(map(str, fields)) != set(map(str, allowed)):
        raise ValueError("source fields do not exactly match the source-column allowlist")
    return {
        "source_contract": _sanitize_source_mapping(contract),
        "source_fields": _sanitize_source_mapping(fields),
    }


def _sanitize_source_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve fields admitted by an explicit source contract, including provenance."""
    output: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if isinstance(item, Mapping):
            output[key_text] = _sanitize_source_mapping(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            output[key_text] = [
                _sanitize_source_mapping(entry) if isinstance(entry, Mapping) else _json_scalar(entry)
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
    values = (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else [value]
    )
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


def _bounded_text(value: Any, limit: int) -> str:
    text = _text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _bounded_list(value: Any, limit: int, text_limit: int) -> list[str]:
    return [_bounded_text(item, text_limit) for item in _string_list(value)[:limit]]


def _drop_empty(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_empty(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _drop_empty(item)) not in (None, "", [], {})
        ]
    return value


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
