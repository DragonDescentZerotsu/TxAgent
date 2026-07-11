"""Harness-side tool prefetching that hides query identity from the LLM.

Retrieval still uses the molecular graph internally.  Before prompting, this
module computes the same property/comparison tools outside the model, removes
query and neighbor SMILES/identifiers, and exposes only tool text plus the
source evidence needed for reasoning.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.openai_reasoning_client import ToolServiceClient


_GENERIC_IDENTITY_NAMES = {
    "compound",
    "drug",
    "molecule",
    "net",
    "not specified",
    "test article",
    "test compound",
    "unknown",
}


def prepare_identity_blind_retrieval(
    retrieval: dict[str, Any],
    tool_service: ToolServiceClient,
) -> dict[str, Any]:
    """Return an LLM-facing retrieval copy with harness-prefetched tools."""
    output = deepcopy(retrieval)
    query = output.get("query") or {}
    query_smiles = str(query.get("canonical_smiles") or query.get("input_smiles") or "")
    property_result = tool_service.invoke(
        "molecule_properties",
        {"query_smiles": query_smiles, "logd_ph": 7.4},
    )
    query.clear()
    query.update(
        {
            "molecule_id": "query",
            "identity_hidden": True,
            "prefetched_molecule_properties": _compact_result(property_result, query_smiles),
        }
    )

    for group_index, group in enumerate(output.get("groups") or [], start=1):
        group["identity_blind"] = True
        for neighbor_index, neighbor in enumerate(group.get("neighbors") or [], start=1):
            reference_smiles = str(neighbor.get("canonical_smiles") or "")
            alias = f"neighbor_{group_index}_{neighbor_index}"
            comparisons = [
                tool_service.invoke(
                    "mmp_structure_compare",
                    {
                        "query_smiles": query_smiles,
                        "reference_smiles": reference_smiles,
                        "max_mmp_alternatives": 5,
                        "mcs_timeout_s": 5,
                    },
                ),
                tool_service.invoke(
                    "properties_compare",
                    {
                        "query_smiles": query_smiles,
                        "reference_smiles": reference_smiles,
                        "logd_ph": 7.4,
                    },
                ),
            ]
            neighbor["molecule_chembl_id"] = alias
            neighbor["canonical_smiles"] = "[hidden]"
            neighbor["standard_inchi_key"] = ""
            neighbor["identity_blind_alias"] = alias
            neighbor["prefetched_comparisons"] = [
                _compact_result(result, query_smiles, reference_smiles) for result in comparisons
            ]
            neighbor["evidence_rows"] = [
                {"minimal_evidence": _redact_evidence_identity(evidence_for_llm(row), alias)}
                for row in neighbor.get("evidence_rows") or []
            ]
    output = _replace_identity_terms(output, _retrieval_sensitive_terms(retrieval))
    output.setdefault("experiment", {})["identity_blind"] = True
    assert_identity_blind_retrieval(retrieval, output)
    return output


def prepare_identity_blind_final_retrieval(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
) -> dict[str, Any]:
    """Redact a persisted retrieval before a final-only retry."""
    output = deepcopy(retrieval)
    tool_results = ((single_output.get("llm") or {}).get("tool_results") or [])
    prefetched = tool_results[0] if tool_results else {}
    output["query"] = {
        "molecule_id": "query",
        "identity_hidden": True,
        "prefetched_molecule_properties": prefetched,
    }
    output.setdefault("experiment", {})["identity_blind"] = True
    return output


def sanitize_identity_blind_branch_outputs(
    branch_outputs: list[dict[str, Any]],
    retrieval: dict[str, Any],
) -> list[dict[str, Any]]:
    """Remove inferred source identities before downstream synthesis."""
    terms = _retrieval_sensitive_terms(retrieval)
    sanitized = _replace_identity_terms(deepcopy(branch_outputs), terms)
    serialized = str(sanitized)
    leaks = [
        term
        for term in terms
        if re.search(_identity_pattern(term), serialized, flags=re.IGNORECASE)
    ]
    if leaks:
        raise ValueError(f"Identity-blind branch sanitization failed: terms={len(leaks)}")
    changed = sanitized != branch_outputs
    for branch in sanitized:
        branch["identity_blind_sanitization"] = {
            "applied": True,
            "changed": changed,
            "n_sensitive_terms": len(terms),
        }
    return sanitized


def _compact_result(result: dict[str, Any], *hidden_values: str) -> dict[str, Any]:
    return {
        "tool_name": result.get("tool_name", ""),
        "status": result.get("status", "error"),
        "content": _hide_structures(str(result.get("content") or ""), hidden_values),
        "warnings": _hide_structures(result.get("warnings") or [], hidden_values),
        "errors": _hide_structures(result.get("errors") or [], hidden_values),
    }


def _hide_structures(value: Any, hidden_values: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for hidden in hidden_values:
            if hidden:
                value = re.sub(re.escape(hidden), "[hidden_structure]", value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [_hide_structures(item, hidden_values) for item in value]
    if isinstance(value, dict):
        return {key: _hide_structures(item, hidden_values) for key, item in value.items()}
    return value


def _redact_evidence_identity(record: dict[str, Any], alias: str) -> dict[str, Any]:
    record = deepcopy(record)
    identity_terms = _identity_terms(record)
    record = _replace_identity_terms(record, identity_terms)
    record["molecule"] = {"id": alias, "canonical_smiles": "", "names": []}
    source = record.get("source")
    if isinstance(source, dict):
        source["record_id"] = ""
    for example in record.get("examples") or []:
        if not isinstance(example, dict):
            continue
        for field in (
            "molecule_name",
            "global_identifier",
            "source_record_id",
            "source_index",
            "smiles",
            "canonical_smiles",
        ):
            example.pop(field, None)
    return record


def _identity_terms(record: dict[str, Any]) -> list[str]:
    molecule = record.get("molecule") or {}
    terms = [molecule.get("id"), molecule.get("canonical_smiles")]
    terms.extend(molecule.get("names") or [])
    for example in record.get("examples") or []:
        if isinstance(example, dict):
            terms.extend(
                example.get(field)
                for field in ("molecule_name", "smiles", "canonical_smiles")
            )
    return sorted(
        {
            str(term).strip()
            for term in terms
            if _is_identity_term(str(term).strip())
        },
        key=len,
        reverse=True,
    )


def _replace_identity_terms(value: Any, terms: list[str]) -> Any:
    if isinstance(value, str):
        for term in terms:
            value = re.sub(_identity_pattern(term), "[neighbor]", value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [_replace_identity_terms(item, terms) for item in value]
    if isinstance(value, dict):
        return {key: _replace_identity_terms(item, terms) for key, item in value.items()}
    return value


def assert_identity_blind_retrieval(
    original: dict[str, Any],
    redacted: dict[str, Any],
) -> None:
    """Fail before an LLM call when a known structure or identity remains."""
    leaks = find_identity_blind_leaks(original, redacted)
    if any(leaks.values()):
        raise ValueError(
            "Identity-blind preflight failed: "
            f"structures={len(leaks['structures'])}, identifiers={len(leaks['identifiers'])}, "
            f"names={len(leaks['names'])}"
        )


def find_identity_blind_leaks(
    original: dict[str, Any],
    payload: Any,
) -> dict[str, list[str]]:
    """Return known query/source identities present in an LLM-facing payload."""
    structures: set[str] = set()
    identifiers: set[str] = set()
    names = set(_retrieval_identity_names(original))
    query = original.get("query") or {}
    structures.update(
        str(query.get(field) or "").strip()
        for field in ("input_smiles", "canonical_smiles")
    )
    identifiers.add(str(query.get("standard_inchi_key") or "").strip())
    for group in original.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            structures.add(str(neighbor.get("canonical_smiles") or "").strip())
            identifiers.update(
                {
                    str(neighbor.get("molecule_chembl_id") or "").strip(),
                    str(neighbor.get("standard_inchi_key") or "").strip(),
                }
            )
    serialized = str(payload).lower()
    leaked_structures = sorted(
        term
        for term in structures
        if term and re.search(_structure_pattern(term), serialized, flags=re.IGNORECASE)
    )
    leaked_identifiers = sorted(
        term
        for term in identifiers
        if term and re.search(_identifier_pattern(term), serialized, flags=re.IGNORECASE)
    )
    leaked_names = [
        term
        for term in names
        if _is_identity_term(term) and re.search(_identity_pattern(term), serialized, flags=re.IGNORECASE)
    ]
    return {
        "structures": leaked_structures,
        "identifiers": leaked_identifiers,
        "names": sorted(leaked_names),
    }


def _retrieval_identity_names(retrieval: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for group in retrieval.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            for row in neighbor.get("evidence_rows") or []:
                record = evidence_for_llm(row)
                molecule = record.get("molecule") or {}
                names.update(str(name).strip() for name in molecule.get("names") or [])
                for example in record.get("examples") or []:
                    if isinstance(example, dict):
                        names.add(str(example.get("molecule_name") or "").strip())
    return sorted(
        {name for name in names if _is_identity_term(name)},
        key=len,
        reverse=True,
    )


def _retrieval_sensitive_terms(retrieval: dict[str, Any]) -> list[str]:
    terms = set(_retrieval_identity_names(retrieval))
    query = retrieval.get("query") or {}
    terms.update(
        str(query.get(field) or "").strip()
        for field in ("input_smiles", "canonical_smiles", "standard_inchi_key")
    )
    for group in retrieval.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            terms.update(
                str(neighbor.get(field) or "").strip()
                for field in ("molecule_chembl_id", "canonical_smiles", "standard_inchi_key")
            )
    return sorted(
        {term for term in terms if _is_identity_term(term)},
        key=len,
        reverse=True,
    )


def _is_identity_term(term: str) -> bool:
    normalized = term.strip().lower()
    return bool(
        len(normalized) >= 2
        and not normalized.isdigit()
        and normalized not in _GENERIC_IDENTITY_NAMES
    )


def _identity_pattern(term: str) -> str:
    escaped = re.escape(term)
    if len(term) <= 3 and term.isalnum():
        return rf"(?<!\w){escaped}(?!\w)"
    return escaped


def _identifier_pattern(term: str) -> str:
    escaped = re.escape(term)
    if term.isalnum():
        return rf"(?<!\w){escaped}(?!\w)"
    return escaped


def _structure_pattern(term: str) -> str:
    """Avoid treating short SMILES as substrings of ordinary words."""
    escaped = re.escape(term)
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
