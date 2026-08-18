"""Harness-side tool prefetching with optional identity redaction.

Retrieval still uses the molecular graph internally.  Before prompting, this
module computes the same property/comparison tools outside the model, removes
query and neighbor SMILES/identifiers, and exposes only tool text plus the
source evidence needed for reasoning.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from tools.chembl_tool.common.coverage_reasoning import (
    COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT,
    STANDARD_NEIGHBOR_CONTEXT,
    attach_mmp_coverage_ledger,
    attach_neighbor_context,
)
from tools.chembl_tool.common.evidence_contract import evidence_for_llm
from tools.chembl_tool.common.openai_reasoning_client import ToolServiceClient


_GENERIC_IDENTITY_NAMES = {
    "compound",
    "drug",
    "molecule",
    "net",
    "not specified",
    "per",
    "test article",
    "test compound",
    "unknown",
}


def prepare_harness_prefetched_retrieval(
    retrieval: dict[str, Any],
    tool_service: ToolServiceClient,
    *,
    identity_blind: bool,
    include_query_tools: bool = True,
    include_neighbor_tools: bool | None = None,
) -> dict[str, Any]:
    """Prefetch the same fixed tools, optionally hiding molecule identities."""
    if include_neighbor_tools is None:
        include_neighbor_tools = include_query_tools
    if include_neighbor_tools and not include_query_tools:
        raise ValueError("Neighbor comparison tools require query tools")
    output = deepcopy(retrieval)
    query = output.get("query") or {}
    query_smiles = str(query.get("canonical_smiles") or query.get("input_smiles") or "")
    calls: list[tuple[str, dict[str, Any]]] = []
    if include_query_tools:
        calls.append(
            ("molecule_properties", {"query_smiles": query_smiles, "logd_ph": 7.4})
        )
    neighbor_smiles: list[str] = []
    for group in output.get("groups") or []:
        for neighbor in group.get("neighbors") or []:
            reference_smiles = str(neighbor.get("canonical_smiles") or "")
            neighbor_smiles.append(reference_smiles)
            if include_neighbor_tools:
                calls.extend(
                    [
                        (
                            "mmp_structure_compare",
                            {
                                "query_smiles": query_smiles,
                                "reference_smiles": reference_smiles,
                                "max_mmp_alternatives": 5,
                                "mcs_timeout_s": 5,
                            },
                        ),
                        (
                            "properties_compare",
                            {
                                "query_smiles": query_smiles,
                                "reference_smiles": reference_smiles,
                                "logd_ph": 7.4,
                            },
                        ),
                    ]
                )
    invoke_many = getattr(tool_service, "invoke_many", None)
    results = (
        []
        if not calls
        else invoke_many(calls)
        if callable(invoke_many)
        else [tool_service.invoke(tool_name, arguments) for tool_name, arguments in calls]
    )
    if len(results) != len(calls):
        raise ValueError(f"Prefetched tool result count mismatch: {len(results)} != {len(calls)}")
    prefetched_properties = (
        _compact_result(results[0], query_smiles) if include_query_tools else {}
    )
    if identity_blind:
        query.clear()
        query.update({
            "molecule_id": "query",
            "identity_hidden": True,
        })
        if include_query_tools:
            query["tools_prefetched"] = True
            query["prefetched_molecule_properties"] = prefetched_properties
    else:
        if include_query_tools:
            query["tools_prefetched"] = True
            query["prefetched_molecule_properties"] = prefetched_properties

    result_index = 1 if include_query_tools else 0
    neighbor_smiles_index = 0
    for group_index, group in enumerate(output.get("groups") or [], start=1):
        if include_neighbor_tools:
            group["tools_prefetched"] = True
        if identity_blind:
            group["identity_blind"] = True
        for neighbor_index, neighbor in enumerate(group.get("neighbors") or [], start=1):
            reference_smiles = neighbor_smiles[neighbor_smiles_index]
            neighbor_smiles_index += 1
            alias = f"neighbor_{group_index}_{neighbor_index}"
            if include_neighbor_tools:
                comparisons = results[result_index : result_index + 2]
                result_index += 2
                neighbor["prefetched_comparisons"] = [
                    _compact_result(result, query_smiles, reference_smiles)
                    for result in comparisons
                ]
            if identity_blind:
                neighbor["molecule_chembl_id"] = alias
                neighbor["canonical_smiles"] = "[hidden]"
                neighbor["standard_inchi_key"] = ""
                neighbor["identity_blind_alias"] = alias
                neighbor["evidence_rows"] = [
                    {"minimal_evidence": _redact_evidence_identity(evidence_for_llm(row), alias)}
                    for row in neighbor.get("evidence_rows") or []
                ]
    experiment = output.setdefault("experiment", {})
    experiment["tool_execution_mode"] = (
        "harness_prefetch"
        if include_neighbor_tools
        else "harness_prefetch_query_only"
        if include_query_tools
        else "omitted"
    )
    if identity_blind:
        output = _replace_identity_terms(output, _retrieval_sensitive_terms(retrieval))
        output.setdefault("experiment", {})["identity_blind"] = True
        assert_identity_blind_retrieval(retrieval, output)
    return output


def prepare_identity_blind_retrieval(
    retrieval: dict[str, Any],
    tool_service: ToolServiceClient,
) -> dict[str, Any]:
    """Return a redacted LLM-facing copy with fixed prefetched tools."""
    return prepare_harness_prefetched_retrieval(retrieval, tool_service, identity_blind=True)


def prepare_reasoning_retrieval(
    retrieval: dict[str, Any],
    tool_service: ToolServiceClient,
    *,
    identity_blind: bool,
    harness_prefetch_tools: bool,
    prefetched_tool_replay_run_dir: str = "",
    neighbor_context_profile: str = STANDARD_NEIGHBOR_CONTEXT,
    include_query_tools: bool = True,
    include_neighbor_tools: bool | None = None,
) -> dict[str, Any]:
    """Apply the requested paper tool-execution contract to retrieval."""
    if not include_query_tools and prefetched_tool_replay_run_dir:
        raise ValueError("Query-tool omission cannot use prefetched tool replay")
    if not include_query_tools and neighbor_context_profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT:
        raise ValueError(
            "coverage_mmp_ledger requires query comparison tools and is incompatible "
            "with query-tool omission"
        )
    if neighbor_context_profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT and identity_blind:
        raise ValueError(
            "coverage_mmp_ledger is visible-only and cannot be used with identity_blind"
        )
    reasoning_input = attach_neighbor_context(
        retrieval,
        profile=neighbor_context_profile,
    )
    if neighbor_context_profile == COVERAGE_MMP_LEDGER_NEIGHBOR_CONTEXT:
        reasoning_input = attach_mmp_coverage_ledger(reasoning_input, tool_service)
    if identity_blind:
        return prepare_harness_prefetched_retrieval(
            reasoning_input,
            tool_service,
            identity_blind=True,
            include_query_tools=include_query_tools,
            include_neighbor_tools=include_neighbor_tools,
        )
    if not include_query_tools:
        reasoning_input.setdefault("experiment", {})["tool_execution_mode"] = "omitted"
        return reasoning_input
    if prefetched_tool_replay_run_dir:
        return prepare_replayed_prefetched_retrieval(
            reasoning_input,
            prefetched_tool_replay_run_dir,
        )
    if harness_prefetch_tools:
        return prepare_harness_prefetched_retrieval(
            reasoning_input,
            tool_service,
            identity_blind=False,
            include_neighbor_tools=include_neighbor_tools,
        )
    return reasoning_input


def query_without_prefetched_tools(query: dict[str, Any]) -> dict[str, Any]:
    """Return the query identity surface without any harness tool payload."""
    output = deepcopy(query)
    output.pop("tools_prefetched", None)
    output.pop("prefetched_molecule_properties", None)
    return output


def expose_neighbor_smiles_only(
    reasoning_retrieval: dict[str, Any],
    source_retrieval: dict[str, Any],
) -> dict[str, Any]:
    """Expose canonical neighbor structures while keeping the query identity blind.

    This is the narrow prompt view used by tool-free analogous reasoning.  Evidence
    rows stay redacted and the query remains the anonymous ``query`` object.
    """
    output = deepcopy(reasoning_retrieval)
    query = output.get("query") or {}
    if not query.get("identity_hidden"):
        raise ValueError("Neighbor-SMILES prompt view requires an identity-blind query")
    source_groups = source_retrieval.get("groups") or []
    visible_groups = output.get("groups") or []
    if len(source_groups) != len(visible_groups):
        raise ValueError("Neighbor-SMILES prompt view group count mismatch")
    for source_group, visible_group in zip(source_groups, visible_groups, strict=True):
        if source_group.get("group_id") != visible_group.get("group_id"):
            raise ValueError("Neighbor-SMILES prompt view group ordering mismatch")
        source_neighbors = source_group.get("neighbors") or []
        visible_neighbors = visible_group.get("neighbors") or []
        if len(source_neighbors) != len(visible_neighbors):
            raise ValueError("Neighbor-SMILES prompt view neighbor count mismatch")
        for source_neighbor, visible_neighbor in zip(
            source_neighbors, visible_neighbors, strict=True
        ):
            smiles = str(source_neighbor.get("canonical_smiles") or "")
            if not smiles:
                raise ValueError("Retrieved neighbor is missing canonical_smiles")
            visible_neighbor["canonical_smiles"] = smiles
    experiment = output.setdefault("experiment", {})
    experiment["prompt_identity_view"] = "query_blind_neighbor_smiles"
    return output


def prepare_replayed_prefetched_retrieval(
    retrieval: dict[str, Any],
    source_run_dir: str,
) -> dict[str, Any]:
    """Attach frozen harness tool outputs while retaining visible identities."""

    source_dir = Path(source_run_dir)
    single = json.loads(
        (source_dir / "single_molecule_reasoning_output.json").read_text(encoding="utf-8")
    )
    query_results = ((single.get("llm") or {}).get("tool_results") or [])
    if not query_results:
        raise ValueError(f"Prefetched tool replay has no query tool result: {source_dir}")

    source_groups: dict[str, dict[int, list[dict[str, Any]]]] = {}
    raw_groups = source_dir / "group_reasoning_outputs_raw.jsonl"
    group_path = raw_groups if raw_groups.exists() else source_dir / "group_reasoning_outputs.jsonl"
    for line in group_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        branch = json.loads(line)
        payload = _initial_group_payload((branch.get("llm") or {}).get("messages") or [])
        group_id = str((payload.get("group") or {}).get("group_id") or branch.get("group_id") or "")
        if not group_id:
            continue
        source_groups[group_id] = {
            int(neighbor.get("rank")): deepcopy(neighbor.get("prefetched_comparisons") or [])
            for neighbor in payload.get("neighbors") or []
            if neighbor.get("rank") is not None
        }

    output = deepcopy(retrieval)
    query = output.setdefault("query", {})
    query["tools_prefetched"] = True
    query["prefetched_molecule_properties"] = deepcopy(query_results[0])
    for group in output.get("groups") or []:
        neighbors = group.get("neighbors") or []
        if not neighbors:
            continue
        group_id = str(group.get("group_id") or "")
        by_rank = source_groups.get(group_id)
        if by_rank is None:
            raise ValueError(f"Prefetched tool replay is missing group {group_id}: {source_dir}")
        group["tools_prefetched"] = True
        for neighbor in neighbors:
            rank = int(neighbor.get("rank"))
            comparisons = by_rank.get(rank)
            if not comparisons:
                raise ValueError(
                    f"Prefetched tool replay is missing {group_id} neighbor rank {rank}: {source_dir}"
                )
            neighbor["prefetched_comparisons"] = deepcopy(comparisons)
    experiment = output.setdefault("experiment", {})
    experiment["tool_execution_mode"] = "harness_prefetch_replay"
    experiment["prefetched_tool_replay_source_run_dir"] = str(source_dir)
    return output


def _initial_group_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            object_start = content.find("{")
            if object_start < 0:
                continue
            try:
                payload, _ = json.JSONDecoder().raw_decode(content[object_start:])
            except json.JSONDecodeError:
                continue
        if isinstance(payload, dict) and "neighbors" in payload:
            return payload
    return {}


def prepare_prefetched_final_retrieval(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    *,
    identity_blind: bool,
) -> dict[str, Any]:
    """Restore prefetched query properties for a final-only retry."""
    output = deepcopy(retrieval)
    tool_results = ((single_output.get("llm") or {}).get("tool_results") or [])
    prefetched = tool_results[0] if tool_results else {}
    if identity_blind:
        output["query"] = {
            "molecule_id": "query",
            "identity_hidden": True,
            "tools_prefetched": True,
            "prefetched_molecule_properties": prefetched,
        }
        output.setdefault("experiment", {})["identity_blind"] = True
    else:
        output.setdefault("query", {})["tools_prefetched"] = True
        output["query"]["prefetched_molecule_properties"] = prefetched
    output.setdefault("experiment", {})["tool_execution_mode"] = "harness_prefetch"
    return output


def prepare_identity_blind_final_retrieval(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
) -> dict[str, Any]:
    """Redact a persisted retrieval before a final-only retry."""
    return prepare_prefetched_final_retrieval(retrieval, single_output, identity_blind=True)


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
    tool_name = str(result.get("tool_name") or "")
    status = str(result.get("status") or "error")
    if status != "ok":
        # Backend exceptions can contain mmpdb fragments, raw structures, or
        # implementation details that are neither evidence nor safe input for
        # an identity-blind prompt.  Preserve an auditable failure receipt but
        # expose no exception payload to the model.
        return {
            "tool_name": tool_name,
            "status": "error",
            "content": (
                f"[{tool_name}]\nTool result unavailable. "
                "Treat this tool comparison as missing evidence."
            ),
            "warnings": [],
            "errors": [
                {
                    "code": "TOOL_RESULT_UNAVAILABLE",
                    "message": "Backend error details redacted by the prompt harness.",
                    "recoverable": False,
                }
            ],
        }
    return {
        "tool_name": tool_name,
        "status": status,
        "content": _hide_structures(str(result.get("content") or ""), hidden_values),
        "warnings": _hide_structures(result.get("warnings") or [], hidden_values),
        "errors": _hide_structures(result.get("errors") or [], hidden_values),
    }


def _hide_structures(value: Any, hidden_values: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for hidden in hidden_values:
            # A one-character atom-only SMILES (for example ``N``) cannot be
            # distinguished from ordinary prose or serialized control text.
            # Replacing it would corrupt every matching letter in the tool
            # summary.  The query identity is removed structurally below; only
            # multi-character structure strings are safe to redact in text.
            if _is_auditable_structure_term(hidden):
                value = re.sub(
                    _structure_pattern(hidden),
                    "[hidden_structure]",
                    value,
                    flags=re.IGNORECASE,
                )
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
    # Structure strings and identifiers are never valid schema keys, so audit
    # the complete structured payload for those.  Drug-name abbreviations can
    # legitimately collide with a fixed schema key (for example ``FA`` versus
    # ``Fa.absorption_solubility_permeability``), so audit names only in JSON
    # values after the identity-bearing values have been redacted.
    serialized = str(payload).lower()
    serialized_values = "\n".join(_payload_string_values(payload)).lower()
    leaked_structures = sorted(
        term
        for term in structures
        if _is_auditable_structure_term(term)
        and re.search(_structure_pattern(term), serialized, flags=re.IGNORECASE)
    )
    leaked_identifiers = sorted(
        term
        for term in identifiers
        if term and re.search(_identifier_pattern(term), serialized, flags=re.IGNORECASE)
    )
    leaked_names = [
        term
        for term in names
        if _is_identity_term(term)
        and re.search(_identity_pattern(term), serialized_values, flags=re.IGNORECASE)
    ]
    return {
        "structures": leaked_structures,
        "identifiers": leaked_identifiers,
        "names": sorted(leaked_names),
    }


def _payload_string_values(payload: Any) -> list[str]:
    """Collect LLM-visible string values without treating schema keys as data."""
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        values: list[str] = []
        for value in payload.values():
            values.extend(_payload_string_values(value))
        return values
    if isinstance(payload, (list, tuple)):
        values = []
        for value in payload:
            values.extend(_payload_string_values(value))
        return values
    return []


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
    if term.isalnum():
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


def _is_auditable_structure_term(term: str) -> bool:
    """Whether a structure token is distinct enough for text leak matching.

    A canonical SMILES containing only one atom is also a normal chemical
    token in assay prose (for example ``Cl`` in ``36Cl influx``).  Treating it
    as an identity leak either corrupts evidence text or produces a false
    preflight failure.  Multi-atom structures such as ``C[Se]`` remain fully
    auditable.
    """
    normalized = str(term).strip()
    if len(normalized) < 2:
        return False
    return re.fullmatch(r"(?:[A-Z][a-z]?|\[[^\[\]]+\])", normalized) is None
