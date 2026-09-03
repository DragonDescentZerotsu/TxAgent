"""Build the branch harness's model-visible query and assay-context payloads.

Inputs are internal query, exact-match, and paired-activity dictionaries;
outputs contain only fields allowed in branch prompts. Condition wording is
delegated to ``predict.llm_io.query`` so progressive uses the same rendering.
"""

from __future__ import annotations

from typing import Any

from predict.llm_io.query import external_condition_sentence


def attach_external_condition(
    retrieval: dict[str, Any],
    query_record: dict[str, Any],
) -> dict[str, Any]:
    """Attach only the natural-language condition sentence to the retrieval query."""
    query = retrieval.setdefault("query", {})
    query.pop("external_condition", None)
    sentence = external_condition_sentence(query_record)
    if sentence:
        query["external_condition"] = sentence
    return retrieval


def llm_query_payload(query: dict[str, Any]) -> dict[str, Any]:
    """Expose only the query fields allowed by the active identity contract."""
    if query.get("identity_hidden"):
        return {
            "molecule_id": "query",
            "identity_hidden": True,
            "prefetched_molecule_properties": query.get(
                "prefetched_molecule_properties"
            )
            or {},
        }
    payload = {
        "input_smiles": query.get("input_smiles", ""),
        "canonical_smiles": query.get("canonical_smiles", ""),
    }
    if query.get("prefetched_molecule_properties"):
        payload["tools_prefetched"] = True
        payload["prefetched_molecule_properties"] = query[
            "prefetched_molecule_properties"
        ]
    return payload


def llm_evidence_query_payload(query: dict[str, Any]) -> dict[str, Any]:
    """Expose the condition only to analog-evidence and final-decision branches."""
    payload = llm_query_payload(query)
    condition = str(query.get("external_condition") or "").strip()
    if condition:
        payload["external_condition"] = condition
    return payload


def clean_exact_match(match: dict[str, Any]) -> dict[str, Any]:
    """Keep the frozen molecule metadata surface for exact ChEMBL matches."""
    fields = (
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_inchi_key",
        "mw_freebase",
        "alogp",
        "hba",
        "hbd",
        "psa",
        "rtb",
        "num_ro5_violations",
        "full_mwt",
        "aromatic_rings",
        "heavy_atoms",
        "qed_weighted",
        "full_molformula",
        "np_likeness_score",
    )
    return {field: match.get(field, "") for field in fields}


def clean_shared_assay_context(context: dict[str, Any]) -> dict[str, Any]:
    """Keep paired activity values without leaking source-internal fields."""
    return {
        "same_endpoint_activity": [
            _clean_shared_activity_card(card)
            for card in context.get("same_endpoint_activity", [])
        ],
        "same_assay_different_endpoint_activity": [
            _clean_shared_activity_card(card)
            for card in context.get("same_assay_different_endpoint_activity", [])
        ],
        "n_same_endpoint_activity": context.get("n_same_endpoint_activity", 0),
        "n_same_assay_different_endpoint_activity": context.get(
            "n_same_assay_different_endpoint_activity", 0
        ),
    }


def _clean_shared_activity_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "assay_chembl_id": card.get("assay_chembl_id", ""),
        "standard_type_match": card.get("standard_type_match", False),
        "units_match": card.get("units_match", False),
        "query_activity": _clean_activity_value(card.get("query_activity") or {}),
        "neighbor_activity": _clean_activity_value(card.get("neighbor_activity") or {}),
    }


def _clean_activity_value(activity: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "assay_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "activity_comment",
        "data_validity_comment",
        "standard_text_value",
        "action_type",
    )
    return {field: activity.get(field, "") for field in fields}
