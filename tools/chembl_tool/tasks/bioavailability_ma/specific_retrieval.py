"""Bioavailability-specific retrieval regrouping for Fa/Fg/Fh reasoning.

The underlying neighbor indices are still source-specific and may contain the
older ChEMBL Tier.endpoint_group buckets or the Starling direct-F bucket. This
module rewrites those retrieved rows into a compact Bioavailability_Ma view:
observed direct/systemic F context plus Fa, Fg, and Fh factor evidence.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.retrieve_neighbors import (
    load_index,
    retrieve_neighbors as retrieve_source_neighbors,
)


DEFAULT_STARLING_INDEX = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling/"
    "starling_oral_bioavailability_neighbor_index.pkl"
)
SPECIFIC_RETRIEVAL_VERSION = "bioavailability_specific_retrieval_fa_fg_fh_v1"
SPECIFIC_GROUP_NEIGHBOR_CAP = 12
SPECIFIC_ROWS_PER_NEIGHBOR_CAP = 8

OBSERVED_F_GROUP_ID = "Observed.direct_parent_f_and_systemic_exposure"
FA_GROUP_ID = "Fa.absorption_solubility_permeability"
FG_GROUP_ID = "Fg.gut_wall_efflux_intestinal_metabolism"
FH_GROUP_ID = "Fh.hepatic_clearance_metabolic_stability"

SPECIFIC_GROUPS = {
    OBSERVED_F_GROUP_ID: {
        "tier": "Observed",
        "endpoint_group": "direct_parent_f_and_systemic_exposure",
        "specific_group_role": "direct_parent_f_or_systemic_exposure_context",
        "factor": "observed_F_product",
        "description": (
            "Direct absolute oral F analog evidence plus oral AUC/Cmax exposure context. "
            "Only clean parent absolute F can act as direct threshold evidence."
        ),
    },
    FA_GROUP_ID: {
        "tier": "Fa",
        "endpoint_group": "absorption_solubility_permeability",
        "specific_group_role": "Fa_absorbed_fraction",
        "factor": "Fa",
        "description": "Dissolution, solubility, permeability, intestinal absorption, and absorptive transport.",
    },
    FG_GROUP_ID: {
        "tier": "Fg",
        "endpoint_group": "gut_wall_efflux_intestinal_metabolism",
        "specific_group_role": "Fg_gut_escape",
        "factor": "Fg",
        "description": "Gut-wall metabolism, intestinal CYP/P-gp/BCRP/MRP efflux, and food/formulation gut effects.",
    },
    FH_GROUP_ID: {
        "tier": "Fh",
        "endpoint_group": "hepatic_clearance_metabolic_stability",
        "specific_group_role": "Fh_hepatic_escape",
        "factor": "Fh",
        "description": "Hepatic clearance, extraction, microsomal/hepatocyte stability, CYP/UGT metabolism.",
    },
}

_SPECIFIC_GROUP_FILTERS = set(SPECIFIC_GROUPS)
_SPECIFIC_GROUP_FILTERS.update({value["tier"] for value in SPECIFIC_GROUPS.values()})


def retrieve_specific_neighbors(
    query_smiles: str,
    index: dict[str, Any],
    *,
    top_k_per_group: int = 3,
    min_similarity: float = 0.3,
    groups: list[str] | None = None,
    starling_index_path: str = DEFAULT_STARLING_INDEX,
    specific_neighbor_cap: int = SPECIFIC_GROUP_NEIGHBOR_CAP,
) -> dict[str, Any]:
    """Retrieve from source indices and regroup rows into specific factor groups."""
    source_group_filters = _source_group_filters(groups)
    base_retrieval = retrieve_source_neighbors(
        query_smiles,
        index,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        groups=source_group_filters,
    )
    if base_retrieval.get("status") != "ok":
        return base_retrieval

    source_retrievals = [base_retrieval]
    starling_retrieval = _retrieve_starling_if_needed(
        query_smiles=query_smiles,
        base_index=index,
        starling_index_path=starling_index_path,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        requested_groups=groups,
    )
    if starling_retrieval:
        source_retrievals.append(starling_retrieval)

    return regroup_retrieval_to_specific(
        base_retrieval,
        extra_retrievals=source_retrievals[1:],
        groups=groups,
        specific_neighbor_cap=specific_neighbor_cap,
    )


def regroup_retrieval_to_specific(
    retrieval: dict[str, Any],
    *,
    extra_retrievals: list[dict[str, Any]] | None = None,
    groups: list[str] | None = None,
    specific_neighbor_cap: int = SPECIFIC_GROUP_NEIGHBOR_CAP,
) -> dict[str, Any]:
    """Rewrite source retrieval groups into observed-F/Fa/Fg/Fh groups."""
    source_retrievals = [retrieval, *(extra_retrievals or [])]
    grouped_neighbors: dict[str, dict[str, dict[str, Any]]] = {group_id: {} for group_id in SPECIFIC_GROUPS}

    for source_retrieval in source_retrievals:
        for source_group in source_retrieval.get("groups") or []:
            source_group_id = str(source_group.get("group_id") or "")
            for neighbor in source_group.get("neighbors") or []:
                row_assignments: list[tuple[str, dict[str, Any]]] = []
                for row in neighbor.get("evidence_rows") or []:
                    for assignment in _assign_specific_groups(source_group_id, row):
                        row_assignments.append((assignment.group_id, _row_with_specific_metadata(row, source_group, assignment)))
                if not row_assignments:
                    continue
                for specific_group_id, row_copy in row_assignments:
                    neighbor_key = _neighbor_key(neighbor)
                    bucket = grouped_neighbors[specific_group_id].setdefault(
                        neighbor_key,
                        _empty_neighbor_for_specific_group(neighbor),
                    )
                    bucket["evidence_rows"].append(row_copy)
                    bucket["source_group_ids"].add(source_group_id)

    specific_groups = []
    requested_specific_groups = _specific_group_filters(groups)
    for group_id, metadata in SPECIFIC_GROUPS.items():
        if requested_specific_groups and group_id not in requested_specific_groups and metadata["tier"] not in requested_specific_groups:
            continue
        neighbors = _rank_specific_neighbors(
            list(grouped_neighbors[group_id].values()),
            cap=max(1, specific_neighbor_cap),
        )
        specific_groups.append(
            {
                "group_id": group_id,
                "tier": metadata["tier"],
                "endpoint_group": metadata["endpoint_group"],
                "specific_group_role": metadata["specific_group_role"],
                "bioavailability_factor": metadata["factor"],
                "specific_group_description": metadata["description"],
                "n_candidate_molecules": len(grouped_neighbors[group_id]),
                "neighbors": neighbors,
            }
        )

    n_neighbors_total = sum(len(group.get("neighbors") or []) for group in specific_groups)
    result = {
        "status": "ok",
        "evidence_source": _specific_evidence_source(source_retrievals),
        "query": retrieval.get("query") or {},
        "groups": specific_groups,
        "coverage": {
            "n_groups": len(specific_groups),
            "n_groups_with_neighbors": sum(1 for group in specific_groups if group.get("neighbors")),
            "n_neighbors_total": n_neighbors_total,
            "min_similarity": (retrieval.get("coverage") or {}).get("min_similarity"),
            "top_k_per_source_group": (retrieval.get("coverage") or {}).get("top_k_per_group"),
            "specific_neighbor_cap": specific_neighbor_cap,
            "specific_retrieval_version": SPECIFIC_RETRIEVAL_VERSION,
        },
        "source_retrieval_coverage": [
            {
                "evidence_source": source.get("evidence_source") or {},
                "coverage": source.get("coverage") or {},
                "group_ids": [group.get("group_id") for group in source.get("groups") or []],
            }
            for source in source_retrievals
        ],
    }
    return result


class _Assignment:
    def __init__(self, group_id: str, evidence_role: str):
        self.group_id = group_id
        self.evidence_role = evidence_role


def _assign_specific_groups(source_group_id: str, row: dict[str, Any]) -> list[_Assignment]:
    text = _row_text(source_group_id, row)
    assignments: list[_Assignment] = []

    if _is_direct_f_source(source_group_id, text):
        assignments.append(_Assignment(OBSERVED_F_GROUP_ID, "direct_absolute_oral_f_candidate"))
        return assignments
    if _is_oral_exposure_source(source_group_id, text):
        assignments.append(_Assignment(OBSERVED_F_GROUP_ID, "oral_exposure_proxy_not_direct_f"))
        return assignments

    if _is_fa_source(source_group_id, text):
        assignments.append(_Assignment(FA_GROUP_ID, "fa_absorption_solubility_permeability"))
    if _is_fg_source(source_group_id, text):
        assignments.append(_Assignment(FG_GROUP_ID, "fg_gut_wall_efflux_or_intestinal_metabolism"))
    if _is_fh_source(source_group_id, text):
        assignments.append(_Assignment(FH_GROUP_ID, "fh_hepatic_clearance_or_metabolism"))

    return _dedupe_assignments(assignments)


def _is_direct_f_source(source_group_id: str, text: str) -> bool:
    if "direct_absolute_bioavailability" in source_group_id:
        return True
    if source_group_id == "Starling.direct_oral_bioavailability":
        return True
    if source_group_id == "Combined.chembl_tier1_and_starling":
        return True
    if "direct_oral_bioavailability" in text:
        return True
    return _has_any(text, {"absolute oral bioavailability", "oral bioavailability", "bioavailability (%)"}) and _has_any(
        text,
        {" f ", " f%", "%f", "bioavailability"},
    )


def _is_oral_exposure_source(source_group_id: str, text: str) -> bool:
    if source_group_id.endswith(".oral_auc_exposure") or source_group_id.endswith(".oral_cmax_exposure"):
        return True
    return _has_any(text, {"oral auc", "oral cmax", "plasma exposure after oral", "oral exposure"})


def _is_fa_source(source_group_id: str, text: str) -> bool:
    if source_group_id.endswith(
        (
            ".absorption_fraction_or_hia",
            ".in_vivo_intestinal_permeability",
            ".in_vivo_intestinal_uptake_or_transport",
            ".cell_permeability_papp",
            ".pampa_or_artificial_membrane",
            ".solubility",
            ".dissolution",
            ".gi_or_chemical_stability",
        )
    ):
        return True
    return _has_any(
        text,
        {
            "fraction absorbed",
            "human intestinal absorption",
            "permeability",
            "papp",
            "pampa",
            "solubility",
            "dissolution",
            "absorptive transport",
            "gi stability",
            "intestinal stability",
        },
    )


def _is_fg_source(source_group_id: str, text: str) -> bool:
    if source_group_id.endswith(
        (
            ".cell_bidirectional_efflux_ratio",
            ".cell_secretory_permeability",
            ".transporter_substrate_or_efflux",
            ".food_effect_or_fed_fasted",
            ".relative_bioavailability_or_formulation",
            ".formulation_auc_cmax_ratio",
        )
    ):
        return True
    if source_group_id.endswith(".first_pass_or_extraction") and _has_any(
        text,
        {"gut", "intestinal", "enterocyte", "presystemic", "cyp3a", "p-gp", "pgp", "efflux"},
    ):
        return True
    return _has_any(
        text,
        {
            "gut wall",
            "intestinal metabolism",
            "enterocyte",
            "presystemic",
            "p-gp",
            "pgp",
            "bcrp",
            "mrp",
            "efflux",
            "secretory",
            "food effect",
            "fed fasted",
            "fed/fasted",
            "high fat meal",
            "formulation",
            "salt form",
            "solid dispersion",
            "cyp3a",
        },
    )


def _is_fh_source(source_group_id: str, text: str) -> bool:
    if source_group_id.endswith(
        (
            ".intrinsic_or_hepatic_clearance",
            ".metabolic_stability",
            ".first_pass_or_extraction",
        )
    ):
        return True
    return _has_any(
        text,
        {
            "hepatic",
            "liver",
            "clearance",
            "clint",
            "extraction ratio",
            "microsomal",
            "microsome",
            "hepatocyte",
            "metabolic stability",
            "substrate depletion",
            "cyp",
            "ugt",
            "glucuronid",
            "first pass",
            "first-pass",
        },
    )


def _row_with_specific_metadata(
    row: dict[str, Any],
    source_group: dict[str, Any],
    assignment: _Assignment,
) -> dict[str, Any]:
    row_copy = copy.deepcopy(row)
    source_group_id = str(source_group.get("group_id") or row.get("group_id") or "")
    row_copy.setdefault("original_group_id", source_group_id)
    row_copy.setdefault("original_tier", source_group.get("tier") or row.get("assay_tier") or "")
    row_copy.setdefault("original_endpoint_group", source_group.get("endpoint_group") or row.get("endpoint_group") or "")
    row_copy["specific_group_id"] = assignment.group_id
    row_copy["specific_evidence_role"] = assignment.evidence_role
    row_copy["specific_retrieval_version"] = SPECIFIC_RETRIEVAL_VERSION
    return row_copy


def _empty_neighbor_for_specific_group(neighbor: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": 0,
        "molecule_chembl_id": neighbor.get("molecule_chembl_id", ""),
        "canonical_smiles": neighbor.get("canonical_smiles", ""),
        "standard_inchi_key": neighbor.get("standard_inchi_key", ""),
        "similarity": neighbor.get("similarity"),
        "similarity_bucket": neighbor.get("similarity_bucket"),
        "n_evidence_rows": 0,
        "evidence_rows": [],
        "source_group_ids": set(),
    }


def _rank_specific_neighbors(neighbors: list[dict[str, Any]], *, cap: int) -> list[dict[str, Any]]:
    ranked = sorted(
        neighbors,
        key=lambda neighbor: (
            -_float_or_zero(neighbor.get("similarity")),
            -_neighbor_evidence_priority(neighbor),
            str(neighbor.get("molecule_chembl_id") or neighbor.get("canonical_smiles") or ""),
        ),
    )
    output = []
    for rank, neighbor in enumerate(ranked[:cap], start=1):
        item = dict(neighbor)
        item["rank"] = rank
        item["source_group_ids"] = sorted(item.get("source_group_ids") or [])
        all_rows = list(item.get("evidence_rows") or [])
        item["n_evidence_rows_before_specific_cap"] = len(all_rows)
        item["evidence_rows"] = _rank_rows_for_neighbor(all_rows)[:SPECIFIC_ROWS_PER_NEIGHBOR_CAP]
        item["n_evidence_rows"] = len(item.get("evidence_rows") or [])
        output.append(item)
    return output


def _rank_rows_for_neighbor(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -_row_priority(row),
            str(row.get("original_group_id") or row.get("group_id") or ""),
            str(row.get("standard_type") or ""),
            str(row.get("assay_chembl_id") or row.get("source_group_id") or ""),
        ),
    )


def _row_priority(row: dict[str, Any]) -> int:
    role = str(row.get("specific_evidence_role") or "")
    if role == "direct_absolute_oral_f_candidate":
        priority = 50
    elif role == "oral_exposure_proxy_not_direct_f":
        priority = 30
    else:
        priority = 10
    if row.get("source_record_examples"):
        priority += 5
    if str(row.get("evidence_source") or "").startswith("starling"):
        priority += 3
    if row.get("standard_value") not in (None, "") or row.get("source_value_median_percent") not in (None, ""):
        priority += 2
    return priority


def _neighbor_evidence_priority(neighbor: dict[str, Any]) -> int:
    priority = 0
    for row in neighbor.get("evidence_rows") or []:
        role = str(row.get("specific_evidence_role") or "")
        if role == "direct_absolute_oral_f_candidate":
            priority += 5
        elif role == "oral_exposure_proxy_not_direct_f":
            priority += 2
        else:
            priority += 1
        if str(row.get("evidence_source") or "").startswith("starling"):
            priority += 1
    return priority


def _retrieve_starling_if_needed(
    *,
    query_smiles: str,
    base_index: dict[str, Any],
    starling_index_path: str,
    top_k_per_group: int,
    min_similarity: float,
    requested_groups: list[str] | None,
) -> dict[str, Any] | None:
    if not _should_include_starling(base_index, requested_groups):
        return None
    path = Path(starling_index_path)
    if not path.exists():
        return None
    starling_index = load_index(path)
    result = retrieve_source_neighbors(
        query_smiles,
        starling_index,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        groups=None,
    )
    return result if result.get("status") == "ok" else None


def _should_include_starling(base_index: dict[str, Any], requested_groups: list[str] | None) -> bool:
    base_groups = set((base_index.get("group_to_molecule_indices") or {}).keys())
    if any(group.startswith("Starling.") for group in base_groups):
        return False
    if "Combined.chembl_tier1_and_starling" in base_groups:
        return False
    if not requested_groups:
        return True
    requested = set(requested_groups)
    if requested & {OBSERVED_F_GROUP_ID, "Observed", "Starling.direct_oral_bioavailability"}:
        return True
    if any(group.startswith("Starling.") for group in requested):
        return True
    return not _specific_group_filters(requested_groups) and "Starling.direct_oral_bioavailability" in requested


def _source_group_filters(groups: list[str] | None) -> list[str] | None:
    if not groups:
        return None
    source_groups = [
        group
        for group in groups
        if group not in _SPECIFIC_GROUP_FILTERS and not group.startswith(("Observed.", "Fa.", "Fg.", "Fh."))
    ]
    return source_groups or None


def _specific_group_filters(groups: list[str] | None) -> set[str]:
    if not groups:
        return set()
    return {
        group
        for group in groups
        if group in _SPECIFIC_GROUP_FILTERS or group.startswith(("Observed.", "Fa.", "Fg.", "Fh."))
    }


def _specific_evidence_source(source_retrievals: list[dict[str, Any]]) -> dict[str, Any]:
    sources = []
    for retrieval in source_retrievals:
        source = retrieval.get("evidence_source") or {}
        if source and source not in sources:
            sources.append(source)
    return {
        "type": "specific_regrouped_source_retrievals",
        "specific_retrieval_version": SPECIFIC_RETRIEVAL_VERSION,
        "source_count": len(source_retrievals),
        "sources": sources,
    }


def _row_text(source_group_id: str, row: dict[str, Any]) -> str:
    parts = [
        source_group_id,
        row.get("group_id"),
        row.get("endpoint_group"),
        row.get("standard_type"),
        row.get("standard_units"),
        row.get("assay_description"),
        row.get("target_pref_name"),
        row.get("target_genes"),
        row.get("activity_comment"),
        row.get("source_report_types"),
        row.get("source_support_texts"),
    ]
    for example in row.get("source_record_examples") or []:
        parts.extend(
            [
                example.get("bioavailability_report_type"),
                example.get("condition_text"),
                example.get("oral_exposure_mode"),
                example.get("qualifying_conditions"),
                example.get("support_text"),
            ]
        )
    for example in row.get("source_qualitative_examples") or []:
        parts.extend(
            [
                example.get("bioavailability_report_type"),
                example.get("condition_text"),
                example.get("oral_exposure_mode"),
                example.get("qualifying_conditions"),
                example.get("support_text"),
            ]
        )
    return " ".join(str(part or "") for part in parts).lower()


def _neighbor_key(neighbor: dict[str, Any]) -> str:
    for key in ("standard_inchi_key", "molecule_chembl_id", "canonical_smiles"):
        value = str(neighbor.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return f"neighbor:{id(neighbor)}"


def _dedupe_assignments(assignments: list[_Assignment]) -> list[_Assignment]:
    seen = set()
    output = []
    for assignment in assignments:
        key = (assignment.group_id, assignment.evidence_role)
        if key in seen:
            continue
        seen.add(key)
        output.append(assignment)
    return output


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
