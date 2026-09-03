"""Cumulative family retrieval used only by the progressive harness."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from predict.llm_io.evidence import ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE
from predict.retrieval.retrieve import retrieve_neighbors


FAMILY_MOLECULE_VIEW_VERSION = "starling_family_molecule_prefix_view.v1"
FAMILY_MOLECULE_GROUP_PREFIX = "Flat.progressive_family_level_"
FLAT_GROUP_ID = "Flat.assay_ranked_evidence"


def build_family_molecule_prefix_view(
    index: dict[str, Any],
    *,
    levels: Iterable[int],
) -> dict[str, Any]:
    """Build cumulative record-family pools without per-assay neighbor caps."""
    normalized = sorted({int(level) for level in levels})
    if not normalized or normalized != list(range(1, normalized[-1] + 1)):
        raise ValueError("levels must be contiguous positive integers starting at 1")
    profile = str((index.get("source") or {}).get("evidence_prompt_profile") or "")
    if profile != ASSAY_MECHANISM_TAGGED_PROMPT_PROFILE:
        raise ValueError(
            "family-molecule prefix retrieval requires a mechanism-tagged index"
        )

    molecules = list(index.get("molecules") or [])
    molecule_index = {
        str(row.get("molecule_chembl_id") or ""): position
        for position, row in enumerate(molecules)
    }
    virtual_groups = {
        level: f"{FAMILY_MOLECULE_GROUP_PREFIX}{level}" for level in normalized
    }
    group_to_molecule_indices = {
        group_id: [] for group_id in virtual_groups.values()
    }
    evidence_by_molecule_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    missing_family_levels = 0
    visible_examples_by_level: Counter[int] = Counter()
    source_assays_by_level = {level: set() for level in normalized}

    for raw_molecule_id, assay_groups in (
        index.get("evidence_by_molecule_group") or {}
    ).items():
        molecule_id = str(raw_molecule_id)
        position = molecule_index.get(molecule_id)
        if position is None:
            raise ValueError(f"evidence references unknown molecule: {molecule_id}")
        cumulative_rows = {level: [] for level in normalized}
        for rows in assay_groups.values():
            for row in rows:
                tagged = []
                for raw_example in row.get("source_record_examples") or []:
                    if not isinstance(raw_example, dict):
                        continue
                    example = dict(raw_example)
                    try:
                        family_level = int(example.get("evidence_family_level") or 0)
                    except (TypeError, ValueError):
                        family_level = 0
                    if family_level <= 0:
                        missing_family_levels += 1
                        continue
                    tagged.append((family_level, example))
                if not tagged:
                    continue
                assay_key = str(
                    row.get("assay_chembl_id")
                    or (row.get("assay_retrieval") or {}).get("assay_id")
                    or row.get("group_id")
                    or ""
                )
                for level in normalized:
                    visible = [
                        example
                        for family_level, example in tagged
                        if family_level <= level
                    ]
                    if not visible:
                        continue
                    retained = dict(row)
                    retained["source_record_examples"] = visible
                    cumulative_rows[level].append(retained)
                    visible_examples_by_level[level] += len(visible)
                    if assay_key:
                        source_assays_by_level[level].add(assay_key)

        molecule_groups = {}
        for level, rows in cumulative_rows.items():
            if not rows:
                continue
            group_id = virtual_groups[level]
            group_to_molecule_indices[group_id].append(position)
            molecule_groups[group_id] = rows
        if molecule_groups:
            evidence_by_molecule_group[molecule_id] = molecule_groups

    if missing_family_levels:
        raise ValueError(
            "mechanism-tagged index contains representative records without a "
            f"family level: {missing_family_levels}"
        )

    view = dict(index)
    view["group_to_molecule_indices"] = group_to_molecule_indices
    view["evidence_by_molecule_group"] = evidence_by_molecule_group
    view["family_molecule_prefix_view"] = {
        "version": FAMILY_MOLECULE_VIEW_VERSION,
        "levels": normalized,
        "group_ids": {
            str(level): virtual_groups[level] for level in normalized
        },
        "candidate_generation": (
            "global_molecule_similarity_within_cumulative_record_families"
        ),
        "per_assay_neighbor_cap": None,
        "n_candidate_molecules_by_level": {
            str(level): len(group_to_molecule_indices[virtual_groups[level]])
            for level in normalized
        },
        "n_source_assays_by_level": {
            str(level): len(source_assays_by_level[level]) for level in normalized
        },
        "n_visible_representative_records_by_level": {
            str(level): int(visible_examples_by_level[level])
            for level in normalized
        },
    }
    view["source"] = {
        **dict(index.get("source") or {}),
        "family_molecule_prefix_view": FAMILY_MOLECULE_VIEW_VERSION,
    }
    return view


def retrieve_family_molecule_prefixes(
    query_smiles: str,
    index: dict[str, Any],
    *,
    levels: Iterable[int],
    min_similarity: float = 0.3,
    neighbor_identity_policy: str = "parent_disjoint",
) -> dict[int, dict[str, Any]]:
    """Retrieve every eligible molecule globally within each family prefix."""
    normalized = sorted({int(level) for level in levels})
    metadata = dict(index.get("family_molecule_prefix_view") or {})
    if metadata.get("version") != FAMILY_MOLECULE_VIEW_VERSION:
        raise ValueError("index is not a family-molecule prefix view")
    if normalized != [int(level) for level in metadata.get("levels") or []]:
        raise ValueError("requested levels do not match the prepared prefix view")
    group_ids = [str(metadata["group_ids"][str(level)]) for level in normalized]
    native = retrieve_neighbors(
        query_smiles,
        index,
        top_k_per_group=max(1, len(index.get("molecules") or [])),
        min_similarity=min_similarity,
        groups=group_ids,
        neighbor_identity_policy=neighbor_identity_policy,
    )
    if native.get("status") != "ok":
        return {level: native for level in normalized}

    results = {}
    for level, group in zip(normalized, native.get("groups") or [], strict=True):
        neighbors = list(group.get("neighbors") or [])
        flat = {
            **dict(group),
            "group_id": FLAT_GROUP_ID,
            "tier": "Flat",
            "endpoint_group": "progressive_family_evidence",
            "evidence_prompt_profile": str(
                (index.get("source") or {}).get("evidence_prompt_profile") or ""
            ),
            "n_unique_neighbor_molecules": len(neighbors),
            "n_candidate_neighbor_molecules_before_budget": len(neighbors),
            "n_source_assays": int(
                metadata["n_source_assays_by_level"][str(level)]
            ),
            "n_visible_representative_records": int(
                metadata["n_visible_representative_records_by_level"][str(level)]
            ),
        }
        results[level] = {
            "status": "ok",
            "evidence_source": dict(index.get("source") or {}),
            "retrieval_policy": dict(native.get("retrieval_policy") or {}),
            "experiment": {
                "mode": "progressive_family_molecule_flat",
                "source": "starling_mechanism_tagged_family_molecule_view",
                "family_level": level,
                "candidate_generation": metadata["candidate_generation"],
                "per_assay_neighbor_cap": None,
                "min_similarity": min_similarity,
                "neighbor_identity_policy": neighbor_identity_policy,
                "relevance_scores_visible_to_llm": False,
            },
            "query": dict(native.get("query") or {}),
            "groups": [flat] if neighbors else [],
            "coverage": {
                "n_groups": 1,
                "n_groups_with_neighbors": int(bool(neighbors)),
                "n_neighbors_total": len(neighbors),
                "n_candidate_neighbor_molecules_before_budget": len(neighbors),
                "n_source_assays": flat["n_source_assays"],
                "n_visible_representative_records": flat[
                    "n_visible_representative_records"
                ],
                "min_similarity": min_similarity,
                "candidate_generation": metadata["candidate_generation"],
                "per_assay_neighbor_cap": None,
            },
        }
    return results
