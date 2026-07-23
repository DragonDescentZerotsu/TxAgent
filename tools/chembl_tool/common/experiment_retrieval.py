"""Reusable retrieval views for molecular-agent ablation experiments.

The source index keeps fine-grained endpoint groups.  This module projects
those groups into task-declared direct or mechanism families without changing
the underlying evidence rows.  A flat view is derived from the selected
mechanism view, so flat and mechanism conditions receive the same evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rdkit import DataStructs

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import (
    NeighborIdentityPolicy,
    decide_candidate,
    policy_metadata,
)
from tools.chembl_tool.common.retrieval_reranker import RetrievalReranker
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import (
    retrieve_neighbors,
    similarity_bucket,
)


EXPERIMENT_MODES = {"none", "direct", "full_flat", "full_mechanism", "native"}


@dataclass(frozen=True)
class EvidenceGroupSpec:
    """Map one paper-facing evidence family to source-index groups."""

    group_id: str
    tier: str
    endpoint_group: str
    source_groups: tuple[str, ...] = ()
    source_group_prefixes: tuple[str, ...] = ()
    exclude_source_groups: tuple[str, ...] = ()

    def resolve(self, available_groups: set[str]) -> tuple[str, ...]:
        selected = {group for group in self.source_groups if group in available_groups}
        selected.update(
            group
            for group in available_groups
            if any(group.startswith(prefix) for prefix in self.source_group_prefixes)
        )
        selected.difference_update(self.exclude_source_groups)
        return tuple(sorted(selected))


@dataclass(frozen=True)
class SourceExperimentConfig:
    """Direct and mechanism views for one task/source pair."""

    source_name: str
    direct_groups: tuple[EvidenceGroupSpec, ...]
    mechanism_groups: tuple[EvidenceGroupSpec, ...]


def retrieve_experiment_view(
    query_smiles: str,
    index: Mapping[str, Any] | None,
    *,
    mode: str,
    config: SourceExperimentConfig | None,
    top_k_per_group: int,
    min_similarity: float,
    native_groups: list[str] | None = None,
    neighbor_identity_policy: str = NeighborIdentityPolicy.OPERATIONAL.value,
    reranker: RetrievalReranker | None = None,
    rerank_raw_pool_size: int = 100,
    rerank_candidate_size: int = 100,
    assay_transfer_min_score: float | None = None,
) -> dict[str, Any]:
    """Build a native, direct, flat, mechanism, or retrieval-free query view."""
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment mode: {mode}")
    if reranker is not None and not (
        0 < top_k_per_group <= rerank_candidate_size <= rerank_raw_pool_size
    ):
        raise ValueError(
            "Reranked retrieval requires 0 < top_k_per_group <= "
            "rerank_candidate_size <= rerank_raw_pool_size"
        )
    if assay_transfer_min_score is not None and not 0.0 <= assay_transfer_min_score <= 1.0:
        raise ValueError("assay_transfer_min_score must be between 0 and 1 inclusive")
    if assay_transfer_min_score is not None and reranker is None:
        raise ValueError("assay_transfer_min_score requires a retrieval reranker")
    if mode == "none":
        return _query_only_retrieval(query_smiles, mode=mode)
    if index is None:
        raise ValueError(f"Experiment mode `{mode}` requires a neighbor index")
    if mode == "native":
        result = retrieve_neighbors(
            query_smiles,
            dict(index),
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            groups=native_groups,
            neighbor_identity_policy=neighbor_identity_policy,
        )
        result["experiment"] = {
            "mode": mode,
            "source": _source_name(config, index),
            **policy_metadata(neighbor_identity_policy),
        }
        return result
    if config is None:
        raise ValueError(f"Experiment mode `{mode}` requires a source experiment config")

    specs = config.direct_groups if mode == "direct" else config.mechanism_groups
    mechanism_view = _retrieve_specs(
        query_smiles,
        index,
        specs=specs,
        source_name=config.source_name,
        mode=mode,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        reranker=reranker,
        rerank_raw_pool_size=rerank_raw_pool_size,
        rerank_candidate_size=rerank_candidate_size,
        assay_transfer_min_score=assay_transfer_min_score,
    )
    if mode == "full_flat" and mechanism_view.get("status") == "ok":
        mechanism_view["groups"] = [_flatten_groups(mechanism_view["groups"])]
        mechanism_view["coverage"] = _coverage(
            mechanism_view["groups"],
            min_similarity=min_similarity,
            top_k_per_group=top_k_per_group,
        )
    return mechanism_view


def _retrieve_specs(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    specs: tuple[EvidenceGroupSpec, ...],
    source_name: str,
    mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str,
    reranker: RetrievalReranker | None,
    rerank_raw_pool_size: int,
    rerank_candidate_size: int,
    assay_transfer_min_score: float | None,
) -> dict[str, Any]:
    canonical_smiles, inchi_key, query_fp = standardize_smiles_and_fp(query_smiles)
    if query_fp is None:
        return _invalid_query(query_smiles)

    query_identity = normalize_molecule_identity(query_smiles)
    similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, index["fingerprints"]))
    available_groups = set(index["group_to_molecule_indices"])
    output_groups = []
    resolved_mapping: dict[str, list[str]] = {}
    for spec in specs:
        source_groups = spec.resolve(available_groups)
        resolved_mapping[spec.group_id] = list(source_groups)
        candidate_indices = sorted(
            {
                molecule_index
                for source_group in source_groups
                for molecule_index in index["group_to_molecule_indices"].get(source_group, [])
            }
        )
        neighbors = _rank_group_candidates(
            index,
            candidate_indices,
            source_groups=source_groups,
            similarities=similarities,
            query_canonical_smiles=canonical_smiles,
            query_inchi_key=inchi_key,
            top_k=top_k_per_group,
            min_similarity=min_similarity,
            query_identity=query_identity,
            neighbor_identity_policy=neighbor_identity_policy,
            query_smiles=query_smiles,
            group_id=spec.group_id,
            reranker=reranker,
            rerank_raw_pool_size=rerank_raw_pool_size,
            rerank_candidate_size=rerank_candidate_size,
            assay_transfer_min_score=assay_transfer_min_score,
        )
        selection_metadata = (
            dict(neighbors.selection_metadata)
            if isinstance(neighbors, _RankedNeighbors)
            else {}
        )
        neighbors = list(neighbors)
        group_payload = {
            "group_id": spec.group_id,
            "tier": spec.tier,
            "endpoint_group": spec.endpoint_group,
            "source_group_ids": list(source_groups),
            "n_candidate_molecules": len(candidate_indices),
            "neighbors": neighbors,
        }
        if reranker is not None:
            group_payload["transfer_neighbor_selection"] = _rerank_group_metadata(
                reranker,
                raw_pool_size=rerank_raw_pool_size,
                candidate_size=rerank_candidate_size,
                n_selected=len(neighbors),
                min_score=assay_transfer_min_score,
                n_below_min_score_dropped=int(
                    selection_metadata.get("n_below_min_score_dropped", 0)
                ),
            )
        output_groups.append(group_payload)

    experiment = {
        "mode": mode,
        "source": source_name,
        "resolved_group_mapping": resolved_mapping,
        **policy_metadata(neighbor_identity_policy),
    }
    if reranker is not None:
        experiment["retrieval_reranker"] = reranker.provenance()
        experiment["assay_transfer_selection_policy"] = {
            "min_score": assay_transfer_min_score,
            "threshold_inclusive": True,
            "threshold_applied_before_top_k": assay_transfer_min_score is not None,
        }
    return {
        "status": "ok",
        "evidence_source": dict(index.get("source") or {}),
        "experiment": experiment,
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": canonical_smiles,
            "standard_inchi_key": inchi_key,
            "fingerprint": dict(index.get("fingerprint") or {}),
        },
        "groups": output_groups,
        "coverage": _coverage(
            output_groups,
            min_similarity=min_similarity,
            top_k_per_group=top_k_per_group,
        ),
    }


def _rank_group_candidates(
    index: Mapping[str, Any],
    candidate_indices: list[int],
    *,
    source_groups: tuple[str, ...],
    similarities: list[float],
    query_canonical_smiles: str,
    query_inchi_key: str,
    top_k: int,
    min_similarity: float,
    query_identity: Any,
    neighbor_identity_policy: str,
    query_smiles: str = "",
    group_id: str = "",
    reranker: RetrievalReranker | None = None,
    rerank_raw_pool_size: int = 100,
    rerank_candidate_size: int = 100,
    assay_transfer_min_score: float | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            (float(similarities[molecule_index]), molecule_index)
            for molecule_index in candidate_indices
            if similarities[molecule_index] >= min_similarity
        ),
        key=lambda item: (-item[0], index["molecules"][item[1]]["molecule_chembl_id"]),
    )
    raw_ranked = ranked[:rerank_raw_pool_size] if reranker is not None else ranked
    neighbors = []
    for structural_rank, (similarity, molecule_index) in enumerate(raw_ranked, start=1):
        molecule = index["molecules"][molecule_index]
        decision = decide_candidate(query_identity, molecule, neighbor_identity_policy)
        if decision.excluded:
            continue
        molecule_id = molecule["molecule_chembl_id"]
        evidence_by_group = index["evidence_by_molecule_group"].get(molecule_id, {})
        matched_groups = [group for group in source_groups if evidence_by_group.get(group)]
        evidence_rows = [row for group in matched_groups for row in evidence_by_group[group]]
        if not evidence_rows:
            continue
        neighbor = {
            "rank": len(neighbors) + 1,
            "molecule_chembl_id": molecule_id,
            "canonical_smiles": molecule["canonical_smiles"],
            "standard_inchi_key": molecule.get("standard_inchi_key", ""),
            "similarity": round(similarity, 6),
            "similarity_bucket": similarity_bucket(similarity),
            "molecule_relation": decision.relation.value,
            "source_group_ids": matched_groups,
            "n_evidence_rows": len(evidence_rows),
            "evidence_rows": evidence_rows,
        }
        if reranker is not None:
            neighbor["structural_rank"] = structural_rank
        neighbors.append(neighbor)
        limit = rerank_candidate_size if reranker is not None else top_k
        if len(neighbors) >= limit:
            break
    if reranker is not None:
        # Record-level top-K: rank every scored record across the candidate molecules and
        # keep the K highest-transfer records (the same molecule may repeat).
        rerank_records = getattr(reranker, "rerank_records", None) or reranker.rerank
        record_neighbors = rerank_records(
            query_smiles=query_smiles,
            group_id=group_id,
            candidates=neighbors,
        )
        n_below_min_score_dropped = 0
        if assay_transfer_min_score is not None:
            retained_records = [
                row
                for row in record_neighbors
                if float(row["transfer_selection_score"]) >= assay_transfer_min_score
            ]
            n_below_min_score_dropped = len(record_neighbors) - len(retained_records)
            record_neighbors = retained_records
        neighbors = _RankedNeighbors(
            record_neighbors[:top_k],
            selection_metadata={
                "assay_transfer_min_score": assay_transfer_min_score,
                "n_below_min_score_dropped": n_below_min_score_dropped,
            },
        )
    for rank, neighbor in enumerate(neighbors, start=1):
        neighbor["rank"] = rank
    return neighbors


def _rerank_group_metadata(
    reranker: RetrievalReranker | None,
    *,
    raw_pool_size: int,
    candidate_size: int,
    n_selected: int,
    min_score: float | None,
    n_below_min_score_dropped: int,
) -> dict[str, Any]:
    if reranker is None:
        return {}
    return {
        "reranker": reranker.name,
        "candidate_contract": "tanimoto_raw_pool_then_identity_exclusion.v1",
        "raw_pool_size": raw_pool_size,
        "candidate_size": candidate_size,
        "n_selected": n_selected,
        "assay_transfer_min_score": min_score,
        "n_below_min_score_dropped": n_below_min_score_dropped,
        "selection_metadata_is_llm_hidden": True,
    }


class _RankedNeighbors(list[dict[str, Any]]):
    """List-compatible retrieval result carrying group-level audit metadata."""

    def __init__(self, values: list[dict[str, Any]], *, selection_metadata: dict[str, Any]):
        super().__init__(values)
        self.selection_metadata = selection_metadata


def _flatten_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    source_group_ids: set[str] = set()
    for group in groups:
        source_group_ids.update(group.get("source_group_ids") or [])
        for neighbor in group.get("neighbors") or []:
            molecule_id = str(neighbor.get("molecule_chembl_id") or "")
            target = merged.get(molecule_id)
            if target is None:
                target = {**neighbor, "evidence_rows": [], "source_group_ids": []}
                merged[molecule_id] = target
            seen_rows = {id(row) for row in target["evidence_rows"]}
            target["evidence_rows"].extend(
                row for row in neighbor.get("evidence_rows") or [] if id(row) not in seen_rows
            )
            target["source_group_ids"] = sorted(
                set(target["source_group_ids"]) | set(neighbor.get("source_group_ids") or [])
            )
            target["n_evidence_rows"] = len(target["evidence_rows"])
    neighbors = sorted(
        merged.values(),
        key=lambda item: (-float(item.get("similarity") or 0.0), str(item.get("molecule_chembl_id") or "")),
    )
    for rank, neighbor in enumerate(neighbors, start=1):
        neighbor["rank"] = rank
    return {
        "group_id": "Flat.all_evidence",
        "tier": "Flat",
        "endpoint_group": "all_evidence",
        "source_group_ids": sorted(source_group_ids),
        "n_candidate_molecules": len(neighbors),
        "neighbors": neighbors,
    }


def _query_only_retrieval(query_smiles: str, *, mode: str) -> dict[str, Any]:
    canonical_smiles, inchi_key, query_fp = standardize_smiles_and_fp(query_smiles)
    if query_fp is None:
        return _invalid_query(query_smiles)
    return {
        "status": "ok",
        "evidence_source": {"type": "none"},
        "experiment": {"mode": mode, "source": "none", "resolved_group_mapping": {}},
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": canonical_smiles,
            "standard_inchi_key": inchi_key,
            "fingerprint": {},
        },
        "groups": [],
        "coverage": _coverage([], min_similarity=0.0, top_k_per_group=0),
    }


def _invalid_query(query_smiles: str) -> dict[str, Any]:
    return {
        "query": {"input_smiles": query_smiles, "canonical_smiles": "", "standard_inchi_key": ""},
        "status": "error",
        "errors": [{"code": "INVALID_SMILES", "message": "Could not parse query SMILES.", "recoverable": True}],
        "groups": [],
        "coverage": _coverage([], min_similarity=0.0, top_k_per_group=0),
    }


def _coverage(groups: list[dict[str, Any]], *, min_similarity: float, top_k_per_group: int) -> dict[str, Any]:
    coverage = {
        "n_groups": len(groups),
        "n_groups_with_neighbors": sum(bool(group.get("neighbors")) for group in groups),
        "n_neighbors_total": sum(len(group.get("neighbors") or []) for group in groups),
        "min_similarity": min_similarity,
        "top_k_per_group": top_k_per_group,
    }
    selections = [
        group["transfer_neighbor_selection"]
        for group in groups
        if group.get("transfer_neighbor_selection") is not None
    ]
    if selections:
        coverage["assay_transfer_min_score"] = selections[0].get("assay_transfer_min_score")
        coverage["n_below_assay_transfer_min_score_dropped"] = sum(
            int(selection.get("n_below_min_score_dropped") or 0)
            for selection in selections
        )
    return coverage


def _source_name(config: SourceExperimentConfig | None, index: Mapping[str, Any]) -> str:
    if config is not None:
        return config.source_name
    source = index.get("source") or {}
    return str(source.get("dataset") or source.get("type") or "native")
