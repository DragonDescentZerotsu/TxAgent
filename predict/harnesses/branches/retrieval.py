"""Shared neighbor selection and branch retrieval implementation.

The low-level functions in this module select and rank task-declared branch
definitions without changing the underlying evidence rows. Public harnesses
own experiment-specific organization: ``predict.harnesses.branches.direct`` chooses
direct groups, ``full`` preserves mechanism families, and ``flat`` collapses
those same mechanism families. ``retrieve_experiment_view`` retains the old
mode-based API by dispatching to those harness-owned functions.

Inputs are a standardized query SMILES, an in-memory molecular evidence index,
task-declared group specifications, and explicit ranking/policy options. The
module fingerprints the query, scores eligible candidate molecules, applies
identity exclusions and optional assay-transfer reranking, expands selected
molecules back into evidence records, and returns the retrieval dictionary
consumed by the stage runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from predict.harnesses.branches.assay_transfer import (
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE,
    assay_transfer_selection_policy,
    collapse_assay_transfer_records_by_mean_molecule,
    collapse_assay_transfer_records_by_molecule,
    select_assay_transfer_records,
    validate_assay_transfer_diversity,
    validate_assay_transfer_records_per_molecule,
)
from predict.retrieval.policies import normalize_molecule_identity
from predict.retrieval.policies import (
    SIMILARITY_SELECTOR,
    NeighborCandidate,
    select_neighbor_candidates,
    selector_metadata,
)
from predict.retrieval.policies import (
    NeighborIdentityPolicy,
    decide_candidate,
    policy_metadata,
)
from predict.harnesses.branches.reranker import RetrievalReranker
from predict.retrieval.features import (
    retrieval_feature_metadata,
    similarity_bucket_for_index,
    similarity_vector,
)
from predict.retrieval.policies import standardize_smiles_and_fp
from predict.retrieval.retrieve import (
    retrieve_neighbors,
)


EXPERIMENT_MODES = {"none", "direct", "full_flat", "full_mechanism", "native"}

# Retrieval strategy is the CLI-level source of truth. Internally the strategy is
# implied by whether a reranker is supplied: morgan_fingerprint -> no reranker
# (ranked by neighbor_selector); assay_transfer_tool -> assay-transfer reranker.
MORGAN_FINGERPRINT_STRATEGY = "morgan_fingerprint"
ASSAY_TRANSFER_TOOL_STRATEGY = "assay_transfer_tool"
RETRIEVAL_STRATEGIES = (MORGAN_FINGERPRINT_STRATEGY, ASSAY_TRANSFER_TOOL_STRATEGY)


@dataclass(frozen=True)
class BranchDefinition:
    """Define one reasoning branch by mapping it to source-index groups."""

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
class BranchRetrievalConfig:
    """Direct and mechanism branch definitions for one task/source pair."""

    source_name: str
    direct_groups: tuple[BranchDefinition, ...]
    mechanism_groups: tuple[BranchDefinition, ...]


def retrieve_branches(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    branch_definitions: tuple[BranchDefinition, ...],
    source_name: str,
    mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str = NeighborIdentityPolicy.OPERATIONAL.value,
    neighbor_selector: str = SIMILARITY_SELECTOR,
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Retrieve one reasoning branch per task-declared branch definition."""
    return _retrieve_branches(
        query_smiles,
        index,
        branch_definitions=branch_definitions,
        source_name=source_name,
        mode=mode,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        neighbor_selector=neighbor_selector,
        reranker=reranker,
        assay_transfer_initial_morgan_filter=assay_transfer_initial_morgan_filter,
        assay_transfer_min_score=assay_transfer_min_score,
        assay_transfer_diversity_mode=assay_transfer_diversity_mode,
        assay_transfer_diversity_score_slack=assay_transfer_diversity_score_slack,
        assay_transfer_selection_unit=assay_transfer_selection_unit,
        assay_transfer_records_per_molecule=assay_transfer_records_per_molecule,
    )


def retrieve_mechanism_evidence(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    config: BranchRetrievalConfig,
    mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str = NeighborIdentityPolicy.OPERATIONAL.value,
    neighbor_selector: str = SIMILARITY_SELECTOR,
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Select and rank mechanism families shared by flat and full inference."""
    return retrieve_branches(
        query_smiles,
        index,
        branch_definitions=config.mechanism_groups,
        source_name=config.source_name,
        mode=mode,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        neighbor_selector=neighbor_selector,
        reranker=reranker,
        assay_transfer_initial_morgan_filter=assay_transfer_initial_morgan_filter,
        assay_transfer_min_score=assay_transfer_min_score,
        assay_transfer_diversity_mode=assay_transfer_diversity_mode,
        assay_transfer_diversity_score_slack=assay_transfer_diversity_score_slack,
        assay_transfer_selection_unit=assay_transfer_selection_unit,
        assay_transfer_records_per_molecule=assay_transfer_records_per_molecule,
    )


def retrieve_experiment_view(
    query_smiles: str,
    index: Mapping[str, Any] | None,
    *,
    mode: str,
    config: BranchRetrievalConfig | None,
    top_k_per_group: int,
    min_similarity: float,
    native_groups: list[str] | None = None,
    neighbor_identity_policy: str = NeighborIdentityPolicy.OPERATIONAL.value,
    neighbor_selector: str = SIMILARITY_SELECTOR,
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Build a native, direct, flat, mechanism, or retrieval-free query view.

    Retrieval strategy is implied by ``reranker``: when it is ``None`` the
    morgan-fingerprint path runs (ranked by ``neighbor_selector``); when a
    reranker is supplied the assay-transfer path runs (initial morgan filter ->
    candidate-validation policies -> record-level rerank -> top-k).
    """
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"Unsupported experiment mode: {mode}")
    if reranker is not None and not (0 < top_k_per_group <= assay_transfer_initial_morgan_filter):
        raise ValueError(
            "Assay-transfer retrieval requires "
            "0 < top_k_per_group <= assay_transfer_initial_morgan_filter"
        )
    if assay_transfer_min_score is not None and not 0.0 <= assay_transfer_min_score <= 1.0:
        raise ValueError("assay_transfer_min_score must be between 0 and 1 inclusive")
    if assay_transfer_min_score is not None and reranker is None:
        raise ValueError("assay_transfer_min_score requires a retrieval reranker")
    if (
        assay_transfer_selection_unit == ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE
        and assay_transfer_min_score is not None
    ):
        raise ValueError("mean_score_molecule requires no assay-transfer score floor")
    validate_assay_transfer_diversity(
        mode=assay_transfer_diversity_mode,
        score_slack=assay_transfer_diversity_score_slack,
    )
    validate_assay_transfer_records_per_molecule(
        assay_transfer_records_per_molecule,
        selection_unit=assay_transfer_selection_unit,
    )
    if reranker is None and assay_transfer_diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE:
        raise ValueError("assay-transfer diversity requires a retrieval reranker")
    if (
        assay_transfer_selection_unit == ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE
        and assay_transfer_diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE
    ):
        raise ValueError("mean_score_molecule does not support diversity selection")
    if (
        reranker is None
        and assay_transfer_records_per_molecule
        > ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT
    ):
        raise ValueError("multiple assay-transfer records per molecule require a retrieval reranker")
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
            neighbor_selector=neighbor_selector,
        )
        result["experiment"] = {
            "mode": mode,
            "source": _source_name(config, index),
            **policy_metadata(neighbor_identity_policy),
            "neighbor_selector": selector_metadata(neighbor_selector),
        }
        return result
    if config is None:
        raise ValueError(f"Experiment mode `{mode}` requires a source experiment config")

    if mode == "direct":
        from predict.harnesses.branches.direct import retrieve_direct_evidence

        retriever = retrieve_direct_evidence
    elif mode == "full_flat":
        from predict.harnesses.branches.flat import retrieve_flat_evidence

        retriever = retrieve_flat_evidence
    else:
        from predict.harnesses.branches.full import retrieve_full_evidence

        retriever = retrieve_full_evidence
    return retriever(
        query_smiles,
        index,
        config=config,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        neighbor_selector=neighbor_selector,
        reranker=reranker,
        assay_transfer_initial_morgan_filter=assay_transfer_initial_morgan_filter,
        assay_transfer_min_score=assay_transfer_min_score,
        assay_transfer_diversity_mode=assay_transfer_diversity_mode,
        assay_transfer_diversity_score_slack=assay_transfer_diversity_score_slack,
        assay_transfer_selection_unit=assay_transfer_selection_unit,
        assay_transfer_records_per_molecule=assay_transfer_records_per_molecule,
    )


def _retrieve_branches(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    branch_definitions: tuple[BranchDefinition, ...],
    source_name: str,
    mode: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str,
    neighbor_selector: str,
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Resolve source groups, rank each branch pool, and assemble retrieval."""
    canonical_smiles, inchi_key, query_fp = standardize_smiles_and_fp(query_smiles)
    if query_fp is None:
        return _invalid_query(query_smiles)

    query_identity = normalize_molecule_identity(query_smiles)
    similarities = similarity_vector(
        query_fp,
        canonical_smiles,
        index,
        neighbor_selector=neighbor_selector,
    )
    available_groups = set(index["group_to_molecule_indices"])
    output_groups = []
    resolved_mapping: dict[str, list[str]] = {}
    for spec in branch_definitions:
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
            query_fingerprint=query_fp,
            neighbor_selector=neighbor_selector,
            query_smiles=query_smiles,
            group_id=spec.group_id,
            reranker=reranker,
            assay_transfer_initial_morgan_filter=assay_transfer_initial_morgan_filter,
            assay_transfer_min_score=assay_transfer_min_score,
            assay_transfer_diversity_mode=assay_transfer_diversity_mode,
            assay_transfer_diversity_score_slack=assay_transfer_diversity_score_slack,
            assay_transfer_selection_unit=assay_transfer_selection_unit,
            assay_transfer_records_per_molecule=assay_transfer_records_per_molecule,
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
                initial_morgan_filter=assay_transfer_initial_morgan_filter,
                n_selected=len(neighbors),
                min_score=assay_transfer_min_score,
                n_below_min_score_dropped=int(
                    selection_metadata.get("n_below_min_score_dropped", 0)
                ),
                diversity=dict(selection_metadata.get("diversity") or {}),
                molecule_collapse=dict(
                    selection_metadata.get("molecule_collapse") or {}
                ),
                selected_record_display=dict(
                    selection_metadata.get("selected_record_display") or {}
                ),
            )
        output_groups.append(group_payload)

    experiment = {
        "mode": mode,
        "source": source_name,
        "resolved_group_mapping": resolved_mapping,
        "retrieval_feature": retrieval_feature_metadata(index),
        **policy_metadata(neighbor_identity_policy),
        "neighbor_selector": selector_metadata(neighbor_selector),
    }
    if reranker is not None:
        experiment["retrieval_reranker"] = reranker.provenance()
        experiment["assay_transfer_selection_policy"] = {
            "min_score": assay_transfer_min_score,
            "threshold_inclusive": True,
            "threshold_applied_before_top_k": assay_transfer_min_score is not None,
            "diversity": assay_transfer_selection_policy(
                mode=assay_transfer_diversity_mode,
                score_slack=assay_transfer_diversity_score_slack,
                selection_unit=assay_transfer_selection_unit,
                records_per_molecule=assay_transfer_records_per_molecule,
            ),
        }
    return {
        "status": "ok",
        "evidence_source": dict(index.get("source") or {}),
        "retrieval_policy": {
            **policy_metadata(neighbor_identity_policy),
            "neighbor_selector": selector_metadata(neighbor_selector),
        },
        "experiment": experiment,
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": canonical_smiles,
            "standard_inchi_key": inchi_key,
            "fingerprint": (
                dict(index.get("fingerprint") or {})
                if retrieval_feature_metadata(index)["feature"] == "morgan_fingerprint"
                else {}
            ),
            "retrieval_feature": retrieval_feature_metadata(index),
        },
        "groups": output_groups,
        "coverage": retrieval_coverage(
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
    neighbor_selector: str = SIMILARITY_SELECTOR,
    query_fingerprint: Any = None,
    query_smiles: str = "",
    group_id: str = "",
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> list[dict[str, Any]]:
    """Dispatch to the retrieval strategy implied by ``reranker``.

    ``reranker is None`` runs the morgan-fingerprint path (ranked by
    ``neighbor_selector``); otherwise the assay-transfer path runs (initial morgan
    filter -> candidate-validation policies -> record-level rerank -> min-score
    filter -> top-k).
    """
    if reranker is None:
        return _morgan_neighbors(
            index,
            candidate_indices,
            source_groups=source_groups,
            similarities=similarities,
            top_k=top_k,
            min_similarity=min_similarity,
            query_identity=query_identity,
            neighbor_identity_policy=neighbor_identity_policy,
            query_fingerprint=query_fingerprint,
            neighbor_selector=neighbor_selector,
        )
    return _assay_transfer_neighbors(
        index,
        candidate_indices,
        source_groups=source_groups,
        similarities=similarities,
        top_k=top_k,
        min_similarity=min_similarity,
        query_identity=query_identity,
        neighbor_identity_policy=neighbor_identity_policy,
        query_fingerprint=query_fingerprint,
        query_smiles=query_smiles,
        group_id=group_id,
        reranker=reranker,
        initial_morgan_filter=assay_transfer_initial_morgan_filter,
        assay_transfer_min_score=assay_transfer_min_score,
        diversity_mode=assay_transfer_diversity_mode,
        diversity_score_slack=assay_transfer_diversity_score_slack,
        selection_unit=assay_transfer_selection_unit,
        records_per_molecule=assay_transfer_records_per_molecule,
    )


def _morgan_neighbors(
    index: Mapping[str, Any],
    candidate_indices: list[int],
    *,
    source_groups: tuple[str, ...],
    similarities: list[float],
    top_k: int,
    min_similarity: float,
    query_identity: Any,
    neighbor_identity_policy: str,
    query_fingerprint: Any,
    neighbor_selector: str,
) -> list[dict[str, Any]]:
    """Apply identity and similarity gates, then select Morgan-ranked molecules."""
    eligible: list[NeighborCandidate] = []
    decisions: dict[int, Any] = {}
    matched_groups_by_index: dict[int, list[str]] = {}
    evidence_by_index: dict[int, list[dict[str, Any]]] = {}
    ordered_candidate_indices = candidate_indices
    stop_after_top_k = neighbor_selector == SIMILARITY_SELECTOR
    if stop_after_top_k:
        ordered_candidate_indices = sorted(
            candidate_indices,
            key=lambda molecule_index: (
                -float(similarities[molecule_index]),
                str(index["molecules"][molecule_index]["molecule_chembl_id"]),
                molecule_index,
            ),
        )
    for molecule_index in ordered_candidate_indices:
        similarity = float(similarities[molecule_index])
        if similarity < min_similarity:
            if stop_after_top_k:
                break
            continue
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
        eligible.append(
            NeighborCandidate(
                molecule_index=molecule_index,
                molecule_id=str(molecule_id),
                similarity=similarity,
            )
        )
        decisions[molecule_index] = decision
        matched_groups_by_index[molecule_index] = matched_groups
        evidence_by_index[molecule_index] = evidence_rows
        if stop_after_top_k and len(eligible) >= top_k:
            break

    selected = (
        eligible
        if stop_after_top_k
        else select_neighbor_candidates(
            eligible,
            query_fingerprint=query_fingerprint,
            candidate_fingerprints=index["fingerprints"],
            top_k=top_k,
            selector=neighbor_selector,
        )
    )
    neighbors = []
    for candidate in selected:
        molecule_index = candidate.molecule_index
        similarity = candidate.similarity
        molecule = index["molecules"][molecule_index]
        molecule_id = molecule["molecule_chembl_id"]
        decision = decisions[molecule_index]
        matched_groups = matched_groups_by_index[molecule_index]
        evidence_rows = evidence_by_index[molecule_index]
        neighbors.append(
            {
                "rank": len(neighbors) + 1,
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": molecule["canonical_smiles"],
                "standard_inchi_key": molecule.get("standard_inchi_key", ""),
                "similarity": round(similarity, 6),
                "similarity_bucket": similarity_bucket_for_index(similarity, index),
                "similarity_metric": retrieval_feature_metadata(index)["similarity"],
                "molecule_relation": decision.relation.value,
                "source_group_ids": matched_groups,
                "n_evidence_rows": len(evidence_rows),
                "evidence_rows": evidence_rows,
            }
        )
    return neighbors


def _assay_transfer_neighbors(
    index: Mapping[str, Any],
    candidate_indices: list[int],
    *,
    source_groups: tuple[str, ...],
    similarities: list[float],
    top_k: int,
    min_similarity: float,
    query_identity: Any,
    neighbor_identity_policy: str,
    query_fingerprint: Any,
    query_smiles: str,
    group_id: str,
    reranker: RetrievalReranker,
    initial_morgan_filter: int,
    assay_transfer_min_score: float | None,
    diversity_mode: str,
    diversity_score_slack: float,
    selection_unit: str,
    records_per_molecule: int,
) -> list[dict[str, Any]]:
    """Build the Morgan candidate pool, rerank records, and select final evidence."""
    # 1. Initial morgan pool: top-N candidates by tanimoto similarity.
    ranked = sorted(
        (
            (float(similarities[molecule_index]), molecule_index)
            for molecule_index in candidate_indices
            if similarities[molecule_index] >= min_similarity
        ),
        key=lambda item: (-item[0], index["molecules"][item[1]]["molecule_chembl_id"]),
    )
    candidate_contract = str(
        (reranker.provenance() or {}).get("candidate_contract") or ""
    )
    eligible_pool = (
        candidate_contract == "tanimoto_identity_exclusion_then_eligible_pool.v1"
    )
    raw_ranked = ranked if eligible_pool else ranked[:initial_morgan_filter]
    # 2. Candidate-validation policies (parent-disjoint / identity exclusion + evidence
    #    presence). Keep *all* validated candidates -- the whole pool is reranked.
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
        neighbors.append(
            {
                "rank": len(neighbors) + 1,
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": molecule["canonical_smiles"],
                "standard_inchi_key": molecule.get("standard_inchi_key", ""),
                "similarity": round(similarity, 6),
                "similarity_bucket": similarity_bucket_for_index(similarity, index),
                "similarity_metric": retrieval_feature_metadata(index)["similarity"],
                "molecule_relation": decision.relation.value,
                "source_group_ids": matched_groups,
                "n_evidence_rows": len(evidence_rows),
                "evidence_rows": evidence_rows,
                "structural_rank": structural_rank,
            }
        )
        if eligible_pool and len(neighbors) == initial_morgan_filter:
            break
    # 3. Record-level rerank across the full validated set.
    rerank_records = getattr(reranker, "rerank_records", None) or reranker.rerank
    record_neighbors = rerank_records(
        query_smiles=query_smiles,
        group_id=group_id,
        candidates=neighbors,
    )
    # 4. Apply the score floor, optionally group endpoint-distinct records under
    #    each molecule, then select top-k molecules/records under the chosen unit.
    n_below_min_score_dropped = 0
    if assay_transfer_min_score is not None:
        retained_records = [
            row
            for row in record_neighbors
            if float(row["transfer_selection_score"]) >= assay_transfer_min_score
        ]
        n_below_min_score_dropped = len(record_neighbors) - len(retained_records)
        record_neighbors = retained_records
    collapse_audit: dict[str, Any] = {
        "selection_unit": ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
        "n_valid_records_before_collapse": len(record_neighbors),
    }
    if selection_unit == ASSAY_TRANSFER_SELECTION_UNIQUE_MOLECULE:
        record_neighbors, collapse_audit = collapse_assay_transfer_records_by_molecule(
            record_neighbors,
            records_per_molecule=records_per_molecule,
        )
    elif selection_unit == ASSAY_TRANSFER_SELECTION_MEAN_SCORE_MOLECULE:
        record_neighbors, collapse_audit = collapse_assay_transfer_records_by_mean_molecule(
            record_neighbors,
            records_per_molecule=records_per_molecule,
        )
    if diversity_mode == ASSAY_TRANSFER_DIVERSITY_NONE or diversity_score_slack == 0.0:
        # Preserve the historical score-only path exactly, including unit-test
        # callers that deliberately supply a minimal index without fingerprints.
        selected_records = record_neighbors[:top_k]
        diversity_audit = {
            **assay_transfer_selection_policy(
                mode=diversity_mode,
                score_slack=diversity_score_slack,
                selection_unit=selection_unit,
                records_per_molecule=records_per_molecule,
            ),
            "n_selected": len(selected_records),
        }
    else:
        if query_fingerprint is None or "fingerprints" not in index:
            raise ValueError("Assay-transfer diversity requires Morgan fingerprints")
        fingerprints_by_molecule = {
            str(index["molecules"][molecule_index]["molecule_chembl_id"]): index["fingerprints"][molecule_index]
            for _, molecule_index in raw_ranked
        }
        selected_records, diversity_audit = select_assay_transfer_records(
            record_neighbors,
            top_k=top_k,
            mode=diversity_mode,
            score_slack=diversity_score_slack,
            query_fingerprint=query_fingerprint,
            fingerprints_by_molecule=fingerprints_by_molecule,
            selection_unit=selection_unit,
            records_per_molecule=records_per_molecule,
        )
    selected_record_display = {
        "records_per_molecule": records_per_molecule,
        "n_selected_molecules": len(selected_records),
        "n_selected_records_displayed": sum(
            int(record.get("transfer_selected_record_count") or 1)
            for record in selected_records
        ),
        "n_underfilled_selected_molecules": sum(
            bool(record.get("transfer_records_underfilled"))
            for record in selected_records
        ),
        "n_duplicate_endpoint_records_skipped": sum(
            int(record.get("transfer_duplicate_endpoint_records_skipped") or 0)
            for record in selected_records
        ),
        "duplicate_endpoint_backfill": False,
    }
    ranked_neighbors = _RankedNeighbors(
        selected_records,
        selection_metadata={
            "assay_transfer_min_score": assay_transfer_min_score,
            "n_below_min_score_dropped": n_below_min_score_dropped,
            "diversity": diversity_audit,
            "selection_unit": selection_unit,
            "molecule_collapse": collapse_audit,
            "selected_record_display": selected_record_display,
        },
    )
    for rank, neighbor in enumerate(ranked_neighbors, start=1):
        neighbor["rank"] = rank
    return ranked_neighbors


def _rerank_group_metadata(
    reranker: RetrievalReranker | None,
    *,
    initial_morgan_filter: int,
    n_selected: int,
    min_score: float | None,
    n_below_min_score_dropped: int,
    diversity: dict[str, Any] | None = None,
    molecule_collapse: dict[str, Any] | None = None,
    selected_record_display: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record candidate, eligibility, reranking, and final-selection counts."""
    if reranker is None:
        return {}
    return {
        "reranker": reranker.name,
        "candidate_contract": "tanimoto_raw_pool_then_identity_exclusion.v1",
        "assay_transfer_initial_morgan_filter": initial_morgan_filter,
        "n_selected": n_selected,
        "assay_transfer_min_score": min_score,
        "n_below_min_score_dropped": n_below_min_score_dropped,
        "diversity": diversity or {},
        "molecule_collapse": molecule_collapse or {},
        "selected_record_display": selected_record_display or {},
        "selection_metadata_is_llm_hidden": True,
    }


class _RankedNeighbors(list[dict[str, Any]]):
    """List-compatible retrieval result carrying group-level audit metadata."""

    def __init__(self, values: list[dict[str, Any]], *, selection_metadata: dict[str, Any]):
        super().__init__(values)
        self.selection_metadata = selection_metadata


def flatten_retrieval_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge family groups into one molecule-deduplicated evidence branch."""
    merged: dict[str, dict[str, Any]] = {}
    source_group_ids: set[str] = set()
    transfer_selections = [
        group.get("transfer_neighbor_selection")
        for group in groups
        if group.get("transfer_neighbor_selection") is not None
    ]
    for group in groups:
        source_group_ids.update(group.get("source_group_ids") or [])
        for neighbor in group.get("neighbors") or []:
            molecule_id = str(neighbor.get("molecule_chembl_id") or "")
            target = merged.get(molecule_id)
            if target is None:
                target = {
                    **neighbor,
                    "evidence_rows": [],
                    "source_group_ids": [],
                    "transfer_family_selections": [],
                }
                merged[molecule_id] = target
            seen_rows = {_evidence_row_key(row) for row in target["evidence_rows"]}
            target["evidence_rows"].extend(
                row
                for row in neighbor.get("evidence_rows") or []
                if _evidence_row_key(row) not in seen_rows
            )
            target["source_group_ids"] = sorted(
                set(target["source_group_ids"]) | set(neighbor.get("source_group_ids") or [])
            )
            target["n_evidence_rows"] = len(target["evidence_rows"])
            if group.get("transfer_neighbor_selection") is not None:
                selected_records = neighbor.get("transfer_selected_records") or [
                    {
                        "record_rank": 1,
                        "transfer_selection_score": neighbor.get(
                            "transfer_selection_score"
                        ),
                        "transfer_winning_record_id": neighbor.get(
                            "transfer_winning_record_id"
                        ),
                        "transfer_winning_record": neighbor.get(
                            "transfer_winning_record"
                        )
                        or {},
                    }
                ]
                existing_family = next(
                    (
                        family
                        for family in target["transfer_family_selections"]
                        if family.get("group_id") == group.get("group_id")
                    ),
                    None,
                )
                if existing_family is None:
                    target["transfer_family_selections"].append(
                        {
                            "group_id": group.get("group_id"),
                            "tier": group.get("tier"),
                            "endpoint_group": group.get("endpoint_group"),
                            "family_rank": neighbor.get("rank"),
                            "transfer_selection_score": neighbor.get(
                                "transfer_selection_score"
                            ),
                            "transfer_molecule_mean_score": neighbor.get(
                                "transfer_molecule_mean_score"
                            ),
                            "transfer_scored_record_count": neighbor.get(
                                "transfer_scored_record_count"
                            ),
                            "transfer_selected_records": list(selected_records),
                            "transfer_winning_record": neighbor.get(
                                "transfer_winning_record"
                            )
                            or {},
                        }
                    )
                else:
                    seen_record_ids = {
                        str(record.get("transfer_winning_record_id") or "")
                        for record in existing_family["transfer_selected_records"]
                    }
                    for record in selected_records:
                        record_id = str(record.get("transfer_winning_record_id") or "")
                        if record_id and record_id not in seen_record_ids:
                            existing_family["transfer_selected_records"].append(
                                {
                                    **record,
                                    "record_rank": len(
                                        existing_family["transfer_selected_records"]
                                    )
                                    + 1,
                                }
                            )
                            seen_record_ids.add(record_id)
    # ``dict`` insertion order is the mechanism-spec order followed by each
    # family's reranker order.  This is the selected-evidence order; a Morgan
    # re-sort here would silently change the flat assay-transfer condition.
    neighbors = list(merged.values())
    for rank, neighbor in enumerate(neighbors, start=1):
        neighbor["rank"] = rank
    payload = {
        "group_id": "Flat.all_evidence",
        "tier": "Flat",
        "endpoint_group": "all_evidence",
        "source_group_ids": sorted(source_group_ids),
        "n_candidate_molecules": len(neighbors),
        "neighbors": neighbors,
    }
    if transfer_selections:
        payload["transfer_neighbor_selection"] = {
            "selection_unit": (
                (transfer_selections[0].get("molecule_collapse") or {}).get(
                    "selection_unit"
                )
                or (transfer_selections[0].get("diversity") or {}).get(
                    "selection_unit"
                )
            ),
            "n_source_families": len(transfer_selections),
            "n_selected": len(neighbors),
            "flat_merge_policy": "stable_family_rank_merge_preserve_family_bundles.v1",
            "selection_metadata_is_llm_hidden": True,
        }
    return payload


def _evidence_row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the stable fields used to deduplicate merged evidence records."""
    return (
        str(row.get("evidence_id") or ""),
        str(row.get("group_id") or ""),
        str(row.get("molecule_chembl_id") or ""),
    )


def _query_only_retrieval(query_smiles: str, *, mode: str) -> dict[str, Any]:
    """Build the retrieval-free payload used by the single-branch condition."""
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
        "coverage": retrieval_coverage([], min_similarity=0.0, top_k_per_group=0),
    }


def _invalid_query(query_smiles: str) -> dict[str, Any]:
    """Return the standard failure payload for an unparseable query molecule."""
    return {
        "query": {"input_smiles": query_smiles, "canonical_smiles": "", "standard_inchi_key": ""},
        "status": "error",
        "errors": [{"code": "INVALID_SMILES", "message": "Could not parse query SMILES.", "recoverable": True}],
        "groups": [],
        "coverage": retrieval_coverage([], min_similarity=0.0, top_k_per_group=0),
    }


def retrieval_coverage(
    groups: list[dict[str, Any]],
    *,
    min_similarity: float,
    top_k_per_group: int,
) -> dict[str, Any]:
    """Compute group and unique-molecule coverage from selected neighbors."""
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


def _source_name(config: BranchRetrievalConfig | None, index: Mapping[str, Any]) -> str:
    """Prefer the task source name and otherwise read it from index metadata."""
    if config is not None:
        return config.source_name
    source = index.get("source") or {}
    return str(source.get("dataset") or source.get("type") or "native")
