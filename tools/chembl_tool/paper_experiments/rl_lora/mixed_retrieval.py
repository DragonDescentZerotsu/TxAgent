"""Full-flat retrieval with a stricter policy only for direct evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.experiment_retrieval import (
    SourceExperimentConfig,
    flatten_retrieval_groups,
    retrieval_coverage,
    retrieve_group_specs_view,
)
from tools.chembl_tool.common.neighbor_selection import SIMILARITY_SELECTOR


MIXED_IDENTITY_POLICY_VERSION = "mixed_family_identity_policy.v1"
DIRECT_POLICY = "scaffold_disjoint"
NON_DIRECT_POLICY = "parent_disjoint"


def retrieve_full_flat_mixed_identity(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    config: SourceExperimentConfig,
    top_k_per_group: int = 3,
    min_similarity: float = 0.3,
    neighbor_selector: str = SIMILARITY_SELECTOR,
) -> dict[str, Any]:
    """Retrieve direct as scaffold-disjoint and other families as parent-disjoint."""
    available_groups = set(index["group_to_molecule_indices"])
    direct_source_groups = {
        source_group
        for spec in config.direct_groups
        for source_group in spec.resolve(available_groups)
    }
    if not direct_source_groups:
        raise ValueError("direct evidence config resolves to no source groups")

    direct_specs = []
    non_direct_specs = []
    family_policies: dict[str, str] = {}
    resolved_mapping: dict[str, list[str]] = {}
    for spec in config.mechanism_groups:
        resolved = set(spec.resolve(available_groups))
        resolved_mapping[spec.group_id] = sorted(resolved)
        direct_overlap = resolved & direct_source_groups
        if direct_overlap and not resolved <= direct_source_groups:
            raise ValueError(
                f"mechanism group {spec.group_id} mixes direct and non-direct source groups"
            )
        if direct_overlap:
            direct_specs.append(spec)
            family_policies[spec.group_id] = DIRECT_POLICY
        else:
            non_direct_specs.append(spec)
            family_policies[spec.group_id] = NON_DIRECT_POLICY
    if not direct_specs:
        raise ValueError(
            "no full-flat mechanism group maps to the direct source family"
        )

    results = []
    for specs, policy in (
        (tuple(direct_specs), DIRECT_POLICY),
        (tuple(non_direct_specs), NON_DIRECT_POLICY),
    ):
        if not specs:
            continue
        result = retrieve_group_specs_view(
            query_smiles,
            index,
            specs=specs,
            source_name=config.source_name,
            mode="full_mechanism",
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy=policy,
            neighbor_selector=neighbor_selector,
        )
        if result.get("status") != "ok":
            return result
        results.append(result)

    group_by_id = {
        str(group["group_id"]): group
        for result in results
        for group in result["groups"]
    }
    groups = [group_by_id[spec.group_id] for spec in config.mechanism_groups]
    output = dict(results[0])
    output["groups"] = [flatten_retrieval_groups(groups)]
    output["coverage"] = retrieval_coverage(
        output["groups"],
        min_similarity=min_similarity,
        top_k_per_group=top_k_per_group,
    )
    output["experiment"] = {
        **output["experiment"],
        "mode": "full_flat",
        "resolved_group_mapping": resolved_mapping,
        "neighbor_identity_policy": "mixed",
        "family_identity_policy": {
            "version": MIXED_IDENTITY_POLICY_VERSION,
            "direct": DIRECT_POLICY,
            "non_direct": NON_DIRECT_POLICY,
            "group_policies": family_policies,
        },
    }
    output["experiment"].pop("excluded_relations", None)
    return output
