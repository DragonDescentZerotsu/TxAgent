"""Cumulative flat/mechanism retrieval views for D/C/H1/H2 experiments."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from tools.chembl_tool.common.evidence_distance import DistanceExpansionConfig
from tools.chembl_tool.common.experiment_retrieval import (
    EvidenceGroupSpec,
    SourceExperimentConfig,
    evidence_family,
    flatten_retrieval_groups,
    retrieval_coverage,
    retrieve_group_specs_view,
    retrieve_experiment_view,
)


DISTANCE_PREFIXES = ("D", "D+C", "D+C+H1", "D+C+H1+H2")


def build_prefix_source_config(
    config: DistanceExpansionConfig,
    prefix: str,
) -> SourceExperimentConfig:
    """Append one stable aggregate tree-node group per C-family depth."""
    if prefix not in DISTANCE_PREFIXES:
        raise ValueError(f"Unsupported distance prefix: {prefix}")
    extension_specs: list[EvidenceGroupSpec] = []
    for tree_node in config.extension_tree_nodes:
        if tree_node.declared_level == "H1" and prefix not in {"D+C+H1", "D+C+H1+H2"}:
            continue
        if tree_node.declared_level == "H2" and prefix != "D+C+H1+H2":
            continue
        extension_specs.append(
            evidence_family(
                tree_node.tree_node_id,
                family_label=f"Distance {tree_node.declared_level}",
                source_group_ids=(tree_node.source_group_id,),
                legacy_output_group_id=tree_node.tree_node_id,
            )
        )
    return SourceExperimentConfig(
        source_name=config.source_name,
        direct_groups=config.base_source_config.direct_groups,
        mechanism_groups=(*config.base_source_config.mechanism_groups, *extension_specs),
    )


def retrieve_distance_view(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    config: DistanceExpansionConfig,
    prefix: str,
    view: str,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str = "operational",
) -> dict[str, Any]:
    """Retrieve one cumulative prefix using the unchanged experiment retrieval path."""
    if view not in {"flat", "mechanism"}:
        raise ValueError("view must be `flat` or `mechanism`.")
    source_config = build_prefix_source_config(config, prefix)
    if prefix == "D":
        mode = "direct"
    else:
        mode = "full_flat" if view == "flat" else "full_mechanism"
    result = retrieve_experiment_view(
        query_smiles,
        index,
        mode=mode,
        config=source_config,
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
    )
    result["distance_expansion"] = {
        "prefix": prefix,
        "view": view,
        "source_release": config.source_release,
        "source_name": config.source_name,
        "mixed_sources": False,
    }
    return result


def retrieve_all_distance_views(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    config: DistanceExpansionConfig,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str = "operational",
) -> dict[str, dict[str, dict[str, Any]]]:
    """Calculate similarity once, then materialize every cumulative flat/mechanism prefix."""
    maximal = build_prefix_source_config(config, "D+C+H1+H2")
    direct_specs = config.base_source_config.direct_groups
    all_specs = (*direct_specs, *maximal.mechanism_groups)
    if len({spec.group_id for spec in all_specs}) != len(all_specs):
        raise ValueError("Direct, base mechanism, and extension group IDs must be unique.")
    maximal_result = retrieve_group_specs_view(
        query_smiles,
        index,
        specs=all_specs,
        source_name=config.source_name,
        mode="distance_maximal",
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
    )
    if maximal_result.get("status") != "ok":
        return {
            prefix: {view: deepcopy(maximal_result) for view in ("flat", "mechanism")}
            for prefix in DISTANCE_PREFIXES
        }

    groups_by_id = {group["group_id"]: group for group in maximal_result["groups"]}
    direct_ids = [spec.group_id for spec in direct_specs]
    base_ids = [spec.group_id for spec in config.base_source_config.mechanism_groups]
    h1_ids = [
        spec.group_id for spec in maximal.mechanism_groups if spec.tier == "Distance H1"
    ]
    h2_ids = [
        spec.group_id for spec in maximal.mechanism_groups if spec.tier == "Distance H2"
    ]
    ids_by_prefix = {
        "D": direct_ids,
        "D+C": base_ids,
        "D+C+H1": [*base_ids, *h1_ids],
        "D+C+H1+H2": [*base_ids, *h1_ids, *h2_ids],
    }
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for prefix, group_ids in ids_by_prefix.items():
        mechanism_groups = [deepcopy(groups_by_id[group_id]) for group_id in group_ids]
        output[prefix] = {}
        for view in ("flat", "mechanism"):
            result = deepcopy(maximal_result)
            if view == "flat":
                result["groups"] = [flatten_retrieval_groups(mechanism_groups)]
            else:
                result["groups"] = deepcopy(mechanism_groups)
            result["coverage"] = retrieval_coverage(
                result["groups"],
                min_similarity=min_similarity,
                top_k_per_group=top_k_per_group,
            )
            result["experiment"]["mode"] = "distance_flat" if view == "flat" else "distance_mechanism"
            result["experiment"]["resolved_group_mapping"] = {
                group_id: maximal_result["experiment"]["resolved_group_mapping"][group_id]
                for group_id in group_ids
            }
            result["distance_expansion"] = {
                "prefix": prefix,
                "view": view,
                "source_release": config.source_release,
                "source_name": config.source_name,
                "mixed_sources": False,
                "similarity_pass": "shared_maximal_view",
            }
            output[prefix][view] = result
    return output
