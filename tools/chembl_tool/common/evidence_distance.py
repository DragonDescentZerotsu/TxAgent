"""Schemas and validation for cumulative evidence-distance experiments."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.chembl_tool.common.experiment_retrieval import SourceExperimentConfig


ALLOWED_RELATION_TYPES = {
    "causal_process",
    "functional_component",
    "exposure_link",
    "validated_mechanistic_readout",
}
ALLOWED_SELF_RELEVANCE_STATUSES = {
    "pass_same_molecule",
    "requires_query_role",
    "context_only",
    "unresolved",
}
FORWARD = "upstream_to_downstream"
REVERSE = "downstream_to_upstream"


class DistanceConfigError(ValueError):
    """Raised when a distance graph or family declaration is not publishable."""


@dataclass(frozen=True)
class FamilySelfRelevanceAudit:
    """Audit whether a family can inform the assayed molecule's own task outcome.

    A graph can be biologically correct while switching causal subject.  For
    example, molecule A may induce a transporter that changes the disposition
    of substrate B.  Such a path cannot predict A's own disposition unless the
    evidence also establishes that A has the required substrate role.
    """

    family_id: str
    status: str
    causal_subject: str
    required_query_roles: tuple[str, ...]
    rationale: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class BiologicalNode:
    node_id: str
    display_name: str
    definition: str
    measurable_state: str
    core_anchor: bool = False


@dataclass(frozen=True)
class MechanismEdge:
    edge_id: str
    upstream_node: str
    downstream_node: str
    relation_type: str
    effect_direction: str
    admissible_inference_directions: tuple[str, ...]
    applicability: str
    rationale: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class DistanceFamilySpec:
    """One assay-measured biological family assigned to exactly one tree node."""

    family_id: str
    display_name: str
    tree_node_id: str
    parent_c_family_id: str
    source_group_id: str
    measured_node: str
    declared_level: str
    admissible_path: tuple[str, ...]
    scope_rule: str
    quality_rule: str


@dataclass(frozen=True)
class DistanceTreeNodeSpec:
    """One aggregate retrieval/reasoning node below a frozen C mechanism family."""

    tree_node_id: str
    display_name: str
    parent_id: str
    parent_c_family_id: str
    source_group_id: str
    declared_level: str
    anchor_nodes: tuple[str, ...]
    measurement_family_ids: tuple[str, ...]


@dataclass(frozen=True)
class DistanceExpansionConfig:
    task_name: str
    source_name: str
    source_release: str
    base_source_config: SourceExperimentConfig
    c_family_ids: tuple[str, ...]
    nodes: tuple[BiologicalNode, ...]
    edges: tuple[MechanismEdge, ...]
    extension_tree_nodes: tuple[DistanceTreeNodeSpec, ...]
    extension_families: tuple[DistanceFamilySpec, ...]


@dataclass(frozen=True)
class BaseSourcePartition:
    direct_groups: tuple[str, ...]
    core_increment_groups: tuple[str, ...]
    full_groups: tuple[str, ...]


def resolve_base_source_partition(
    source_config: SourceExperimentConfig,
    available_source_groups: Iterable[str],
) -> BaseSourcePartition:
    """Resolve frozen D/C groups without changing the existing source mapping."""
    available = set(available_source_groups)
    direct = {
        source_group
        for spec in source_config.direct_groups
        for source_group in spec.resolve(available)
    }
    full = {
        source_group
        for spec in source_config.mechanism_groups
        for source_group in spec.resolve(available)
    }
    if not direct:
        raise DistanceConfigError("D resolved to an empty source-group set.")
    if not full:
        raise DistanceConfigError("D+C resolved to an empty source-group set.")
    if not direct <= full:
        missing = sorted(direct - full)
        raise DistanceConfigError(f"Direct groups are absent from the full evidence union: {missing}")
    core_increment = full - direct
    return BaseSourcePartition(
        direct_groups=tuple(sorted(direct)),
        core_increment_groups=tuple(sorted(core_increment)),
        full_groups=tuple(sorted(full)),
    )


def shortest_admissible_path(
    config: DistanceExpansionConfig,
    start_node: str,
    target_nodes: Iterable[str] | None = None,
) -> tuple[str, ...] | None:
    """Return a deterministic shortest directed path to any frozen base anchor."""
    nodes = {node.node_id for node in config.nodes}
    if start_node not in nodes:
        raise DistanceConfigError(f"Unknown start node: {start_node}")
    targets = set(target_nodes or (node.node_id for node in config.nodes if node.core_anchor))
    unknown_targets = targets - nodes
    if unknown_targets:
        raise DistanceConfigError(f"Unknown target nodes: {sorted(unknown_targets)}")
    if start_node in targets:
        return (start_node,)

    adjacency = _admissible_adjacency(config.edges)
    queue: deque[tuple[str, ...]] = deque([(start_node,)])
    visited = {start_node}
    while queue:
        path = queue.popleft()
        for neighbor in sorted(adjacency.get(path[-1], ())):
            if neighbor in visited:
                continue
            next_path = (*path, neighbor)
            if neighbor in targets:
                return next_path
            visited.add(neighbor)
            queue.append(next_path)
    return None


def validate_distance_config(
    config: DistanceExpansionConfig,
    *,
    available_source_groups: Iterable[str] | None = None,
) -> None:
    """Validate graph structure, shortest paths, source, and optional D/C parity."""
    errors: list[str] = []
    if config.source_name.lower() != "chembl":
        errors.append("E12 source_name must be `chembl` for D/C/H1/H2.")
    if not config.source_release.strip():
        errors.append("source_release must be frozen and non-empty.")
    if config.base_source_config.source_name.lower() != "chembl":
        errors.append("base_source_config must also be ChEMBL; mixed-source D/C is forbidden.")

    nodes = {node.node_id: node for node in config.nodes}
    if len(nodes) != len(config.nodes):
        errors.append("Biological node IDs must be unique.")
    if not any(node.core_anchor for node in config.nodes):
        errors.append("At least one frozen base anchor is required.")
    for node in config.nodes:
        if not all((node.node_id, node.display_name, node.definition, node.measurable_state)):
            errors.append(f"Node `{node.node_id}` has incomplete metadata.")

    edge_ids: set[str] = set()
    for edge in config.edges:
        if edge.edge_id in edge_ids:
            errors.append(f"Duplicate edge ID: {edge.edge_id}")
        edge_ids.add(edge.edge_id)
        if edge.upstream_node not in nodes or edge.downstream_node not in nodes:
            errors.append(f"Edge `{edge.edge_id}` references an unknown node.")
        if edge.relation_type not in ALLOWED_RELATION_TYPES:
            errors.append(f"Edge `{edge.edge_id}` has unsupported relation type `{edge.relation_type}`.")
        if not edge.admissible_inference_directions or not set(edge.admissible_inference_directions) <= {
            FORWARD,
            REVERSE,
        }:
            errors.append(f"Edge `{edge.edge_id}` has invalid inference directions.")
        if not edge.applicability.strip() or not edge.rationale.strip() or not edge.effect_direction.strip():
            errors.append(f"Edge `{edge.edge_id}` lacks applicability, rationale, or direction.")
        if not edge.citations or any(not citation.strip() for citation in edge.citations):
            errors.append(f"Edge `{edge.edge_id}` requires at least one checkable citation.")

    base_mechanism_family_ids = {spec.group_id for spec in config.base_source_config.mechanism_groups}
    base_c_family_ids = set(config.c_family_ids)
    if not base_c_family_ids:
        errors.append("At least one C mechanism family must be declared for tree expansion.")
    unknown_c_families = base_c_family_ids - base_mechanism_family_ids
    if unknown_c_families:
        errors.append(f"Unknown C mechanism families: {sorted(unknown_c_families)}")
    if len(base_c_family_ids) != len(config.c_family_ids):
        errors.append("C mechanism family IDs must be unique.")
    tree_nodes = {node.tree_node_id: node for node in config.extension_tree_nodes}
    if len(tree_nodes) != len(config.extension_tree_nodes):
        errors.append("Distance tree-node IDs must be unique.")
    tree_source_groups: set[str] = set()
    h1_by_c: dict[str, list[DistanceTreeNodeSpec]] = {family_id: [] for family_id in base_c_family_ids}
    h2_by_h1: dict[str, list[DistanceTreeNodeSpec]] = {}
    for tree_node in config.extension_tree_nodes:
        if tree_node.declared_level not in {"H1", "H2"}:
            errors.append(
                f"Tree node `{tree_node.tree_node_id}` has invalid level `{tree_node.declared_level}`."
            )
        if tree_node.source_group_id in tree_source_groups:
            errors.append(f"Tree source group `{tree_node.source_group_id}` is assigned more than once.")
        tree_source_groups.add(tree_node.source_group_id)
        expected_prefix = f"Distance {tree_node.declared_level}."
        if not tree_node.source_group_id.startswith(expected_prefix):
            errors.append(
                f"Tree node `{tree_node.tree_node_id}` source group must start with `{expected_prefix}`."
            )
        if not tree_node.display_name.strip() or not tree_node.measurement_family_ids:
            errors.append(f"Tree node `{tree_node.tree_node_id}` lacks display metadata or measurement families.")
        if not tree_node.anchor_nodes or any(anchor not in nodes for anchor in tree_node.anchor_nodes):
            errors.append(f"Tree node `{tree_node.tree_node_id}` has invalid or empty anchor nodes.")
        if tree_node.parent_c_family_id not in base_c_family_ids:
            errors.append(
                f"Tree node `{tree_node.tree_node_id}` references unknown C family "
                f"`{tree_node.parent_c_family_id}`."
            )
        if tree_node.declared_level == "H1":
            if tree_node.parent_id != tree_node.parent_c_family_id:
                errors.append(
                    f"H1 tree node `{tree_node.tree_node_id}` must be directly parented by its C family."
                )
            h1_by_c.setdefault(tree_node.parent_c_family_id, []).append(tree_node)
        else:
            h2_by_h1.setdefault(tree_node.parent_id, []).append(tree_node)

    for c_family_id in sorted(base_c_family_ids):
        n_h1 = len(h1_by_c.get(c_family_id, ()))
        if n_h1 != 1:
            errors.append(
                f"C family `{c_family_id}` must have exactly one aggregate H1 child; found {n_h1}."
            )
    for h1_id, h2_nodes in sorted(h2_by_h1.items()):
        parent = tree_nodes.get(h1_id)
        if parent is None or parent.declared_level != "H1":
            errors.append(f"H2 parent `{h1_id}` is not a declared H1 tree node.")
            continue
        if len(h2_nodes) > 1:
            errors.append(f"H1 tree node `{h1_id}` has more than one H2 child.")
        for h2_node in h2_nodes:
            if h2_node.parent_c_family_id != parent.parent_c_family_id:
                errors.append(
                    f"H2 tree node `{h2_node.tree_node_id}` changes C-family ancestry."
                )

    family_ids: set[str] = set()
    declared_tree_memberships: dict[str, str] = {}
    for tree_node in config.extension_tree_nodes:
        for family_id in tree_node.measurement_family_ids:
            if family_id in declared_tree_memberships:
                errors.append(
                    f"Measurement family `{family_id}` is assigned to multiple tree nodes."
                )
            declared_tree_memberships[family_id] = tree_node.tree_node_id

    for family in config.extension_families:
        if family.family_id in family_ids:
            errors.append(f"Duplicate distance family ID: {family.family_id}")
        family_ids.add(family.family_id)
        tree_node = tree_nodes.get(family.tree_node_id)
        if tree_node is None:
            errors.append(
                f"Family `{family.family_id}` references unknown tree node `{family.tree_node_id}`."
            )
            continue
        if declared_tree_memberships.get(family.family_id) != family.tree_node_id:
            errors.append(
                f"Family `{family.family_id}` is not listed exactly once by tree node `{family.tree_node_id}`."
            )
        if family.parent_c_family_id != tree_node.parent_c_family_id:
            errors.append(f"Family `{family.family_id}` changes its tree node's C-family parent.")
        if family.declared_level != tree_node.declared_level:
            errors.append(f"Family `{family.family_id}` level differs from its tree node.")
        if family.source_group_id != tree_node.source_group_id:
            errors.append(f"Family `{family.family_id}` source group differs from its tree node.")
        if family.measured_node not in nodes:
            errors.append(f"Family `{family.family_id}` references unknown node `{family.measured_node}`.")
            continue
        if family.declared_level not in {"H1", "H2"}:
            errors.append(f"Family `{family.family_id}` has invalid level `{family.declared_level}`.")
            continue
        if not family.scope_rule.strip() or not family.quality_rule.strip():
            errors.append(f"Family `{family.family_id}` lacks scope or quality rules.")
        try:
            shortest = shortest_admissible_path(config, family.measured_node)
        except DistanceConfigError as exc:
            errors.append(str(exc))
            continue
        expected_edges = 1 if family.declared_level == "H1" else 2
        if shortest is None:
            errors.append(f"Family `{family.family_id}` has no admissible path to the base envelope.")
        elif len(shortest) - 1 != expected_edges:
            errors.append(
                f"Family `{family.family_id}` declares {family.declared_level} but shortest path is "
                f"{len(shortest) - 1} edge(s): {shortest}"
            )
        if tuple(family.admissible_path) != shortest:
            errors.append(
                f"Family `{family.family_id}` path {family.admissible_path} does not equal deterministic "
                f"shortest path {shortest}."
            )
        try:
            to_parent = shortest_admissible_path(
                config,
                family.measured_node,
                target_nodes=tree_node.anchor_nodes,
            )
        except DistanceConfigError as exc:
            errors.append(str(exc))
            continue
        if to_parent is None or len(to_parent) - 1 != 1:
            errors.append(
                f"Family `{family.family_id}` is not one edge from tree parent anchors "
                f"{tree_node.anchor_nodes}: {to_parent}."
            )

    measured_nodes_by_tree: dict[str, set[str]] = {}
    for family in config.extension_families:
        measured_nodes_by_tree.setdefault(family.tree_node_id, set()).add(family.measured_node)
    core_anchor_ids = {node.node_id for node in config.nodes if node.core_anchor}
    for tree_node in config.extension_tree_nodes:
        anchors = set(tree_node.anchor_nodes)
        if tree_node.declared_level == "H1" and not anchors <= core_anchor_ids:
            errors.append(
                f"H1 tree node `{tree_node.tree_node_id}` anchors must belong to frozen B nodes."
            )
        if tree_node.declared_level == "H2":
            parent_nodes = measured_nodes_by_tree.get(tree_node.parent_id, set())
            if not anchors <= parent_nodes:
                errors.append(
                    f"H2 tree node `{tree_node.tree_node_id}` anchors {sorted(anchors)} are not measured "
                    f"by parent H1 `{tree_node.parent_id}`."
                )

    undeclared_families = sorted(set(declared_tree_memberships) - family_ids)
    ungrouped_families = sorted(family_ids - set(declared_tree_memberships))
    if undeclared_families:
        errors.append(f"Tree nodes list missing family specs: {undeclared_families}")
    if ungrouped_families:
        errors.append(f"Family specs are absent from tree nodes: {ungrouped_families}")

    if available_source_groups is not None:
        try:
            resolve_base_source_partition(config.base_source_config, available_source_groups)
        except DistanceConfigError as exc:
            errors.append(str(exc))

    if errors:
        raise DistanceConfigError("Invalid distance config:\n- " + "\n- ".join(errors))


def validate_self_relevance_audit(
    config: DistanceExpansionConfig,
    audits: Iterable[FamilySelfRelevanceAudit],
    *,
    require_publishable: bool = False,
) -> None:
    """Validate complete causal-subject auditing for all extension families.

    Structural graph validation alone cannot detect a subject switch such as
    ``query activates regulator -> transporter abundance -> transport of some
    other substrate``.  This audit is deliberately family-level because the
    missing query role is a property of the inference being attempted, not an
    assay-quality or organism-distance field.

    When ``require_publishable`` is true, every family must have
    ``pass_same_molecule`` status.  A family that requires an unobserved query
    role may only pass after that role is explicitly supplied by the retrieval
    contract; merely naming the role here is not sufficient.
    """
    audit_rows = tuple(audits)
    errors: list[str] = []
    known_family_ids = {family.family_id for family in config.extension_families}
    seen_family_ids: set[str] = set()

    for audit in audit_rows:
        if audit.family_id in seen_family_ids:
            errors.append(f"Duplicate self-relevance audit for family `{audit.family_id}`.")
        seen_family_ids.add(audit.family_id)
        if audit.family_id not in known_family_ids:
            errors.append(f"Self-relevance audit references unknown family `{audit.family_id}`.")
        if audit.status not in ALLOWED_SELF_RELEVANCE_STATUSES:
            errors.append(
                f"Family `{audit.family_id}` has unsupported self-relevance status `{audit.status}`."
            )
        if not audit.causal_subject.strip() or not audit.rationale.strip():
            errors.append(
                f"Family `{audit.family_id}` lacks causal-subject or self-relevance rationale."
            )
        if not audit.citations or any(not citation.strip() for citation in audit.citations):
            errors.append(
                f"Family `{audit.family_id}` requires at least one checkable self-relevance citation."
            )
        if audit.status == "requires_query_role" and not audit.required_query_roles:
            errors.append(
                f"Family `{audit.family_id}` requires an explicit missing query role."
            )
        if audit.status == "pass_same_molecule" and audit.required_query_roles:
            errors.append(
                f"Family `{audit.family_id}` cannot pass while retaining unmet query-role requirements."
            )
        if require_publishable and audit.status != "pass_same_molecule":
            roles = ", ".join(audit.required_query_roles) or "none declared"
            errors.append(
                f"Family `{audit.family_id}` is not publishable: status={audit.status}; "
                f"required_query_roles={roles}."
            )

    missing = sorted(known_family_ids - seen_family_ids)
    if missing:
        errors.append(f"Missing self-relevance audits for families: {missing}")

    if errors:
        raise DistanceConfigError("Invalid self-relevance audit:\n- " + "\n- ".join(errors))


def config_to_dict(config: DistanceExpansionConfig) -> dict[str, Any]:
    """Serialize a config without embedding executable source-config objects."""
    payload = asdict(config)
    payload["base_source_config"] = {
        "source_name": config.base_source_config.source_name,
        "direct_groups": [asdict(spec) for spec in config.base_source_config.direct_groups],
        "mechanism_groups": [asdict(spec) for spec in config.base_source_config.mechanism_groups],
    }
    return payload


def stable_config_hash(config: DistanceExpansionConfig) -> str:
    payload = json.dumps(config_to_dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _admissible_adjacency(edges: Iterable[MechanismEdge]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        if FORWARD in edge.admissible_inference_directions:
            adjacency.setdefault(edge.upstream_node, set()).add(edge.downstream_node)
        if REVERSE in edge.admissible_inference_directions:
            adjacency.setdefault(edge.downstream_node, set()).add(edge.upstream_node)
    return adjacency
