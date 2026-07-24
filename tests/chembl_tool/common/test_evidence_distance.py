from dataclasses import replace

import pytest

from tools.chembl_tool.common.evidence_distance import (
    DistanceConfigError,
    FamilySelfRelevanceAudit,
    resolve_base_source_partition,
    shortest_admissible_path,
    stable_config_hash,
    validate_distance_config,
    validate_self_relevance_audit,
)
from tools.chembl_tool.tasks.bbb_martins.distance_config import DISTANCE_CONFIG
from tools.chembl_tool.tasks.bbb_martins.distance_self_relevance import SELF_RELEVANCE_AUDIT
from tools.chembl_tool.tasks.bbb_martins.build_distance_assay_manifest import (
    CONFIG as BBB_MANIFEST_CONFIG,
)
from tools.chembl_tool.common.task_workflows.distance_assay_manifest import (
    main as run_distance_manifest,
)


def test_bbb_distance_config_has_expected_shortest_paths():
    validate_distance_config(DISTANCE_CONFIG)

    assert shortest_admissible_path(DISTANCE_CONFIG, "tight_junction_integrity") == (
        "tight_junction_integrity",
    )
    assert shortest_admissible_path(DISTANCE_CONFIG, "pxr_car_activation") == (
        "pxr_car_activation",
    )
    assert shortest_admissible_path(DISTANCE_CONFIG, "mmp9_activity") == (
        "mmp9_activity",
        "tight_junction_integrity",
    )
    assert shortest_admissible_path(DISTANCE_CONFIG, "mmp3_activity") == (
        "mmp3_activity",
        "tight_junction_integrity",
    )
    assert shortest_admissible_path(DISTANCE_CONFIG, "keap1_nrf2_interaction") == (
        "keap1_nrf2_interaction",
        "nrf2_activation",
        "efflux_transporter_abundance",
    )
    assert shortest_admissible_path(DISTANCE_CONFIG, "phd2_activity") == (
        "phd2_activity",
        "hif1_activation",
        "influx_transporter_abundance",
    )


def test_declared_h2_is_rejected_when_shortest_path_is_one_edge():
    original = next(
        family for family in DISTANCE_CONFIG.extension_families if family.family_id == "mmp9_activity"
    )
    invalid_family = replace(original, declared_level="H2")
    invalid = replace(
        DISTANCE_CONFIG,
        extension_families=tuple(
            invalid_family if family.family_id == original.family_id else family
            for family in DISTANCE_CONFIG.extension_families
        ),
    )

    with pytest.raises(DistanceConfigError, match="declares H2 but shortest path is 1 edge"):
        validate_distance_config(invalid)


def test_bbb_base_partition_preserves_direct_full_union():
    available = {
        "Tier 1.context_dependent",
        "Tier 1.direct_brain_plasma",
        "Tier 1.direct_unbound_brain",
        "Tier 2.passive_papp",
        "Tier 3.efflux_inhibition_or_binding",
        "Tier 4.influx_functional_uptake_or_transport",
    }
    partition = resolve_base_source_partition(DISTANCE_CONFIG.base_source_config, available)

    assert partition.direct_groups == (
        "Tier 1.direct_brain_plasma",
        "Tier 1.direct_unbound_brain",
    )
    assert set(partition.direct_groups).isdisjoint(partition.core_increment_groups)
    assert set(partition.direct_groups) | set(partition.core_increment_groups) == set(partition.full_groups)
    assert set(partition.full_groups) == available


def test_distance_config_hash_is_stable_and_source_is_chembl_only():
    assert DISTANCE_CONFIG.source_name == "chembl"
    assert DISTANCE_CONFIG.base_source_config.source_name == "chembl"
    assert stable_config_hash(DISTANCE_CONFIG) == stable_config_hash(DISTANCE_CONFIG)


def test_bbb_self_relevance_audit_covers_every_extension_family():
    validate_self_relevance_audit(DISTANCE_CONFIG, SELF_RELEVANCE_AUDIT)

    by_family = {audit.family_id: audit for audit in SELF_RELEVANCE_AUDIT}
    assert set(by_family) == {
        family.family_id for family in DISTANCE_CONFIG.extension_families
    }
    assert by_family["mmp9_activity"].status == "pass_same_molecule"
    assert by_family["mmp3_activity"].status == "pass_same_molecule"
    assert by_family["nrf2_activation"].status == "requires_query_role"
    assert by_family["keap1_nrf2_interaction"].status == "requires_query_role"
    assert by_family["hif1_activation"].status == "requires_query_role"
    assert by_family["phd2_activity"].status == "requires_query_role"


def test_bbb_v3_is_blocked_by_publishable_self_relevance_gate():
    with pytest.raises(
        DistanceConfigError,
        match="Family `nrf2_activation` is not publishable",
    ):
        validate_self_relevance_audit(
            DISTANCE_CONFIG,
            SELF_RELEVANCE_AUDIT,
            require_publishable=True,
        )


def test_bbb_v3_manifest_rebuild_is_blocked_before_source_scan():
    with pytest.raises(
        DistanceConfigError,
        match="Family `nrf2_activation` is not publishable",
    ):
        run_distance_manifest(
            BBB_MANIFEST_CONFIG,
            ["--chembl-sqlite", "unused.db", "--limit", "1"],
        )


def test_missing_query_role_is_rejected_for_conditional_family():
    original = next(
        audit for audit in SELF_RELEVANCE_AUDIT if audit.family_id == "nrf2_activation"
    )
    invalid = FamilySelfRelevanceAudit(
        family_id=original.family_id,
        status=original.status,
        causal_subject=original.causal_subject,
        required_query_roles=(),
        rationale=original.rationale,
        citations=original.citations,
    )
    audits = tuple(
        invalid if audit.family_id == original.family_id else audit
        for audit in SELF_RELEVANCE_AUDIT
    )

    with pytest.raises(DistanceConfigError, match="requires an explicit missing query role"):
        validate_self_relevance_audit(DISTANCE_CONFIG, audits)


def test_extension_nodes_are_not_frozen_base_nodes():
    base_nodes = {node.node_id for node in DISTANCE_CONFIG.nodes if node.core_anchor}

    assert not {
        family.measured_node for family in DISTANCE_CONFIG.extension_families
    } & base_nodes


def test_each_c_family_has_one_h1_and_each_h1_at_most_one_h2():
    h1_by_parent = {
        family_id: [
            node
            for node in DISTANCE_CONFIG.extension_tree_nodes
            if node.declared_level == "H1" and node.parent_c_family_id == family_id
        ]
        for family_id in DISTANCE_CONFIG.c_family_ids
    }
    assert all(len(nodes) == 1 for nodes in h1_by_parent.values())
    for h1_nodes in h1_by_parent.values():
        h2_children = [
            node
            for node in DISTANCE_CONFIG.extension_tree_nodes
            if node.declared_level == "H2" and node.parent_id == h1_nodes[0].tree_node_id
        ]
        assert len(h2_children) <= 1
