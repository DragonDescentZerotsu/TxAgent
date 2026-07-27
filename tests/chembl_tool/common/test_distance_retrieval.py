import pytest

from tools.chembl_tool.common.distance_index import merge_neighbor_indices
from tools.chembl_tool.common.distance_retrieval import (
    build_prefix_source_config,
    retrieve_all_distance_views,
    retrieve_distance_view,
)
from tools.chembl_tool.common.task_workflows.evidence_library import build_neighbor_index
from tools.chembl_tool.tasks.bbb_martins.distance_config import DISTANCE_CONFIG


def _row(molecule_id, smiles, group_id, value, source="ChEMBL"):
    tier, endpoint = group_id.split(".", 1)
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "group_id": group_id,
        "assay_tier": tier,
        "endpoint_group": endpoint,
        "standard_type": endpoint,
        "standard_value": value,
        "standard_units": "%",
        "evidence_source": source,
    }


def _base_index():
    return build_neighbor_index(
        [
            _row("direct", "CCN", "Tier 1.direct_brain_plasma", 1),
            _row("passive", "CCC", "Tier 2.passive_papp", 2),
            _row("efflux", "CCCl", "Tier 3.efflux_inhibition_or_binding", 3),
            _row("influx", "CCBr", "Tier 4.influx_functional_uptake_or_transport", 4),
        ],
        index_version="base.v1",
    )


def _extension_index():
    return build_neighbor_index(
        [
            _row("mmp9", "CCCO", "Distance H1.passive_permeability", 5),
            _row("mmp3", "CCCCN", "Distance H1.passive_permeability", 6),
            _row("nrf2", "CCCF", "Distance H1.efflux_transport", 7),
            _row("keap1", "CCCS", "Distance H2.efflux_transport", 8),
            _row("hif1", "CCCP", "Distance H1.influx_transport", 9),
            _row("phd2", "CCCCO", "Distance H2.influx_transport", 10),
        ],
        index_version="extension.v1",
    )


def test_prefix_configs_keep_h1_branch_stable_when_h2_is_added():
    h1 = build_prefix_source_config(DISTANCE_CONFIG, "D+C+H1")
    h2 = build_prefix_source_config(DISTANCE_CONFIG, "D+C+H1+H2")

    h1_spec = next(
        spec
        for spec in h1.mechanism_groups
        if spec.group_id == "Distance.h1.passive_permeability"
    )
    h1_spec_in_h2 = next(
        spec
        for spec in h2.mechanism_groups
        if spec.group_id == "Distance.h1.passive_permeability"
    )
    assert h1_spec == h1_spec_in_h2
    assert [spec.endpoint_group for spec in h1.mechanism_groups if spec.tier == "Distance H2"] == []
    assert {spec.group_id for spec in h2.mechanism_groups if spec.tier == "Distance H2"} == {
        "Distance.h2.efflux_transport",
        "Distance.h2.influx_transport",
    }


def test_flat_and_mechanism_distance_prefixes_have_identical_evidence_rows():
    index = merge_neighbor_indices(
        _base_index(),
        _extension_index(),
        index_version="distance.v1",
        source_release="ChEMBL 36",
    )
    mechanism = retrieve_distance_view(
        "CO",
        index,
        config=DISTANCE_CONFIG,
        prefix="D+C+H1+H2",
        view="mechanism",
        top_k_per_group=2,
        min_similarity=0.0,
    )
    flat = retrieve_distance_view(
        "CO",
        index,
        config=DISTANCE_CONFIG,
        prefix="D+C+H1+H2",
        view="flat",
        top_k_per_group=2,
        min_similarity=0.0,
    )

    mechanism_values = sorted(
        row["standard_value"]
        for group in mechanism["groups"]
        for neighbor in group["neighbors"]
        for row in neighbor["evidence_rows"]
    )
    flat_values = sorted(
        row["standard_value"]
        for neighbor in flat["groups"][0]["neighbors"]
        for row in neighbor["evidence_rows"]
    )
    assert mechanism_values == flat_values
    assert mechanism["distance_expansion"]["source_name"] == "chembl"
    assert not mechanism["distance_expansion"]["mixed_sources"]


def test_distance_index_rejects_non_chembl_extension_rows():
    bad_extension = build_neighbor_index(
        [_row("bad", "CCCO", "Distance H1.passive_permeability", 5, source="Starling")],
        index_version="bad.v1",
    )

    with pytest.raises(ValueError, match="cannot mix non-ChEMBL"):
        merge_neighbor_indices(
            _base_index(),
            bad_extension,
            index_version="distance.v1",
            source_release="ChEMBL 36",
        )


def test_distance_index_merges_shared_molecule_without_rebuilding_base_rows():
    base = _base_index()
    extension = build_neighbor_index(
        [_row("direct", "CCN", "Distance H1.passive_permeability", 5)],
        index_version="extension.v1",
    )
    base_row = base["evidence_by_molecule_group"]["direct"]["Tier 1.direct_brain_plasma"][0]

    merged = merge_neighbor_indices(
        base,
        extension,
        index_version="distance.v1",
        source_release="ChEMBL 36",
    )

    groups = merged["evidence_by_molecule_group"]["direct"]
    assert set(groups) == {
        "Tier 1.direct_brain_plasma",
        "Distance H1.passive_permeability",
    }
    assert groups["Tier 1.direct_brain_plasma"][0] is base_row
    assert len(merged["molecules"]) == len(base["molecules"])


def test_shared_maximal_retrieval_materializes_nested_prefixes():
    index = merge_neighbor_indices(
        _base_index(),
        _extension_index(),
        index_version="distance.v1",
        source_release="ChEMBL 36",
    )
    views = retrieve_all_distance_views(
        "CO",
        index,
        config=DISTANCE_CONFIG,
        top_k_per_group=2,
        min_similarity=0.0,
    )

    assert set(views) == {"D", "D+C", "D+C+H1", "D+C+H1+H2"}
    assert views["D+C+H1"]["mechanism"]["distance_expansion"]["similarity_pass"] == (
        "shared_maximal_view"
    )
    h1_groups = {
        group["group_id"]: group for group in views["D+C+H1"]["mechanism"]["groups"]
    }
    h2_groups = {
        group["group_id"]: group for group in views["D+C+H1+H2"]["mechanism"]["groups"]
    }
    h1_id = "Distance.h1.passive_permeability"
    assert h1_groups[h1_id] == h2_groups[h1_id]


def test_measurement_families_under_one_tree_node_share_one_top_k_budget():
    index = merge_neighbor_indices(
        _base_index(),
        _extension_index(),
        index_version="distance.v1",
        source_release="ChEMBL 36",
    )
    view = retrieve_distance_view(
        "CO",
        index,
        config=DISTANCE_CONFIG,
        prefix="D+C+H1",
        view="mechanism",
        top_k_per_group=1,
        min_similarity=0.0,
    )

    passive_h1 = next(
        group
        for group in view["groups"]
        if group["group_id"] == "Distance.h1.passive_permeability"
    )
    assert len(passive_h1["neighbors"]) == 1
    assert view["experiment"]["resolved_group_mapping"][
        "Distance.h1.passive_permeability"
    ] == ["Distance H1.passive_permeability"]
