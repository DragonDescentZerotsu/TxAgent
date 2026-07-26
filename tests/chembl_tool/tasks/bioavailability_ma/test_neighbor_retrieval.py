from tools.chembl_tool.tasks.bioavailability_ma.build_evidence_library import build_neighbor_index
from tools.chembl_tool.tasks.bioavailability_ma.retrieve_neighbors import retrieve_neighbors, similarity_bucket
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import _parse_args


def test_retrieve_neighbors_returns_top_per_group_and_excludes_exact_query():
    evidence_rows = [
        _row("CHEMBL_EXACT", "CCO", "Tier 1.direct_absolute_bioavailability", "Bioavailability"),
        _row("CHEMBL_CLOSE", "CCCO", "Tier 1.direct_absolute_bioavailability", "Bioavailability"),
        _row("CHEMBL_PERM", "CCCO", "Tier 3.cell_permeability_papp", "Papp"),
    ]
    index = build_neighbor_index(evidence_rows)

    result = retrieve_neighbors("CCO", index, top_k_per_group=1, min_similarity=0.0)

    groups = {group["group_id"]: group for group in result["groups"]}
    direct_neighbors = groups["Tier 1.direct_absolute_bioavailability"]["neighbors"]
    permeability_neighbors = groups["Tier 3.cell_permeability_papp"]["neighbors"]

    assert direct_neighbors[0]["molecule_chembl_id"] == "CHEMBL_CLOSE"
    assert direct_neighbors[0]["evidence_rows"][0]["standard_type"] == "Bioavailability"
    assert permeability_neighbors[0]["molecule_chembl_id"] == "CHEMBL_PERM"
    assert result["coverage"]["n_groups_with_neighbors"] == 2


def test_similarity_bucket_ranges():
    assert similarity_bucket(0.95) == "very_close_analog"
    assert similarity_bucket(0.80) == "close_analog"
    assert similarity_bucket(0.60) == "moderate_analog"
    assert similarity_bucket(0.40) == "weak_analog"
    assert similarity_bucket(0.20) == "distant_analog"
    assert similarity_bucket(0.19) == "very_distant_analog"


def test_pipeline_accepts_group_filter():
    groups = [
        "Tier 1.direct_absolute_bioavailability",
        "Tier 1.context_dependent",
    ]

    args = _parse_args(["--groups", *groups])

    assert args.groups == groups


def test_pipeline_accepts_query_feature_coverage_selector():
    args = _parse_args(["--neighbor-selector", "query_feature_coverage"])

    assert args.neighbor_selector == "query_feature_coverage"


def _row(molecule_id: str, smiles: str, group_id: str, standard_type: str) -> dict:
    tier, endpoint_group = group_id.split(".", 1)
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "standard_inchi_key": "",
        "assay_chembl_id": f"ASSAY_{molecule_id}",
        "assay_tier": tier,
        "endpoint_group": endpoint_group,
        "group_id": group_id,
        "standard_type": standard_type,
        "standard_relation": "=",
        "standard_value": "1",
        "standard_units": "",
        "assay_description": "test assay",
        "evidence_direction": "context_dependent",
        "evidence_strength": "weak",
    }
