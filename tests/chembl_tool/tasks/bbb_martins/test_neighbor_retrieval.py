from tools.chembl_tool.tasks.bbb_martins.build_evidence_library import build_neighbor_index
from tools.chembl_tool.tasks.bbb_martins.chembl_exact_context import attach_shared_assay_context
from tools.chembl_tool.tasks.bbb_martins.retrieve_neighbors import (
    _is_exact_same_molecule,
    retrieve_neighbors,
    similarity_bucket,
)


def test_retrieve_neighbors_returns_top_per_group_and_excludes_exact_query():
    evidence_rows = [
        _row("CHEMBL_EXACT", "CCO", "Tier 2.passive_papp", "Papp"),
        _row("CHEMBL_CLOSE", "CCCO", "Tier 2.passive_papp", "Papp"),
        _row("CHEMBL_WEAK", "c1ccccc1", "Tier 2.passive_papp", "Papp"),
        _row("CHEMBL_EFFLUX", "CCCO", "Tier 3.efflux_inhibition_or_binding", "IC50"),
    ]
    index = build_neighbor_index(evidence_rows)

    result = retrieve_neighbors("CCO", index, top_k_per_group=1, min_similarity=0.0)

    groups = {group["group_id"]: group for group in result["groups"]}
    passive_neighbors = groups["Tier 2.passive_papp"]["neighbors"]
    efflux_neighbors = groups["Tier 3.efflux_inhibition_or_binding"]["neighbors"]

    assert passive_neighbors[0]["molecule_chembl_id"] == "CHEMBL_CLOSE"
    assert passive_neighbors[0]["rank"] == 1
    assert passive_neighbors[0]["evidence_rows"][0]["standard_type"] == "Papp"
    assert efflux_neighbors[0]["molecule_chembl_id"] == "CHEMBL_EFFLUX"
    assert result["coverage"]["n_groups_with_neighbors"] == 2


def test_similarity_bucket_includes_distant_analog_ranges():
    assert similarity_bucket(0.95) == "very_close_analog"
    assert similarity_bucket(0.80) == "close_analog"
    assert similarity_bucket(0.60) == "moderate_analog"
    assert similarity_bucket(0.40) == "weak_analog"
    assert similarity_bucket(0.20) == "distant_analog"
    assert similarity_bucket(0.19) == "very_distant_analog"


def test_exact_same_molecule_excludes_same_inchi_key_connectivity_layer():
    molecule = {
        "standard_inchi_key": "PMXMIIMHBWHSKN-UHFFFAOYSA-N",
        "canonical_smiles": "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2O",
    }

    assert _is_exact_same_molecule(
        molecule,
        "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCC[C@H]2O",
        "PMXMIIMHBWHSKN-LJQANCHMSA-N",
    )


def test_exact_same_molecule_keeps_different_inchi_key_connectivity_layer():
    molecule = {
        "standard_inchi_key": "AAAAAAAAAAAAAA-UHFFFAOYSA-N",
        "canonical_smiles": "CCCO",
    }

    assert not _is_exact_same_molecule(molecule, "CCO", "PMXMIIMHBWHSKN-LJQANCHMSA-N")


def test_shared_assay_context_splits_same_endpoint_from_different_endpoint():
    retrieval = {
        "groups": [
            {
                "neighbors": [
                    {
                        "evidence_rows": [
                            {
                                "assay_chembl_id": "CHEMBL_A1",
                                "standard_type": "Papp",
                                "standard_value": "12",
                                "standard_units": "10^-6 cm/s",
                            },
                            {
                                "assay_chembl_id": "CHEMBL_A2",
                                "standard_type": "Efflux ratio",
                                "standard_value": "3",
                                "standard_units": "",
                            },
                        ]
                    }
                ]
            }
        ]
    }
    query_activities_by_assay = {
        "CHEMBL_A1": [
            {
                "assay_chembl_id": "CHEMBL_A1",
                "standard_type": " papp ",
                "standard_value": "8",
                "standard_units": "10^-6 cm/s",
            },
            {
                "assay_chembl_id": "CHEMBL_A1",
                "standard_type": "IC50",
                "standard_value": "100",
                "standard_units": "nM",
            },
        ],
        "CHEMBL_A2": [
            {
                "assay_chembl_id": "CHEMBL_A2",
                "standard_type": "IC50",
                "standard_value": "50",
                "standard_units": "nM",
            }
        ],
    }

    attach_shared_assay_context(retrieval, query_activities_by_assay)

    context = retrieval["groups"][0]["neighbors"][0]["shared_assay_context"]
    assert context["n_same_endpoint_activity"] == 1
    assert context["same_endpoint_activity"][0]["standard_type_match"]
    assert context["same_endpoint_activity"][0]["units_match"]
    assert context["same_endpoint_activity"][0]["query_activity"]["standard_value"] == "8"

    assert context["n_same_assay_different_endpoint_activity"] == 2
    endpoint_names = {
        card["query_activity"]["standard_type"] for card in context["same_assay_different_endpoint_activity"]
    }
    assert endpoint_names == {"IC50"}


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
