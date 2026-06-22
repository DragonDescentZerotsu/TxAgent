from tools.chembl_tool.tasks.bioavailability_ma.build_combined_tier1_starling_evidence_library import (
    COMBINED_GROUP_ID,
    build_combined_evidence_rows,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_evidence_library import build_neighbor_index
from tools.chembl_tool.tasks.bioavailability_ma.retrieve_neighbors import retrieve_neighbors
from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import _group_evidence_source


def test_combined_index_merges_shared_molecule_evidence():
    chembl = _index(
        "CHEMBL_SHARED",
        "CCCO",
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "Tier 1.direct_absolute_bioavailability",
        "ChEMBL F",
    )
    starling = _index(
        "STARLING_SHARED",
        "OCCC",
        "LFQSCWFLJHTTHZ-TESTTESTSA-N",
        "Starling.direct_oral_bioavailability",
        "Starling F",
    )

    rows, stats = build_combined_evidence_rows(chembl, starling)

    assert stats["n_combined_molecules"] == 1
    assert stats["n_shared_molecules"] == 1
    assert {row["evidence_source"] for row in rows} == {
        "ChEMBL",
        "starling-labs/Oral_Bioavailability",
    }
    assert {row["source_molecule_id"] for row in rows} == {"CHEMBL_SHARED", "STARLING_SHARED"}
    assert all(row["group_id"] == COMBINED_GROUP_ID for row in rows)


def test_combined_retrieval_returns_shared_sources_in_one_neighbor():
    chembl = _index(
        "CHEMBL_SHARED",
        "CCCO",
        "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "Tier 1.direct_absolute_bioavailability",
        "ChEMBL F",
    )
    starling = _index(
        "STARLING_SHARED",
        "OCCC",
        "LFQSCWFLJHTTHZ-TESTTESTSA-N",
        "Starling.direct_oral_bioavailability",
        "Starling F",
    )
    rows, _ = build_combined_evidence_rows(chembl, starling)
    index = build_neighbor_index(rows)

    result = retrieve_neighbors("CCO", index, top_k_per_group=3, min_similarity=0.0)

    assert len(result["groups"]) == 1
    assert len(result["groups"][0]["neighbors"]) == 1
    evidence = result["groups"][0]["neighbors"][0]["evidence_rows"]
    assert {row["evidence_source"] for row in evidence} == {
        "ChEMBL",
        "starling-labs/Oral_Bioavailability",
    }
    assert _group_evidence_source(result["groups"][0]) == (
        "ChEMBL + starling-labs/Oral_Bioavailability"
    )


def _index(molecule_id: str, smiles: str, inchi_key: str, group_id: str, description: str) -> dict:
    tier, endpoint_group = group_id.split(".", 1)
    return {
        "molecules": [
            {
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": smiles,
                "standard_inchi_key": inchi_key,
            }
        ],
        "group_to_molecule_indices": {group_id: [0]},
        "evidence_by_molecule_group": {
            molecule_id: {
                group_id: [
                    {
                        "molecule_chembl_id": molecule_id,
                        "canonical_smiles": smiles,
                        "assay_tier": tier,
                        "endpoint_group": endpoint_group,
                        "group_id": group_id,
                        "standard_type": "Oral bioavailability",
                        "standard_value": 50,
                        "standard_units": "%",
                        "assay_description": description,
                    }
                ]
            }
        },
    }
