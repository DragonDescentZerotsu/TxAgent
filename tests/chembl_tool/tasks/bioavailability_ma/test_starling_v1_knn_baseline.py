from rdkit import Chem

from tools.chembl_tool.tasks.bioavailability_ma.build_evidence_library import build_neighbor_index
from tools.chembl_tool.tasks.bioavailability_ma.run_starling_v1_knn_baseline import (
    GROUP_ID,
    numeric_only_index,
    predict_record,
)


def test_knn3_majority_uses_numeric_neighbors_without_similarity_cutoff():
    rows = [
        _row("exact", "CCO", 90.0),
        _row("high_1", "CCCO", 80.0),
        _row("high_2", "CCCCO", 60.0),
        _row("low_1", "c1ccccc1", 10.0),
        _qualitative_row("qualitative", "CCN"),
    ]
    index = build_neighbor_index(rows)
    numeric_index = numeric_only_index(index)

    prediction = predict_record(
        query_index=0,
        record={"drug": "CCO", "Y": 1},
        index=numeric_index,
    )

    assert prediction["prediction"] == "high"
    assert prediction["high_votes"] == 2
    assert len(prediction["neighbors"]) == 3
    assert all(neighbor["molecule_id"] != "exact" for neighbor in prediction["neighbors"])
    assert all(neighbor["molecule_id"] != "qualitative" for neighbor in prediction["neighbors"])
    assert prediction["neighbors"][-1]["similarity"] < 0.3


def _row(molecule_id: str, smiles: str, median: float) -> dict:
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True),
        "group_id": GROUP_ID,
        "source_value_median_percent": median,
        "standard_value": median,
    }


def _qualitative_row(molecule_id: str, smiles: str) -> dict:
    return {
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True),
        "group_id": GROUP_ID,
        "source_value_median_percent": "",
        "standard_value": "",
    }
