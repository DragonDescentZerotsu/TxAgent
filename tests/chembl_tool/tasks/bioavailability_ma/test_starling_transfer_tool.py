from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import _clean_evidence_row
from tools.chembl_tool.tasks.bioavailability_ma.starling_transfer_tool import (
    StarlingTransferConfig,
    annotate_retrieval_with_starling_transfer,
    select_top_transfer_neighbors,
)


def test_annotates_starling_numeric_examples(monkeypatch):
    retrieval = {
        "query": {"canonical_smiles": "CCN"},
        "groups": [
            {
                "group_id": "Starling.direct_oral_bioavailability",
                "neighbors": [
                    {
                        "canonical_smiles": "CCO",
                        "evidence_rows": [
                            {
                                "assay_chembl_id": "STARLING_ORAL_BIOAVAILABILITY",
                                "evidence_source": "starling-labs/Oral_Bioavailability",
                                "source_record_examples": [
                                    {
                                        "source_index": 11,
                                        "molecule_name": "source molecule",
                                        "oral_bioavailability_value_percent": 75.0,
                                        "species_or_population": "human",
                                        "dose": "10 mg",
                                        "oral_exposure_mode": "tablet",
                                    },
                                    {
                                        "source_index": 12,
                                        "molecule_name": "source molecule",
                                        "oral_bioavailability_value_percent": 8.0,
                                        "species_or_population": "rat",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    seen_tasks = []

    def fake_score(tasks, config):
        seen_tasks.extend(tasks)
        return [0.91, 0.13]

    monkeypatch.setattr(
        "tools.chembl_tool.tasks.bioavailability_ma.starling_transfer_tool._score_tasks",
        fake_score,
    )

    annotated, summary = annotate_retrieval_with_starling_transfer(
        retrieval,
        config=StarlingTransferConfig(model_name_or_path="fake", query_metadata_mode="same_source_context"),
    )

    assert summary["n_pairs_scored"] == 2
    assert seen_tasks[0]["metadata_a"]["molecule_name"] == "source molecule"
    assert seen_tasks[0]["metadata_b"] == {
        "species_or_population": "human",
        "dose": "10 mg",
        "oral_exposure_mode": "tablet",
    }
    annotation = annotated["groups"][0]["neighbors"][0]["evidence_rows"][0]["starling_transfer_tool"]
    assert annotation["likely_transfer_count"] == 1
    assert annotation["unlikely_transfer_count"] == 1
    assert annotation["transfer_probability_median"] == 0.52
    assert annotation["source_example_scores"][0]["transfer_prediction"] == "likely_transfer"
    clean_row = _clean_evidence_row(annotated["groups"][0]["neighbors"][0]["evidence_rows"][0])
    assert clean_row["starling_transfer_tool"]["transfer_probability_max"] == 0.91


def test_select_top_transfer_neighbors_ranks_by_transfer_score():
    retrieval = {
        "coverage": {"n_neighbors_total": 4, "top_k_per_group": 10},
        "groups": [
            {
                "group_id": "Starling.direct_oral_bioavailability",
                "neighbors": [
                    _neighbor("n1", rank=1, similarity=0.9, transfer_max=0.12, transfer_mean=0.12),
                    _neighbor("n2", rank=2, similarity=0.8, transfer_max=0.95, transfer_mean=0.4),
                    _neighbor("n3", rank=3, similarity=0.7, transfer_max=0.95, transfer_mean=0.8),
                    _neighbor("n4", rank=4, similarity=0.99, transfer_max=None, transfer_mean=None),
                ],
            }
        ],
    }

    selected, summary = select_top_transfer_neighbors(retrieval, top_k=2)

    assert summary["candidate_neighbors_total"] == 4
    assert summary["scored_candidate_neighbors_total"] == 3
    assert summary["selected_neighbors_total"] == 2
    neighbors = selected["groups"][0]["neighbors"]
    assert [neighbor["molecule_chembl_id"] for neighbor in neighbors] == ["n3", "n2"]
    assert [neighbor["transfer_selection_rank"] for neighbor in neighbors] == [1, 2]
    assert [neighbor["structural_rank"] for neighbor in neighbors] == [3, 2]
    assert selected["coverage"]["pre_transfer_selection_n_neighbors_total"] == 4
    assert selected["coverage"]["n_neighbors_total"] == 2
    assert selected["coverage"]["transfer_selection_top_k_per_group"] == 2


def _neighbor(
    molecule_id: str,
    *,
    rank: int,
    similarity: float,
    transfer_max: float | None,
    transfer_mean: float | None,
) -> dict:
    annotation = {}
    if transfer_max is not None:
        annotation = {
            "transfer_probability_max": transfer_max,
            "transfer_probability_mean": transfer_mean,
            "transfer_probability_median": transfer_mean,
            "likely_transfer_count": 1,
            "n_scored_source_examples": 1,
        }
    return {
        "rank": rank,
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": "CCO",
        "similarity": similarity,
        "evidence_rows": [{"starling_transfer_tool": annotation} if annotation else {}],
    }
