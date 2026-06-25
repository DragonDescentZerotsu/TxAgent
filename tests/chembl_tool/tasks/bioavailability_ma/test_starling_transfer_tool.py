from tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline import _clean_evidence_row
from tools.chembl_tool.tasks.bioavailability_ma.starling_transfer_tool import (
    StarlingTransferConfig,
    annotate_retrieval_with_starling_transfer,
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
