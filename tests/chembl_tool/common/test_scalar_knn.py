from tools.chembl_tool.common.scalar_knn import _majority_vote, _metrics, _neighbor_vote


def test_neighbor_vote_uses_median_numeric_measurement():
    vote = _neighbor_vote(
        {
            "molecule_chembl_id": "M1",
            "similarity": 0.8,
            "evidence_rows": [
                {"standard_value": 10},
                {"standard_value": 30},
                {"standard_value": "qualitative"},
            ],
        },
        threshold=20,
    )

    assert vote == {"molecule_id": "M1", "similarity": 0.8, "value": 20.0, "label": 1}


def test_majority_vote_and_metrics_are_deterministic():
    assert _majority_vote([{"label": 1, "similarity": 0.8}, {"label": 0, "similarity": 0.5}], fallback_label=0) == 1
    assert _majority_vote([], fallback_label=0) == 0
    metrics = _metrics(
        [
            {"label": 0, "predicted_label": 0},
            {"label": 0, "predicted_label": 1},
            {"label": 1, "predicted_label": 1},
        ]
    )
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 1}
