from pathlib import Path

import pytest

from tools.chembl_tool.paper_experiments.audit_skin_direct_scope_retrieval import (
    _prediction_map,
    card_agreement_state,
    summarize,
)


def test_card_agreement_state_uses_frozen_record_agreement_threshold():
    assert card_agreement_state({"positive": 7, "negative": 3}) == "consistent_positive"
    assert card_agreement_state({"positive": 3, "negative": 7}) == "consistent_negative"
    assert card_agreement_state({"positive": 6, "negative": 4}) == "mixed"
    assert card_agreement_state({"inconclusive": 2}) == "insufficient"


def test_scope_retrieval_summary_counts_query_and_neighbor_changes():
    rows = [
        {
            "Y": 0,
            "neighbor_identity_changed": True,
            "direct_context_changed": True,
            "old_neighbor_smiles": ["CCO"],
            "new_neighbor_smiles": ["CCN"],
        },
        {
            "Y": 1,
            "neighbor_identity_changed": False,
            "direct_context_changed": True,
            "old_neighbor_smiles": ["CCC"],
            "new_neighbor_smiles": ["CCC"],
        },
    ]

    summary = summarize(rows)

    assert summary["n_queries"] == 2
    assert summary["neighbor_identity_changed"] == 1
    assert summary["direct_context_changed"] == 2
    assert summary["n_old_neighbors_absent_from_new_retrieval"] == 1
    assert summary["n_new_backfill_neighbors"] == 1
    assert summary["by_label"]["0"]["neighbor_identity_changed"] == 1


def test_prediction_map_rejects_duplicate_query_indices():
    with pytest.raises(ValueError, match="Duplicate query index"):
        _prediction_map(
            [{"query_index": 1}, {"query_index": 1}],
            Path("predictions.jsonl"),
        )
