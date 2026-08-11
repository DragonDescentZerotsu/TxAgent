import pytest

from tools.chembl_tool.paper_experiments.audit_starling_trace_failure_causes import (
    TASK_SPECS,
    _align_predictions_with_labels,
    _bioavailability_neighbor_signal,
    _direction_labels,
    _majority_vote,
    _paired_prediction_summary,
    _skin_neighbor_signal,
    _strict_signals,
    _validation_receipt,
)


def test_task_specs_have_unique_cli_names() -> None:
    names = [spec.run_task for spec in TASK_SPECS]
    assert len(names) == len(set(names))


def _spec(task: str):
    return next(spec for spec in TASK_SPECS if spec.dataset_task == task)


def test_direct_neighbor_signal_uses_frozen_task_label_contracts() -> None:
    assert _bioavailability_neighbor_signal({"source_value_median": 20}) == 1
    assert _bioavailability_neighbor_signal({"source_value_median": 19.9}) == 0
    assert _bioavailability_neighbor_signal({}) is None
    assert _skin_neighbor_signal({"source_endpoint_counts": {"positive": 3, "negative": 1}}) == 1
    assert _skin_neighbor_signal({"source_endpoint_counts": {"positive": 2, "negative": 2}}) is None


def test_majority_vote_rejects_ties_and_missing_values() -> None:
    assert _majority_vote([1, 1, 0]) == 1
    assert _majority_vote([1, 0, None]) is None
    assert _majority_vote([None, None]) is None


def test_strict_signal_gate_and_typo_normalization_are_separate() -> None:
    spec = _spec("Bioavailability_Ma")
    content = {
        spec.useful_fields[0]: True,
        "transferability": "moderate",
        "confidence": "moderate",
        "evidence_direction": "args_against_high_bioavailability",
    }
    assert _direction_labels(spec, content["evidence_direction"]) == {0}
    assert _strict_signals(spec, content) == {0}

    content["transferability"] = "low"
    assert _strict_signals(spec, content) == set()


def test_strict_signal_gate_accepts_aligned_skin_field_names() -> None:
    spec = _spec("Skin_Reaction")
    content = {
        "useful_for_skin_sensitization_reasoning": True,
        "transferability": "moderate",
        "confidence": "moderate",
        "sensitization_evidence_direction": "supports_sensitizer",
    }

    assert _strict_signals(spec, content) == {1}


def test_paired_summary_aligns_by_query_index() -> None:
    left = {
        1: {"label": 1, "pred_label": 0},
        0: {"label": 0, "pred_label": 0},
    }
    right = {
        0: {"label": 0, "pred_label": 1},
        1: {"label": 1, "pred_label": 1},
    }

    result = _paired_prediction_summary(left, right, bootstrap_replicates=20)

    assert result["prediction_flips"] == 2
    assert result["left_only_correct"] == 1
    assert result["right_only_correct"] == 1
    assert result["mcnemar_exact_p"] == 1.0


def test_validation_receipt_extracts_retry_provenance() -> None:
    output = {
        "llm": {
            "structured_output_validation": {
                "attempt_count": 2,
                "retried": True,
                "valid": True,
            }
        }
    }

    assert _validation_receipt(output) == {
        "attempt_count": 2,
        "retried": True,
        "valid": True,
    }


def test_gold_alignment_uses_query_index_and_checks_identity() -> None:
    predictions = [
        {"query_index": 1, "smiles": "B", "label": 1},
        {"query_index": 0, "smiles": "A", "label": 0},
    ]
    labels = [{"drug": "A", "Y": 0}, {"drug": "B", "Y": 1}]
    aligned = _align_predictions_with_labels(predictions, labels, "Task")
    assert [row[0]["query_index"] for row in aligned] == [0, 1]

    predictions[0]["smiles"] = "wrong"
    with pytest.raises(ValueError, match="molecule mismatch"):
        _align_predictions_with_labels(predictions, labels, "Task")
