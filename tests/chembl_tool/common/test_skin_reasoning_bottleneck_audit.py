import json
import pytest

from tools.chembl_tool.paper_experiments.audit_skin_reasoning_bottleneck import (
    _paired_reference_summary,
    _prediction_map,
    classify_aligned_signals,
    classify_error,
)
from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    LEGACY_SKIN_REACTION_V1,
    SENSITIZATION_ALIGNED_V2,
)


def _group(group_id, direction, *, useful=True, transferability="moderate", confidence="moderate"):
    return {
        "group_id": group_id,
        "status": "ok",
        "llm": {
            "content": {
                "useful_for_skin_reaction_reasoning": useful,
                "transferability": transferability,
                "confidence": confidence,
                "evidence_direction": direction,
            }
        },
    }


def test_strict_signal_gate_uses_only_confident_transferable_tier_1_and_2():
    result = classify_aligned_signals(
        [
            _group("Mechanism.tier_1", "sensitization_risk"),
            _group("Mechanism.tier_2", "argues_against_skin_reaction_risk", confidence="low"),
            _group("Mechanism.tier_3", "supports_skin_reaction_risk"),
        ]
    )

    assert result == {
        "state": "positive_only",
        "positive_group_ids": ["Mechanism.tier_1"],
        "negative_group_ids": [],
    }


def test_strict_signal_gate_accepts_aligned_v2_field_names():
    group = _group("Mechanism.tier_1", "unused")
    content = group["llm"]["content"]
    content.pop("useful_for_skin_reaction_reasoning")
    content.pop("evidence_direction")
    content["useful_for_skin_sensitization_reasoning"] = True
    content["sensitization_evidence_direction"] = "supports_sensitizer"

    assert classify_aligned_signals([group])["state"] == "positive_only"


def test_error_categories_separate_final_from_upstream_failures():
    assert classify_error(1, "positive_only") == "final_recoverable"
    assert classify_error(0, "negative_only") == "final_recoverable"
    assert classify_error(1, "negative_only") == "upstream_wrong_direction"
    assert classify_error(0, "positive_only") == "upstream_wrong_direction"
    assert classify_error(1, "both") == "upstream_conflict"
    assert classify_error(0, "none") == "upstream_insufficient"


def test_paired_reference_summary_aligns_same_cohort(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "manifest.json").write_text(
        json.dumps({"task_prompt_profile": LEGACY_SKIN_REACTION_V1}),
        encoding="utf-8",
    )
    (reference / "predictions.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"query_index": 0, "label": 0, "pred_label": 0},
                {"query_index": 1, "label": 1, "pred_label": 0},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    source = [
        {"query_index": 1, "label": 1, "pred_label": 1},
        {"query_index": 0, "label": 0, "pred_label": 1},
    ]

    result = _paired_reference_summary(
        source,
        SENSITIZATION_ALIGNED_V2,
        reference,
    )

    assert result["prediction_flips"] == 2
    assert result["left_only_correct"] == 1
    assert result["right_only_correct"] == 1
    assert result["reference_profile"] == LEGACY_SKIN_REACTION_V1


def test_prediction_map_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="Duplicate source query_index"):
        _prediction_map([{"query_index": 1}, {"query_index": 1}], "source")
