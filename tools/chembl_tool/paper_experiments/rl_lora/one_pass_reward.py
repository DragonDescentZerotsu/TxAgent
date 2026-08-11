"""Deterministic hierarchical reward for one-pass full-flat RL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .experiment_contract import ONE_PASS_REWARD_VERSION, ONE_PASS_REWARD_WEIGHTS
from .one_pass_contract import nested_required_fields, prediction_to_label
from .reward import parse_json_object


REWARD_VERSION = ONE_PASS_REWARD_VERSION


@dataclass(frozen=True)
class OnePassRewardResult:
    reward: float
    final_correct: bool
    single_correct: bool
    analog_correct: bool
    branches_disagree: bool
    conflict_resolved_correctly: bool
    parsed_json: bool
    full_schema: bool
    predictions: dict[str, int | None]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_one_pass_response(
    response: str,
    metadata: Mapping[str, Any],
) -> OnePassRewardResult:
    """Make final correctness dominant and reward correct conflict resolution."""

    gold = int(metadata["gold_label"])
    if gold not in {0, 1}:
        raise ValueError(f"gold_label must be binary, got {gold}")
    fields = {
        "single": str(metadata["single_prediction_field"]),
        "analog": str(metadata["analog_prediction_field"]),
        "final": str(metadata["prediction_field"]),
    }
    content = parse_json_object(response)
    predictions = {
        name: prediction_to_label(content.get(field), metadata) if content else None
        for name, field in fields.items()
    }
    single_correct = predictions["single"] == gold
    analog_correct = predictions["analog"] == gold
    final_correct = predictions["final"] == gold
    branches_disagree = (
        predictions["single"] is not None
        and predictions["analog"] is not None
        and predictions["single"] != predictions["analog"]
    )
    required_fields = tuple(str(field) for field in metadata.get("required_fields", ()))
    top_level_complete = bool(content) and all(
        field in content and content[field] not in (None, "")
        for field in required_fields
    )
    nested_complete = bool(content) and all(
        isinstance(content.get(field), dict)
        and all(
            key in content[field] and content[field][key] not in (None, "")
            for key in nested_fields
        )
        for field, nested_fields in nested_required_fields(metadata).items()
    )
    full_schema = top_level_complete and nested_complete

    # Final correctness has a two-point margin.  The maximum possible reward for
    # a wrong final answer (-0.65) stays below the minimum for a correct one
    # (+1.00), while each independently correct branch adds useful signal.
    # Correctly resolving a branch conflict receives one extra branch-sized
    # bonus: one-correct/one-wrong/final-correct therefore ties, rather than
    # under-scores, both-correct/final-correct.  This is not a generic
    # consistency reward and never fires when the final answer is wrong.
    weights = ONE_PASS_REWARD_WEIGHTS
    reward = weights.final_correct if final_correct else weights.final_wrong
    reward += weights.single_correct if single_correct else 0.0
    reward += weights.analog_correct if analog_correct else 0.0
    reward += weights.resolved_conflict if branches_disagree and final_correct else 0.0
    reward += weights.parsed_json if content is not None else 0.0
    reward += weights.full_schema if full_schema else 0.0
    return OnePassRewardResult(
        reward=round(reward, 2),
        final_correct=final_correct,
        single_correct=single_correct,
        analog_correct=analog_correct,
        branches_disagree=branches_disagree,
        conflict_resolved_correctly=branches_disagree and final_correct,
        parsed_json=content is not None,
        full_schema=full_schema,
        predictions=predictions,
    )
