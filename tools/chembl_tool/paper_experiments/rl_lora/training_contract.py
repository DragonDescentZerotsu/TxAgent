"""Single reward/metadata surface shared by every RL backend.

Backends may render and transport responses differently, but they must pass the
same normalized private metadata and response text through this dispatcher.
No provider SDK, Ray, Torch, or model implementation is imported here.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias, TypedDict

from .one_pass_contract import is_one_pass_contract
from .one_pass_reward import OnePassRewardResult, score_one_pass_response
from .reward import RewardResult, score_response


class StarlingRewardMetadata(TypedDict, total=False):
    gold_label: int
    prediction_field: str
    negative_value: str
    positive_value: str
    required_fields: list[str]
    source_task: str
    source_fold: int
    source_index: int
    contract_version: str
    messages: list[dict[str, str]]
    single_prediction_field: str
    analog_prediction_field: str


TrainingRewardResult: TypeAlias = RewardResult | OnePassRewardResult


def reward_metadata_from_row(row: Mapping[str, Any]) -> StarlingRewardMetadata:
    """Build the exact private reward view used by Tinker and NeMo."""

    metadata: StarlingRewardMetadata = {
        "gold_label": int(row["gold_label"]),
        "prediction_field": str(row["prediction_field"]),
        "negative_value": str(row["negative_value"]),
        "positive_value": str(row["positive_value"]),
        "required_fields": [str(field) for field in row["required_fields"]],
        "source_task": str(row["source_task"]),
        "source_fold": int(row.get("source_fold", -1)),
        "source_index": int(row["source_index"]),
    }
    if is_one_pass_contract(row.get("contract_version")):
        metadata.update(
            {
                "contract_version": str(row["contract_version"]),
                "messages": [
                    {
                        "role": str(message["role"]),
                        "content": str(message["content"]),
                    }
                    for message in row["messages"]
                ],
                "single_prediction_field": str(row["single_prediction_field"]),
                "analog_prediction_field": str(row["analog_prediction_field"]),
            }
        )
    return metadata


def score_training_response(
    response: str,
    metadata_or_row: Mapping[str, Any],
) -> TrainingRewardResult:
    """Dispatch one response through the frozen contract, independent of backend."""

    metadata = reward_metadata_from_row(metadata_or_row)
    if is_one_pass_contract(metadata.get("contract_version")):
        return score_one_pass_response(response, metadata)
    return score_response(response, metadata)


def training_result_metrics(
    result: TrainingRewardResult,
    *,
    clean_stop: bool | None = None,
) -> dict[str, float]:
    metrics = {
        "correct": float(training_result_final_correct(result)),
        "parsed_json": float(result.parsed_json),
        "full_schema": float(result.full_schema),
    }
    if isinstance(result, OnePassRewardResult):
        metrics.update(
            {
                "final_correct": float(result.final_correct),
                "single_correct": float(result.single_correct),
                "analog_correct": float(result.analog_correct),
                "branches_disagree": float(result.branches_disagree),
                "conflict_resolved_correctly": float(
                    result.conflict_resolved_correctly
                ),
            }
        )
    if clean_stop is not None:
        metrics["clean_stop"] = float(clean_stop)
    return metrics


def training_result_logs(
    result: TrainingRewardResult,
    metadata_or_row: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = reward_metadata_from_row(metadata_or_row)
    logs: dict[str, Any] = {
        "source_task": str(metadata["source_task"]),
        "source_index": int(metadata["source_index"]),
    }
    if isinstance(result, OnePassRewardResult):
        logs.update(
            {
                "single_prediction": result.predictions["single"],
                "analog_prediction": result.predictions["analog"],
                "final_prediction": result.predictions["final"],
            }
        )
    else:
        logs["prediction_value"] = result.prediction_value or ""
    return logs


def training_result_final_correct(result: TrainingRewardResult) -> bool:
    return (
        result.final_correct
        if isinstance(result, OnePassRewardResult)
        else result.correct
    )


def training_result_final_prediction(result: TrainingRewardResult) -> int | None:
    if isinstance(result, OnePassRewardResult):
        return result.predictions["final"]
    return result.predicted_label
