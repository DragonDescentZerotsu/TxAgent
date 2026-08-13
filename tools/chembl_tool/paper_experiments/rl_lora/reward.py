"""Pure, deterministic reward contract for Starling binary final decisions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardResult:
    reward: float
    predicted_label: int | None
    parsed_json: bool
    full_schema: bool
    correct: bool
    prediction_value: str | None


def parse_json_object(response: str) -> dict[str, Any] | None:
    """Parse a response as one JSON object, tolerating surrounding prose/fences."""
    text = str(response or "").strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and value:
            return value
    return None


def score_response(response: str, metadata: Mapping[str, Any]) -> RewardResult:
    """Score correctness first, with small bounded JSON/schema bonuses."""
    gold_label = int(metadata["gold_label"])
    if gold_label not in {0, 1}:
        raise ValueError(f"gold_label must be binary, got {gold_label}")

    prediction_field = str(metadata["prediction_field"])
    label_values = {
        str(metadata["negative_value"]): 0,
        str(metadata["positive_value"]): 1,
    }
    required_fields = tuple(str(field) for field in metadata.get("required_fields", ()))

    content = parse_json_object(response)
    parsed_json = content is not None
    prediction_value = (
        str(content.get(prediction_field))
        if content is not None and content.get(prediction_field) is not None
        else None
    )
    predicted_label = label_values.get(prediction_value or "")
    correct = predicted_label == gold_label
    full_schema = bool(content) and all(
        field in content and content[field] not in (None, "")
        for field in required_fields
    )

    reward = 1.0 if correct else -1.0
    if parsed_json:
        reward += 0.05
    if full_schema:
        reward += 0.05
    reward = round(reward, 2)
    return RewardResult(
        reward=reward,
        predicted_label=predicted_label,
        parsed_json=parsed_json,
        full_schema=full_schema,
        correct=correct,
        prediction_value=prediction_value,
    )
