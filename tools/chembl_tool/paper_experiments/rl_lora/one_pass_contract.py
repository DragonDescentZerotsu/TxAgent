"""Dependency-light one-pass identifiers and binary label mapping.

Ray environment actors load this module in a narrow worker environment.  Keep it
free of project-level data builders and hashing utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


CONTRACT_VERSION = "one_pass_full_flat.v1"
VISIBLE_PREFETCHED_CONTRACT_VERSION = "one_pass_full_flat_visible_prefetched.v1"
ONE_PASS_CONTRACT_VERSIONS = frozenset(
    {CONTRACT_VERSION, VISIBLE_PREFETCHED_CONTRACT_VERSION}
)
IDENTITY_BLIND = "identity_blind"
DEPLOYMENT_VISIBLE_PREFETCHED = "deployment_visible_prefetched"
VISIBILITY_MODES = frozenset({IDENTITY_BLIND, DEPLOYMENT_VISIBLE_PREFETCHED})
SINGLE_PREDICTION_FIELD = "single_prediction"
ANALOG_PREDICTION_FIELD = "analog_prediction"


@dataclass(frozen=True)
class OnePassTaskContract:
    prediction_field: str
    negative_value: str
    positive_value: str

    @property
    def prediction_schema(self) -> str:
        return f"{self.positive_value} | {self.negative_value}"


TASK_CONTRACTS = {
    "bbb_martins": OnePassTaskContract("bbb_prediction", "fail", "pass"),
    "bioavailability_ma": OnePassTaskContract(
        "bioavailability_prediction", "low", "high"
    ),
    "skin_reaction": OnePassTaskContract("skin_reaction_prediction", "no_risk", "risk"),
}


def contract_version_for_visibility(visibility_mode: str) -> str:
    if visibility_mode == IDENTITY_BLIND:
        return CONTRACT_VERSION
    if visibility_mode == DEPLOYMENT_VISIBLE_PREFETCHED:
        return VISIBLE_PREFETCHED_CONTRACT_VERSION
    raise ValueError(f"unsupported one-pass visibility mode: {visibility_mode}")


def is_one_pass_contract(value: Any) -> bool:
    return str(value) in ONE_PASS_CONTRACT_VERSIONS


def prediction_to_label(value: Any, metadata: Mapping[str, Any]) -> int | None:
    mapping = {
        str(metadata["negative_value"]): 0,
        str(metadata["positive_value"]): 1,
    }
    return mapping.get(str(value))


def nested_required_fields(
    metadata: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Recover deterministic nested analysis keys from the frozen prompt."""

    for message in reversed(list(metadata.get("messages") or [])):
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        schema = (
            payload.get("required_json_schema") if isinstance(payload, dict) else None
        )
        if not isinstance(schema, dict):
            continue
        return {
            field: tuple(str(key) for key in nested)
            for field in ("single_analysis", "analog_analysis")
            if isinstance((nested := schema.get(field)), dict)
        }
    return {}
