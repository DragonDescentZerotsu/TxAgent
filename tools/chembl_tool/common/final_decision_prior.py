"""Shared final-stage decision contracts for binary reasoning tasks.

The default profile is a strict no-op.  The train-ratio profile exposes one
frozen training-split prior and requires the final model to use it only for a
genuine evidence tie.  It never changes retrieval, branch reasoning, or an
already directional evidence decision. Task-specific profiles live with their
task prompts and delegate shared profiles back to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


STANDARD_FINAL_DECISION = "standard"
TRAIN_RATIO_TIEBREAK_V1 = "train_ratio_tiebreak_v1"
GENERAL_FINAL_DECISION_PROFILES = (
    STANDARD_FINAL_DECISION,
    TRAIN_RATIO_TIEBREAK_V1,
)
EVIDENCE_STATES = (
    "consistent_positive",
    "consistent_negative",
    "mixed",
    "insufficient",
)


@dataclass(frozen=True)
class TrainRatioPrior:
    """Frozen train-split prior and task label vocabulary."""

    dataset_lineage: str
    split: str
    positive_count: int
    negative_count: int
    positive_label: str
    negative_label: str

    def __post_init__(self) -> None:
        if self.positive_count < 1 or self.negative_count < 1:
            raise ValueError("Train-ratio prior requires both binary classes")
        if self.positive_count <= self.negative_count:
            raise ValueError(
                "train_ratio_tiebreak_v1 expects the positive class to be the majority"
            )
        if not self.positive_label or not self.negative_label:
            raise ValueError("Train-ratio prior labels must be non-empty")

    @property
    def total(self) -> int:
        return self.positive_count + self.negative_count

    @property
    def positive_fraction(self) -> float:
        return self.positive_count / self.total


@dataclass(frozen=True)
class FinalDecisionPrompt:
    """Additional final-prompt fields and validation for one profile."""

    fields: dict[str, Any]
    instructions: tuple[str, ...]
    schema: dict[str, str]
    required_fields: tuple[str, ...]


def add_final_decision_profile_argument(
    parser: Any,
    *,
    choices: tuple[str, ...] = GENERAL_FINAL_DECISION_PROFILES,
) -> None:
    parser.add_argument(
        "--final-decision-profile",
        choices=choices,
        default=STANDARD_FINAL_DECISION,
        help=(
            "Versioned final-stage decision contract. The default is a strict "
            "no-op; every non-standard profile is an explicit final-only ablation."
        ),
    )


def build_final_decision_prompt(
    profile: str,
    prior: TrainRatioPrior,
) -> FinalDecisionPrompt:
    """Return deterministic additions without altering the standard prompt."""
    _validate_profile(profile)
    if profile == STANDARD_FINAL_DECISION:
        return FinalDecisionPrompt({}, (), {}, ())

    positive = prior.positive_label
    negative = prior.negative_label
    return FinalDecisionPrompt(
        fields={
            "training_label_prior": {
                "dataset_lineage": prior.dataset_lineage,
                "split": prior.split,
                "negative_label": negative,
                "negative_count": prior.negative_count,
                "positive_label": positive,
                "positive_count": prior.positive_count,
                "total": prior.total,
                "positive_fraction": round(prior.positive_fraction, 6),
                "policy": "genuine_evidence_tie_only",
            }
        },
        instructions=(
            "Before choosing the final label, classify the integrated transferable evidence into exactly one evidence_state.",
            f"Use evidence_state='consistent_positive' only when the usable evidence clearly favors '{positive}', and evidence_state='consistent_negative' only when it clearly favors '{negative}'.",
            "Use evidence_state='mixed' when meaningful usable evidence supports both directions; use evidence_state='insufficient' when usable evidence is too sparse or non-directional.",
            "A mixed or insufficient state is not automatically a tie. First decide whether the available evidence still favors one class. If it does, follow that evidence and set prior_used=false.",
            f"Only when evidence_state is mixed or insufficient AND neither class is better supported, you MUST break the genuine tie using the frozen training prior: choose '{positive}' and set prior_used=true.",
            "When evidence_state is consistent_positive or consistent_negative, the evidence determines the prediction and prior_used MUST be false; the prior may not override it.",
            "The training prior is a per-sample tie-break rule, not a target batch quota. Do not force the aggregate predictions to match the training ratio.",
        ),
        schema={
            "evidence_state": "consistent_positive | consistent_negative | mixed | insufficient",
            "prior_used": "boolean",
        },
        required_fields=("evidence_state", "prior_used"),
    )


def final_decision_validation_errors(
    content: Mapping[str, Any],
    *,
    profile: str,
    prior: TrainRatioPrior,
    prediction_field: str,
) -> list[str]:
    """Validate the executable cross-field tie-break contract."""
    _validate_profile(profile)
    if profile == STANDARD_FINAL_DECISION:
        return []

    errors: list[str] = []
    state = content.get("evidence_state")
    prediction = content.get(prediction_field)
    prior_used = content.get("prior_used")
    if state not in EVIDENCE_STATES:
        errors.append("invalid_value:evidence_state")
    if not isinstance(prior_used, bool):
        errors.append("invalid_type:prior_used:expected_boolean")
        return errors
    if state == "consistent_positive":
        if prediction != prior.positive_label:
            errors.append("inconsistent_decision:consistent_positive")
        if prior_used:
            errors.append("invalid_prior_use:consistent_positive")
    elif state == "consistent_negative":
        if prediction != prior.negative_label:
            errors.append("inconsistent_decision:consistent_negative")
        if prior_used:
            errors.append("invalid_prior_use:consistent_negative")
    elif prior_used and prediction != prior.positive_label:
        errors.append("inconsistent_prior_tiebreak:majority_label_required")
    return errors


def final_decision_allowed_values(profile: str) -> dict[str, set[str]]:
    """Return profile-owned enum validation without task-specific labels."""
    _validate_profile(profile)
    if profile == STANDARD_FINAL_DECISION:
        return {}
    return {"evidence_state": set(EVIDENCE_STATES)}


def _validate_profile(profile: str) -> None:
    if profile not in GENERAL_FINAL_DECISION_PROFILES:
        raise ValueError(f"Unknown final decision profile: {profile}")
