"""BBB-only final-stage evidence-integration diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

from tools.chembl_tool.common.final_decision_prior import (
    GENERAL_FINAL_DECISION_PROFILES,
    FinalDecisionPrompt,
    TrainRatioPrior,
    build_final_decision_prompt,
    final_decision_allowed_values,
    final_decision_validation_errors,
)


DIRECT_ANCHORED_RESIDUAL_V1 = "direct_anchored_residual_v1"
DIRECT_OVERRIDE_RECHECK_V1 = "direct_override_recheck_v1"
BBB_FINAL_DECISION_PROFILES = (
    *GENERAL_FINAL_DECISION_PROFILES,
    DIRECT_ANCHORED_RESIDUAL_V1,
    DIRECT_OVERRIDE_RECHECK_V1,
)
DIRECT_ANCHOR_STATES = (
    "supports_positive",
    "supports_negative",
    "mixed",
    "insufficient",
)
MECHANISM_OVERRIDES = (
    "no_change",
    "override_to_positive",
    "override_to_negative",
)
MECHANISM_EVIDENCE_WEIGHTS = (
    "decisive",
    "supporting",
    "context_only",
    "none",
)
OVERRIDE_VERDICTS = ("uphold", "reject")


def build_bbb_final_decision_prompt(
    profile: str,
    prior: TrainRatioPrior,
) -> FinalDecisionPrompt:
    if profile not in {
        DIRECT_ANCHORED_RESIDUAL_V1,
        DIRECT_OVERRIDE_RECHECK_V1,
    }:
        return build_final_decision_prompt(profile, prior)

    positive = prior.positive_label
    negative = prior.negative_label
    instructions = [
        "First infer a direct_anchor_state and direct_anchor_prediction from the same evidence surface as the Direct condition: the single-molecule assessment plus the direct observed-outcome group. Do not mix passive, efflux, influx, or other mechanism-only groups into this anchor.",
        f"Map Direct-condition evidence favoring '{positive}' to direct_anchor_state='supports_positive' and evidence favoring '{negative}' to direct_anchor_state='supports_negative'; use mixed or insufficient when that evidence surface does not support one direction.",
        "Treat the remaining groups as mechanism modifiers, not equal label votes. Repeated or correlated mechanism records do not become independent corroboration by count.",
        "Use mechanism_evidence_weight='decisive' only when an experimentally supported mechanism is query-specific, structurally transferable, endpoint-role compatible, and strong enough to change the direct outcome expectation.",
        "Similarity, a passive-property prior, inhibition without demonstrated substrate transport, or a generic transporter association may be supporting/context evidence but cannot be decisive by itself.",
        "Set mechanism_override='no_change' unless decisive mechanism evidence justifies changing the direct_anchor_prediction. For an override, name the decisive mechanism evidence and the competing direct evidence explicitly.",
        "When mechanism_override='no_change', preserve direct_anchor_prediction. An override_to_positive or override_to_negative must change the anchor in the named direction and must set override_supported=true.",
    ]
    schema: dict[str, Any] = {
        "direct_anchor_state": "supports_positive | supports_negative | mixed | insufficient",
        "direct_anchor_prediction": f"{positive} | {negative}",
        "mechanism_evidence_weight": "decisive | supporting | context_only | none",
        "mechanism_override": "no_change | override_to_positive | override_to_negative",
        "override_supported": "boolean",
        "decisive_mechanism_evidence": ["string"],
        "competing_direct_evidence": ["string"],
        "anchor_reasoning": "string",
    }
    required = (
        "direct_anchor_state",
        "direct_anchor_prediction",
        "mechanism_evidence_weight",
        "mechanism_override",
        "override_supported",
        "decisive_mechanism_evidence",
        "competing_direct_evidence",
        "anchor_reasoning",
    )
    if profile == DIRECT_OVERRIDE_RECHECK_V1:
        instructions.extend(
            [
                "This is an independent recheck applied only to cases where an earlier residual adjudication proposed changing the direct anchor.",
                "Rebuild the direct anchor and audit the claimed override from the supplied branch summaries. Set override_verdict='uphold' only if decisive transferable mechanism evidence still supports the change; otherwise set it to 'reject', use mechanism_override='no_change', and preserve the anchor.",
            ]
        )
        schema["override_verdict"] = "uphold | reject"
        schema["recheck_reason"] = "string"
        required = (*required, "override_verdict", "recheck_reason")
    return FinalDecisionPrompt({}, tuple(instructions), schema, required)


def bbb_final_decision_allowed_values(profile: str) -> dict[str, set[str]]:
    if profile not in {
        DIRECT_ANCHORED_RESIDUAL_V1,
        DIRECT_OVERRIDE_RECHECK_V1,
    }:
        return final_decision_allowed_values(profile)
    values = {
        "direct_anchor_state": set(DIRECT_ANCHOR_STATES),
        "mechanism_override": set(MECHANISM_OVERRIDES),
        "mechanism_evidence_weight": set(MECHANISM_EVIDENCE_WEIGHTS),
    }
    if profile == DIRECT_OVERRIDE_RECHECK_V1:
        values["override_verdict"] = set(OVERRIDE_VERDICTS)
    return values


def bbb_final_decision_validation_errors(
    content: Mapping[str, Any],
    *,
    profile: str,
    prior: TrainRatioPrior,
    prediction_field: str,
) -> list[str]:
    if profile not in {
        DIRECT_ANCHORED_RESIDUAL_V1,
        DIRECT_OVERRIDE_RECHECK_V1,
    }:
        return final_decision_validation_errors(
            content,
            profile=profile,
            prior=prior,
            prediction_field=prediction_field,
        )

    positive = prior.positive_label
    negative = prior.negative_label
    anchor_state = content.get("direct_anchor_state")
    anchor = content.get("direct_anchor_prediction")
    override = content.get("mechanism_override")
    weight = content.get("mechanism_evidence_weight")
    supported = content.get("override_supported")
    prediction = content.get(prediction_field)
    errors: list[str] = []

    if anchor_state not in DIRECT_ANCHOR_STATES:
        errors.append("invalid_value:direct_anchor_state")
    if anchor not in {positive, negative}:
        errors.append("invalid_value:direct_anchor_prediction")
    if override not in MECHANISM_OVERRIDES:
        errors.append("invalid_value:mechanism_override")
    if weight not in MECHANISM_EVIDENCE_WEIGHTS:
        errors.append("invalid_value:mechanism_evidence_weight")
    if not isinstance(supported, bool):
        errors.append("invalid_type:override_supported:expected_boolean")
        return errors
    if anchor_state == "supports_positive" and anchor != positive:
        errors.append("inconsistent_anchor:supports_positive")
    if anchor_state == "supports_negative" and anchor != negative:
        errors.append("inconsistent_anchor:supports_negative")

    if override == "no_change":
        if prediction != anchor:
            errors.append("inconsistent_decision:no_change")
        if supported:
            errors.append("invalid_override_support:no_change")
    elif override == "override_to_positive":
        if anchor != negative or prediction != positive:
            errors.append("inconsistent_decision:override_to_positive")
        if not supported or weight != "decisive":
            errors.append("insufficient_override_support:override_to_positive")
    elif override == "override_to_negative":
        if anchor != positive or prediction != negative:
            errors.append("inconsistent_decision:override_to_negative")
        if not supported or weight != "decisive":
            errors.append("insufficient_override_support:override_to_negative")

    if profile == DIRECT_OVERRIDE_RECHECK_V1:
        verdict = content.get("override_verdict")
        if verdict not in OVERRIDE_VERDICTS:
            errors.append("invalid_value:override_verdict")
        elif verdict == "reject" and override != "no_change":
            errors.append("inconsistent_recheck:reject_requires_no_change")
        elif verdict == "uphold" and override == "no_change":
            errors.append("inconsistent_recheck:uphold_requires_override")
    return errors
