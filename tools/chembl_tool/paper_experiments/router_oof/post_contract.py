"""Artifact and feature contracts for the output-aware post-selector v3."""

from __future__ import annotations


POST_FEATURE_SCHEMA_VERSION = "router_post_selector_features.direct.v3"
POST_SELECTOR_SCHEMA_VERSION = "task_local_knn_agent_post_selector.v3"
POST_FEATURE_STEM = "post_selector_features_v3"
POST_ARTIFACT_DIR = "post_selector_v3"
POST_V31_SELECTOR_SCHEMA_VERSION = "task_local_knn_agent_post_selector.v3.1"
POST_V31_ARTIFACT_DIR = "post_selector_v31"
POST_V31_LEARNING_CURVE_SCHEMA_VERSION = (
    "task_local_knn_agent_post_selector.v3.1.learning_curve"
)
POST_V31_LEARNING_CURVE_ARTIFACT_DIR = "post_selector_v31_learning_curve"
POST_V31_TRANSFER_SCHEMA_VERSION = "task_local_knn_agent_post_selector.v3.1.transfer"
POST_V31_TRANSFER_ARTIFACT_DIR = "post_selector_v31_transfer"

POST_SELECTOR_DIRECTIONS = {
    "knn0_agent1": (0, 1),
    "knn1_agent0": (1, 0),
}

OUTPUT_COLUMNS = (
    "direction_knn0_agent1",
    "direction_knn1_agent0",
)

DECISION_EVIDENCE_COLUMNS = (
    "decision_group_count",
    "decision_useful_group_fraction",
    "decision_positive_weight",
    "decision_negative_weight",
    "decision_neutral_weight",
    "decision_signed_margin",
    "decision_conflict_fraction",
    "decision_high_transferability_fraction",
    "decision_high_confidence_fraction",
    "decision_supports_agent_weight",
    "decision_supports_knn_weight",
    "decision_agent_support_margin",
)

TRACE_COLUMNS = (
    "trace_final_confidence_high",
    "trace_final_confidence_moderate",
    "trace_final_confidence_low",
    "trace_single_confidence_high",
    "trace_single_confidence_moderate",
    "trace_single_confidence_low",
    "trace_group_confidence_mean",
    "trace_group_transferability_mean",
    "trace_single_supports_agent",
    "trace_single_supports_knn",
    "trace_single_neutral",
    "trace_single_final_agreement",
    "trace_group_supports_agent_fraction",
    "trace_group_supports_knn_fraction",
    "trace_group_neutral_fraction",
    "trace_group_final_agreement",
    "trace_final_conflicting_evidence_count_log1p",
    "trace_final_evidence_gap_count_log1p",
    "trace_final_caveat_count_log1p",
    "trace_final_reasoning_chars_log1p",
    "trace_final_summary_chars_log1p",
)
