"""Shared paired statistics for aligned binary prediction vectors."""

from __future__ import annotations

from typing import Any, Sequence

from tools.chembl_tool.paper_experiments.summarize_results import (
    macro_f1,
    mcnemar_exact_p,
    paired_bootstrap_delta_ci,
)


def paired_binary_summary(
    labels: Sequence[int],
    left: Sequence[int],
    right: Sequence[int],
    *,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260809,
) -> dict[str, Any]:
    """Summarize right-minus-left changes for one aligned binary cohort."""
    if not labels or len(labels) != len(left) or len(labels) != len(right):
        raise ValueError("Paired binary vectors must be non-empty and equal length")
    label_values = [int(value) for value in labels]
    left_values = [int(value) for value in left]
    right_values = [int(value) for value in right]
    if any(value not in {0, 1} for value in label_values + left_values + right_values):
        raise ValueError("Paired binary vectors may contain only 0 and 1")

    left_only = sum(
        left_prediction == label and right_prediction != label
        for label, left_prediction, right_prediction in zip(
            label_values, left_values, right_values, strict=True
        )
    )
    right_only = sum(
        right_prediction == label and left_prediction != label
        for label, left_prediction, right_prediction in zip(
            label_values, left_values, right_values, strict=True
        )
    )
    ci_low, ci_high = paired_bootstrap_delta_ci(
        label_values,
        left_values,
        right_values,
        bootstrap_replicates,
        seed=seed,
    )
    left_f1 = macro_f1(label_values, left_values)
    right_f1 = macro_f1(label_values, right_values)
    return {
        "n": len(label_values),
        "left_macro_f1": left_f1,
        "right_macro_f1": right_f1,
        "delta_macro_f1": right_f1 - left_f1,
        "delta_macro_f1_bootstrap_95ci": [ci_low, ci_high],
        "left_accuracy": sum(
            prediction == label
            for prediction, label in zip(left_values, label_values, strict=True)
        )
        / len(label_values),
        "right_accuracy": sum(
            prediction == label
            for prediction, label in zip(right_values, label_values, strict=True)
        )
        / len(label_values),
        "prediction_flips": sum(
            left_prediction != right_prediction
            for left_prediction, right_prediction in zip(
                left_values, right_values, strict=True
            )
        ),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "mcnemar_exact_p": mcnemar_exact_p(left_only, right_only),
    }
