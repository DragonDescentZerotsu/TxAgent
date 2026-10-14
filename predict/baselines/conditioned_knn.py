"""Condition-aware neighbor selection, benchmark metadata and diagnostics."""

from __future__ import annotations

from typing import Any


NO_REPORTED_CONDITION = "no_reported_external_condition"
CONDITION_POLICIES = (
    "row_agnostic",
    "all_train_unique_molecules",
    "same_condition_then_null",
)


def select_conditioned_ranked_indices(
    ranked_indices: list[int],
    reference: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    k: int,
    policy: str,
) -> list[tuple[int, str]]:
    """Select unique-molecule neighbors from a pre-ranked reference list."""
    if policy not in CONDITION_POLICIES[1:]:
        raise ValueError(f"Unsupported conditioned KNN policy: {policy}")
    if k <= 0:
        raise ValueError("k must be positive")
    selected: list[tuple[int, str]] = []
    selected_keys: set[str] = set()
    if policy == "all_train_unique_molecules":
        representative = _canonical_reference_indices(reference)
        for index in ranked_indices:
            key = _molecule_key(reference[index])
            if representative[key] != index or key in selected_keys:
                continue
            selected.append((index, "all_train_unique_molecules"))
            selected_keys.add(key)
            if len(selected) == k:
                break
        return selected

    query_group = _condition_group(query)
    groups = [(query_group, "same_condition")]
    if query_group != NO_REPORTED_CONDITION:
        groups.append((NO_REPORTED_CONDITION, "null_fallback"))
    for group, source in groups:
        for index in ranked_indices:
            row = reference[index]
            key = _molecule_key(row)
            if _condition_group(row) != group or key in selected_keys:
                continue
            selected.append((index, source))
            selected_keys.add(key)
            if len(selected) == k:
                return selected
    return selected


def benchmark_row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return benchmark identity fields that must survive prediction output."""
    return {
        key: row[key]
        for key in (
            "condition_group",
            "condition_scope",
            "molecule_identity_key",
            "benchmark_row_id",
        )
        if key in row
    }


def condition_metrics(
    rows: list[dict[str, Any]], *, field: str = "condition_group"
) -> dict[str, Any]:
    """Report condition cohorts, retaining insufficient-neighbor coverage."""
    groups = sorted({str(row.get(field) or NO_REPORTED_CONDITION) for row in rows})
    return {
        "overall": _condition_metric_block(rows),
        "no_reported_external_condition": _condition_metric_block(
            [row for row in rows if str(row.get(field) or NO_REPORTED_CONDITION) == NO_REPORTED_CONDITION]
        ),
        "external_conditions": _condition_metric_block(
            [row for row in rows if str(row.get(field) or NO_REPORTED_CONDITION) != NO_REPORTED_CONDITION]
        ),
        "per_group": {
            group: _condition_metric_block(
                [row for row in rows if str(row.get(field) or NO_REPORTED_CONDITION) == group]
            )
            for group in groups
        },
    }


def _condition_metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    evaluated = [row for row in rows if row.get("prediction") is not None and row.get("status", "ok") == "ok"]
    metrics = {
        "n": len(evaluated),
        "n_total": len(rows),
        "n_skipped_insufficient_neighbors": len(rows) - len(evaluated),
        "evaluation_coverage": len(evaluated) / len(rows) if rows else 0.0,
    }
    if not evaluated:
        return metrics
    labels = [int(row["Y"]) for row in evaluated]
    predictions = [int(row["prediction"]) for row in evaluated]
    scores = [float(row["score"]) for row in evaluated]
    metrics.update({
        "n_y0": labels.count(0),
        "n_y1": labels.count(1),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)) if set(labels) == {0, 1} else None,
        "positive_precision": float(precision_score(labels, predictions, zero_division=0)),
        "positive_recall": float(recall_score(labels, predictions, zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions)),
            "fp": sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions)),
            "fn": sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions)),
            "tp": sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions)),
        },
    })
    return metrics


def _canonical_reference_indices(reference: list[dict[str, Any]]) -> dict[str, int]:
    by_molecule: dict[str, list[int]] = {}
    for index, row in enumerate(reference):
        by_molecule.setdefault(_molecule_key(row), []).append(index)
    return {
        key: min(
            indices,
            key=lambda index: (
                _condition_group(reference[index]) != NO_REPORTED_CONDITION,
                _condition_group(reference[index]),
                index,
            ),
        )
        for key, indices in by_molecule.items()
    }


def _molecule_key(row: dict[str, Any]) -> str:
    return str(row.get("molecule_identity_key") or row["drug"])


def _condition_group(row: dict[str, Any]) -> str:
    return str(row.get("condition_group") or NO_REPORTED_CONDITION)
