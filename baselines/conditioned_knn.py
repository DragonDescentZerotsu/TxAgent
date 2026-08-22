"""Shared condition-aware selection for pre-ranked KNN candidates."""

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

    representative = _canonical_reference_indices(reference)

    def select_group(
        condition_group: str | None,
        source: str,
        selected: list[tuple[int, str]],
    ) -> None:
        selected_keys = {_molecule_key(reference[index]) for index, _ in selected}
        for index in ranked_indices:
            row = reference[index]
            key = _molecule_key(row)
            if representative[key] != index or key in selected_keys:
                continue
            if condition_group is not None and _condition_group(row) != condition_group:
                continue
            selected.append((index, source))
            selected_keys.add(key)
            if len(selected) == k:
                return

    selected: list[tuple[int, str]] = []
    if policy == "all_train_unique_molecules":
        select_group(None, "all_train_unique_molecules", selected)
        return selected

    query_group = _condition_group(query)
    # In the same-condition stage, use that condition's row instead of the
    # null-preferred canonical row used by the condition-agnostic policy.
    same_group_indices: dict[str, int] = {}
    for index, row in enumerate(reference):
        if _condition_group(row) == query_group:
            same_group_indices.setdefault(_molecule_key(row), index)
    selected_keys: set[str] = set()
    for index in ranked_indices:
        key = _molecule_key(reference[index])
        same_index = same_group_indices.get(key)
        if same_index is None or same_index != index or key in selected_keys:
            continue
        selected.append((index, "same_condition"))
        selected_keys.add(key)
        if len(selected) == k:
            return selected
    if query_group == NO_REPORTED_CONDITION:
        return selected

    null_indices: dict[str, int] = {}
    for index, row in enumerate(reference):
        if _condition_group(row) == NO_REPORTED_CONDITION:
            null_indices.setdefault(_molecule_key(row), index)
    for index in ranked_indices:
        key = _molecule_key(reference[index])
        null_index = null_indices.get(key)
        if null_index is None or null_index != index or key in selected_keys:
            continue
        selected.append((index, "null_fallback"))
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
