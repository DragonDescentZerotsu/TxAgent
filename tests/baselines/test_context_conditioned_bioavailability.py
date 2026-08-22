from __future__ import annotations

import numpy as np
import pytest

from baselines.context_conditioned_bioavailability.run import (
    condition_one_hot,
    condition_vocabulary,
    select_context_neighbors,
)
from tools.chembl_tool.tasks.bioavailability_ma.condition_ontology import (
    NO_REPORTED_CONDITION,
)


def row(parent: str, group: str, label: int = 1) -> dict:
    return {
        "drug": "CC",
        "Y": label,
        "condition_group": group,
        "condition_scope": "none_reported" if group == NO_REPORTED_CONDITION else "external",
        "molecule_identity_key": parent,
    }


def test_external_retrieval_prefers_exact_then_fills_from_null() -> None:
    train = [
        row("p1", "prandial_state=fasted"),
        row("p2", "prandial_state=fasted"),
        row("p3", NO_REPORTED_CONDITION),
        row("p4", NO_REPORTED_CONDITION),
    ]
    similarities = np.asarray([0.1, 0.2, 0.99, 0.8])
    selected = select_context_neighbors(
        row("query", "prandial_state=fasted"), train, similarities, k=3
    )
    assert selected == [
        (1, "exact_condition"),
        (0, "exact_condition"),
        (2, "fallback_no_reported_external_condition"),
    ]


def test_null_query_never_retrieves_external_rows() -> None:
    train = [
        row("p1", "prandial_state=fasted"),
        row("p2", NO_REPORTED_CONDITION),
        row("p3", NO_REPORTED_CONDITION),
        row("p4", NO_REPORTED_CONDITION),
    ]
    selected = select_context_neighbors(
        row("query", NO_REPORTED_CONDITION), train, np.asarray([1.0, 0.3, 0.2, 0.1]), k=3
    )
    assert [index for index, _ in selected] == [1, 2, 3]


def test_fallback_deduplicates_parent_already_selected_exact() -> None:
    train = [
        row("p1", "prandial_state=fasted"),
        row("p1", NO_REPORTED_CONDITION),
        row("p2", NO_REPORTED_CONDITION),
        row("p3", NO_REPORTED_CONDITION),
    ]
    selected = select_context_neighbors(
        row("query", "prandial_state=fasted"), train, np.asarray([0.5, 1.0, 0.4, 0.3]), k=3
    )
    assert [index for index, _ in selected] == [0, 2, 3]


def test_condition_one_hot_is_train_derived_and_one_active() -> None:
    train = [row("p1", NO_REPORTED_CONDITION), row("p2", "prandial_state=fasted")]
    vocabulary = condition_vocabulary(train)
    encoded = condition_one_hot(train, vocabulary)
    assert encoded.shape == (2, 2)
    assert encoded.sum(dim=1).tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="absent from train"):
        condition_one_hot([row("p3", "disease=cirrhosis")], vocabulary)
