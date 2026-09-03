"""Train-derived categorical condition features for MiniMol task heads."""

from __future__ import annotations

import torch


CONDITION_ENCODING_VERSION = "train_derived_one_hot.v1"


def condition_vocabulary(values: list[str]) -> list[str]:
    """Return the stable outer-train vocabulary for a categorical condition."""
    if not values:
        raise ValueError("Condition vocabulary requires at least one training value")
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError("Condition values must be non-empty strings")
    return sorted(set(normalized))


def condition_one_hot(values: list[str], vocabulary: list[str]) -> torch.Tensor:
    """Encode values against an immutable train-derived vocabulary."""
    if not vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Condition vocabulary must be non-empty and unique")
    index = {value: position for position, value in enumerate(vocabulary)}
    normalized = [str(value).strip() for value in values]
    unknown = sorted(set(normalized) - index.keys())
    if unknown:
        raise ValueError(f"Condition values are absent from train: {unknown}")
    encoded = torch.zeros((len(normalized), len(vocabulary)), dtype=torch.float32)
    if normalized:
        encoded[torch.arange(len(normalized)), [index[value] for value in normalized]] = 1.0
    return encoded


def append_condition_one_hot(
    embeddings: torch.Tensor,
    values: list[str] | None,
    vocabulary: list[str] | None,
) -> torch.Tensor:
    """Append one-hot features, or return molecule embeddings unchanged."""
    if values is None and vocabulary is None:
        return embeddings
    if values is None or vocabulary is None:
        raise ValueError("Condition values and vocabulary must be provided together")
    if len(values) != embeddings.shape[0]:
        raise ValueError("Condition values must align with embedding rows")
    return torch.cat([embeddings, condition_one_hot(values, vocabulary)], dim=1)


def condition_feature_contract(
    *,
    field: str | None,
    vocabulary: list[str] | None,
    molecule_embedding_dim: int,
) -> dict[str, object]:
    enabled = field is not None
    dimension = len(vocabulary or [])
    return {
        "enabled": enabled,
        "field": field,
        "encoding": CONDITION_ENCODING_VERSION if enabled else None,
        "vocabulary_scope": "outer_train_only" if enabled else None,
        "vocabulary": list(vocabulary or []),
        "condition_feature_dim": dimension,
        "molecule_embedding_dim": molecule_embedding_dim,
        "model_input_dim": molecule_embedding_dim + dimension,
        "unseen_evaluation_policy": "error" if enabled else None,
    }
