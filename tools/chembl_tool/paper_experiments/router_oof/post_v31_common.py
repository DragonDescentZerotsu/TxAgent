"""Shared data semantics for the frozen v3.1 train-only diagnostics.

This module intentionally contains no model fitting or experiment-specific
policy.  It centralizes only the row-to-array conversion and direction rule
that must remain identical in the matched-size and cross-task diagnoses.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .post_contract import POST_SELECTOR_DIRECTIONS


@dataclass(frozen=True)
class PostSelectorArrays:
    """Canonical numerical views of one task's post-selector feature rows."""

    folds: np.ndarray
    labels: np.ndarray
    knn: np.ndarray
    agent: np.ndarray
    agent_win_target: np.ndarray

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "PostSelectorArrays":
        return cls(
            folds=np.asarray([int(row["fold"]) for row in rows], dtype=int),
            labels=np.asarray([int(row["Y"]) for row in rows], dtype=int),
            knn=np.asarray([int(row["knn_prediction"]) for row in rows], dtype=int),
            agent=np.asarray([int(row["agent_prediction"]) for row in rows], dtype=int),
            agent_win_target=np.asarray(
                [int(row["paired_outcome"] == "agent_only_correct") for row in rows],
                dtype=int,
            ),
        )

    def direction_masks(self) -> dict[str, np.ndarray]:
        return {
            name: (self.knn == pair[0]) & (self.agent == pair[1])
            for name, pair in POST_SELECTOR_DIRECTIONS.items()
        }


def direction_only_prediction(knn: np.ndarray, agent: np.ndarray) -> np.ndarray:
    """Return the no-feature OR comparator used by both diagnostics."""

    return np.where((knn == 0) & (agent == 1), agent, knn)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
