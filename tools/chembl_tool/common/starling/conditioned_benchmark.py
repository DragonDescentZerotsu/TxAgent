"""Canonical paths and schema constants for the current conditioned benchmark."""

from __future__ import annotations

from pathlib import Path


BENCHMARK_ROOT = Path("data/conditioned_benchmark")
BUILD_ROOT = Path("data/.build/conditioned_benchmark_sources")
CONTRACT = "conditioned_benchmark.v1"
NO_REPORTED_CONDITION = "no_reported_external_condition"

TASK_DIRECTORIES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "clintox": "ClinTox",
    "skin_reaction": "Skin_Reaction",
}


def task_root(task: str) -> Path:
    """Return the single current scaffold-split root for ``task``."""

    try:
        directory = TASK_DIRECTORIES[task]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_DIRECTORIES))
        raise ValueError(f"Unknown conditioned benchmark task {task!r}; choose {choices}") from exc
    return BENCHMARK_ROOT / directory / "scaffold"


def split_path(task: str, split: str) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown split {split!r}")
    return task_root(task) / f"{split}.jsonl"
