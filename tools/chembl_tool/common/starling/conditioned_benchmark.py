"""Canonical paths and schema constants for the current conditioned benchmark."""

from __future__ import annotations

from pathlib import Path


BENCHMARK_ROOT = Path("data/conditioned_benchmark")
BUILD_ROOT = Path("data/.build/conditioned_benchmark_sources")
CONTRACT = "conditioned_benchmark.v1"
NO_REPORTED_CONDITION = "no_reported_external_condition"
SPLIT_SCHEMES = ("scaffold", "random")

TASK_DIRECTORIES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "clintox": "ClinTox",
    "skin_reaction": "Skin_Reaction",
    "ames": "Ames",
    "dili": "DILI",
    "carcinogens": "Carcinogens",
}


def task_root(task: str, split_scheme: str = "scaffold") -> Path:
    """Return one current benchmark root, defaulting to the scaffold split."""

    try:
        directory = TASK_DIRECTORIES[task]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_DIRECTORIES))
        raise ValueError(f"Unknown conditioned benchmark task {task!r}; choose {choices}") from exc
    if split_scheme not in SPLIT_SCHEMES:
        choices = ", ".join(SPLIT_SCHEMES)
        raise ValueError(f"Unknown split scheme {split_scheme!r}; choose {choices}")
    return BENCHMARK_ROOT / directory / split_scheme


def split_path(task: str, split: str, split_scheme: str = "scaffold") -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown split {split!r}")
    return task_root(task, split_scheme) / f"{split}.jsonl"
