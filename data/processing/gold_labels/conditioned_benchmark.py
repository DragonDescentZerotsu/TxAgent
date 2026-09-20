"""Canonical paths and schema constants for the current conditioned benchmark."""

from __future__ import annotations

from pathlib import Path

from data.processing.paths import GOLD_LABELS_ROOT

BENCHMARK_ROOT = GOLD_LABELS_ROOT
TDC_ROOT = BENCHMARK_ROOT / "TDC"
BUILD_ROOT = Path("data/.build/conditioned_benchmark_sources")
CONTRACT = "conditioned_benchmark.v1"
NO_REPORTED_CONDITION = "no_reported_external_condition"

TASK_DIRECTORIES = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
    "ames": "Ames",
    "dili": "DILI",
    "carcinogens": "Carcinogens",
}

SPLITS = frozenset({"train", "valid", "valid_small", "test"})


def task_root(task: str, version: str = "current") -> Path:
    """Return the current or explicitly versioned scaffold-split root."""

    try:
        directory = TASK_DIRECTORIES[task]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_DIRECTORIES))
        raise ValueError(f"Unknown conditioned benchmark task {task!r}; choose {choices}") from exc
    task_dir = BENCHMARK_ROOT / directory
    if version == "current":
        current = task_dir / "CURRENT"
        if not current.is_file():
            raise FileNotFoundError(f"active gold-label pointer is absent: {current}")
        version = current.read_text(encoding="utf-8").strip()
    if not version or "/" in version or "\\" in version:
        raise ValueError(f"invalid gold-label release version: {version!r}")
    root = task_dir / version / "scaffold"
    if not root.is_dir():
        raise FileNotFoundError(f"gold-label release is absent: {root}")
    return root


def split_path(task: str, split: str, version: str = "current") -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    return task_root(task, version) / f"{split}.jsonl"


def tdc_task_root(task: str, version: str = "v1") -> Path:
    """Return an explicitly versioned TDC scaffold-split root."""

    try:
        directory = TASK_DIRECTORIES[task]
    except KeyError as exc:
        choices = ", ".join(sorted(TASK_DIRECTORIES))
        raise ValueError(f"Unknown TDC benchmark task {task!r}; choose {choices}") from exc
    if not version or "/" in version or "\\" in version:
        raise ValueError(f"invalid TDC release version: {version!r}")
    root = TDC_ROOT / directory / version / "scaffold"
    if not root.is_dir():
        raise FileNotFoundError(f"TDC release is absent: {root}")
    return root


def tdc_split_path(task: str, split: str, version: str = "v1") -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    return tdc_task_root(task, version) / f"{split}.jsonl"
