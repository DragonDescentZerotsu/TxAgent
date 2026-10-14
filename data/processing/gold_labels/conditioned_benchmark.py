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


TDC_MANIFEST = TDC_ROOT / "manifest.json"


def tdc_config(task: str | None = None) -> dict:
    """Read the single canonical TDC configuration, without version selection."""
    import json
    document = json.loads(TDC_MANIFEST.read_text(encoding="utf-8"))
    if document.get("benchmark") != "tdc" or document.get("status") != "complete":
        raise ValueError(f"Invalid canonical TDC configuration: {TDC_MANIFEST}")
    if set(document.get("tasks", {})) != set(TASK_DIRECTORIES):
        raise ValueError("Canonical TDC must configure all six tasks")
    if task is None:
        return document
    if task not in TASK_DIRECTORIES:
        raise ValueError(f"Unknown TDC benchmark task: {task!r}")
    return document["tasks"][task]


def tdc_task_root(task: str) -> Path:
    root = TDC_ROOT / tdc_config(task)["directory"]
    if not root.is_dir():
        raise FileNotFoundError(f"Canonical TDC task is absent: {root}")
    return root


def tdc_split_path(task: str, split: str) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported TDC split: {split}")
    return tdc_task_root(task) / f"{split}.jsonl"


def tdc_manifest(task: str, split: str) -> Path:
    """Validate the selected frozen query files against the canonical config."""
    from predict.utils.json import sha256_file
    spec = tdc_config(task)
    root = tdc_task_root(task)
    if split not in {"valid", "test"}:
        raise ValueError(f"Unsupported TDC evaluation split: {split}")
    for name in (f"{split}.jsonl", f"{split}_molecule_condition_labels.jsonl"):
        if sha256_file(root / name) != spec["outputs"].get(name):
            raise ValueError(f"Canonical TDC split changed: {root / name}")
    return TDC_MANIFEST
