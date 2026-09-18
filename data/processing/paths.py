"""Canonical paths for raw data, gold labels, and evidence-library releases."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
GOLD_LABELS_ROOT = DATA_ROOT / "gold_labels"
LEGACY_GOLD_LABELS_ROOT = GOLD_LABELS_ROOT / "legacy"
EVIDENCE_LIBRARIES_ROOT = DATA_ROOT / "evidence_libraries"
ARTIFACTS_ROOT = DATA_ROOT / "artifacts"
LEGACY_ROOT = DATA_ROOT / "legacy"

_TASK_NAMES = {
    "BBB_Martins": "bbb_martins",
    "Bioavailability_Ma": "bioavailability_ma",
    "Skin_Reaction": "skin_reaction",
    "Ames": "ames",
    "DILI": "dili",
    "Carcinogens": "carcinogens",
    "bbb_martins": "bbb_martins",
    "bioavailability_ma": "bioavailability_ma",
    "skin_reaction": "skin_reaction",
    "ames": "ames",
    "dili": "dili",
    "carcinogens": "carcinogens",
}
KNOWN_RELEASES = {
    "bbb_martins": ("v7", "v8", "v9", "v10"),
    "bioavailability_ma": ("v7", "v8", "v9", "v10"),
    "skin_reaction": ("v7", "v8", "v9", "v10"),
    "ames": ("v8", "v9", "v10"),
    "dili": ("v10",),
    "carcinogens": ("v10",),
}


def task_id(value: str) -> str:
    try:
        return _TASK_NAMES[value]
    except KeyError as error:
        raise ValueError(f"unsupported active data task: {value}") from error


def raw_starling_task_root(task: str) -> Path:
    return RAW_ROOT / "starling" / task_id(task)


def evidence_library_root(task: str, version: str | None = None) -> Path:
    task = task_id(task)
    task_root = EVIDENCE_LIBRARIES_ROOT / task
    if version is None:
        current = task_root / "CURRENT"
        if not current.is_file():
            raise FileNotFoundError(f"active release pointer is absent: {current}")
        version = current.read_text(encoding="utf-8").strip()
    if version not in KNOWN_RELEASES[task]:
        raise ValueError(f"unsupported {task} evidence-library release: {version}")
    return task_root / version


def assert_canonical_library_output(path: str | Path) -> Path:
    """Reject writes through the historical output-tree aliases."""
    candidate = Path(path)
    lexical = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    legacy = REPO_ROOT / "outputs" / "chembl_tool" / "tasks"
    try:
        lexical.relative_to(legacy)
    except ValueError:
        return candidate
    if "evidence_library" in lexical.parts:
        raise ValueError(
            "historical evidence-library paths are read-only aliases; write under "
            f"{EVIDENCE_LIBRARIES_ROOT}"
        )
    return candidate
