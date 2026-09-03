"""Data contract for the Starling paper Table 2 MiniMol reproduction."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rdkit import Chem, RDLogger

TaskType = Literal["classification", "regression"]
ExtractionWeight = Literal["none", "sqrt", "linear"]


@dataclass(frozen=True)
class Table2Task:
    slug: str
    paper_name: str
    prefix: str
    task_type: TaskType
    metric: Literal["auroc", "mae"]
    hidden_dim: int
    depth: int
    learning_rate: float
    positive_threshold: float | None = None


TASKS = {
    task.slug: task
    for task in (
        Table2Task(
            slug="oral_bioavailability",
            paper_name="Oral Bioavailability",
            prefix="oral_ba",
            task_type="classification",
            metric="auroc",
            hidden_dim=512,
            depth=3,
            learning_rate=3e-4,
            positive_threshold=0.20,
        ),
        Table2Task(
            slug="ld50",
            paper_name="LD50",
            prefix="ld50",
            task_type="regression",
            metric="mae",
            hidden_dim=1024,
            depth=4,
            learning_rate=1e-4,
        ),
        Table2Task(
            slug="bbb",
            paper_name="BBB",
            prefix="bbb",
            task_type="classification",
            metric="auroc",
            hidden_dim=2048,
            depth=3,
            learning_rate=1e-4,
            positive_threshold=0.50,
        ),
    )
}

PAPER_TABLE2 = {
    "oral_bioavailability": {
        "tdc_test": 0.692,
        "tdc_test_augmented": 0.779,
        "literature_test": 0.768,
        "literature_test_augmented": 0.821,
    },
    "ld50": {
        "tdc_test": 0.602,
        "tdc_test_augmented": 0.575,
        "literature_test": 0.650,
        "literature_test_augmented": 0.641,
    },
    "bbb": {
        "tdc_test": 0.916,
        "tdc_test_augmented": 0.909,
        "literature_test": 0.768,
        "literature_test_augmented": 0.890,
    },
}


@dataclass
class MoleculeTable:
    smiles: list[str]
    targets: list[float]
    raw_labels: list[float]
    sources: list[str | None]
    n_extractions: list[int | None]

    def subset(self, indices: list[int]) -> "MoleculeTable":
        return MoleculeTable(
            smiles=[self.smiles[index] for index in indices],
            targets=[self.targets[index] for index in indices],
            raw_labels=[self.raw_labels[index] for index in indices],
            sources=[self.sources[index] for index in indices],
            n_extractions=[self.n_extractions[index] for index in indices],
        )


@dataclass
class TaskTables:
    tdc_train: MoleculeTable
    augmented_train: MoleculeTable
    tdc_test: MoleculeTable
    literature_test: MoleculeTable
    files: dict[str, Path]


def extraction_weights(table: MoleculeTable, policy: ExtractionWeight) -> list[float]:
    """Return molecule-row weights; TDC rows without extraction counts stay at 1."""
    if policy == "none":
        return [1.0] * len(table.smiles)
    if policy not in {"sqrt", "linear"}:
        raise ValueError(f"unsupported extraction-weight policy: {policy}")
    transform = math.sqrt if policy == "sqrt" else float
    return [transform(count) if count is not None else 1.0 for count in table.n_extractions]


def filter_rdkit_invalid(tables: TaskTables) -> tuple[TaskTables, list[dict[str, object]]]:
    """Remove molecules MiniMol cannot featurize and return an audit trail.

    The released CSVs remain immutable. A molecule is excluded from every surface
    where it occurs when RDKit's normal sanitized parser rejects it; this is the
    same prerequisite used by MiniMol's graph featurizer and scaffold splitting.
    """
    named = {
        "tdc_train": tables.tdc_train,
        "augmented_train": tables.augmented_train,
        "tdc_test": tables.tdc_test,
        "literature_test": tables.literature_test,
    }
    RDLogger.DisableLog("rdApp.*")
    invalid: dict[str, dict[str, object]] = {}
    for surface, table in named.items():
        for index, molecule in enumerate(table.smiles):
            if Chem.MolFromSmiles(molecule) is not None:
                continue
            record = invalid.setdefault(
                molecule,
                {
                    "canonical_smiles": molecule,
                    "reason": "rdkit_sanitized_parse_failed",
                    "occurrences": [],
                },
            )
            record["occurrences"].append(
                {
                    "surface": surface,
                    "row_index": index,
                    "raw_label": table.raw_labels[index],
                    "source": table.sources[index],
                }
            )

    invalid_smiles = set(invalid)

    def keep_valid(table: MoleculeTable) -> MoleculeTable:
        return table.subset(
            [index for index, molecule in enumerate(table.smiles) if molecule not in invalid_smiles]
        )

    filtered = TaskTables(
        tdc_train=keep_valid(tables.tdc_train),
        augmented_train=keep_valid(tables.augmented_train),
        tdc_test=keep_valid(tables.tdc_test),
        literature_test=keep_valid(tables.literature_test),
        files=tables.files,
    )
    return filtered, list(invalid.values())


def load_task_tables(
    data_dir: Path,
    task: Table2Task,
    *,
    train_label_policy: Literal["paper_soft", "hard"] = "paper_soft",
) -> TaskTables:
    """Load the four Table 2 surfaces and validate their label semantics."""
    files = {
        "augmented_train": data_dir / f"{task.prefix}_tdc_plus_lit_train.csv",
        "tdc_test": data_dir / f"{task.prefix}_tdc_test.csv",
        "literature_test": data_dir / f"{task.prefix}_lit_test.csv",
    }
    augmented = _load_csv(
        files["augmented_train"],
        task,
        evaluation=False,
        train_label_policy=train_label_policy,
    )
    tdc_indices = [index for index, source in enumerate(augmented.sources) if source == "tdc"]
    if not tdc_indices or any(source not in {"tdc", "lit"} for source in augmented.sources):
        raise ValueError(f"{files['augmented_train']} must contain source values tdc and lit")

    tables = TaskTables(
        tdc_train=augmented.subset(tdc_indices),
        augmented_train=augmented,
        tdc_test=_load_csv(
            files["tdc_test"],
            task,
            evaluation=True,
            train_label_policy=train_label_policy,
        ),
        literature_test=_load_csv(
            files["literature_test"],
            task,
            evaluation=True,
            train_label_policy=train_label_policy,
        ),
        files=files,
    )
    _validate_tables(task, tables)
    return tables


def _load_csv(
    path: Path,
    task: Table2Task,
    *,
    evaluation: bool,
    train_label_policy: Literal["paper_soft", "hard"],
) -> MoleculeTable:
    if not path.is_file():
        raise FileNotFoundError(path)
    smiles: list[str] = []
    targets: list[float] = []
    raw_labels: list[float] = []
    sources: list[str | None] = []
    n_extractions: list[int | None] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                molecule = row["canonical_smiles"].strip()
                raw_label = float(row["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} has an invalid required field") from exc
            if not molecule:
                raise ValueError(f"{path}:{line_number} has an empty canonical_smiles")

            if task.task_type == "classification":
                if not 0.0 <= raw_label <= 1.0:
                    raise ValueError(f"{path}:{line_number} classification label is outside [0, 1]")
                assert task.positive_threshold is not None
                hard_label = float(raw_label >= task.positive_threshold)
                if evaluation and row.get("label_hard", "") != "":
                    declared = float(row["label_hard"])
                    if declared != hard_label:
                        raise ValueError(f"{path}:{line_number} label_hard contradicts the task threshold")
                    target = declared
                elif evaluation:
                    target = hard_label
                else:
                    target = raw_label if train_label_policy == "paper_soft" else hard_label
            else:
                if train_label_policy == "hard":
                    raise ValueError("hard train labels are not defined for the LD50 regression task")
                target = raw_label

            smiles.append(molecule)
            targets.append(target)
            raw_labels.append(raw_label)
            sources.append(row.get("source") or None)
            count = row.get("n_extractions")
            n_extractions.append(int(count) if count not in {None, ""} else None)

    if not smiles:
        raise ValueError(f"{path} is empty")
    return MoleculeTable(smiles, targets, raw_labels, sources, n_extractions)


def _validate_tables(task: Table2Task, tables: TaskTables) -> None:
    named = {
        "tdc_train": tables.tdc_train,
        "augmented_train": tables.augmented_train,
        "tdc_test": tables.tdc_test,
        "literature_test": tables.literature_test,
    }
    for name, table in named.items():
        if len(table.smiles) != len(set(table.smiles)):
            raise ValueError(f"{task.slug} {name} contains duplicate canonical_smiles")

    tdc_train = set(tables.tdc_train.smiles)
    augmented = set(tables.augmented_train.smiles)
    tdc_test = set(tables.tdc_test.smiles)
    literature_test = set(tables.literature_test.smiles)
    if not tdc_train <= augmented:
        raise ValueError(f"{task.slug} TDC training molecules are missing from augmented training")
    if augmented & tdc_test or augmented & literature_test or tdc_test & literature_test:
        raise ValueError(f"{task.slug} train/test molecule overlap detected")

    if task.task_type == "classification":
        for name in ("tdc_test", "literature_test"):
            labels = set(named[name].targets)
            if labels != {0.0, 1.0}:
                raise ValueError(f"{task.slug} {name} must contain both hard classes")


def file_receipt(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
