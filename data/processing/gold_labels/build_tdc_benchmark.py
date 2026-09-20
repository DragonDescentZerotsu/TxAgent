"""Publish TDC binary datasets in the conditioned-benchmark row schema.

The builder reads pinned raw TDC tables, normalizes molecules with the shared
parent policy, removes conflicting parent labels, repairs split leakage at the
scaffold level, and writes an explicit TDC v1 lineage.  It never changes a
task's active Gold ``CURRENT`` pointer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from random import Random
from typing import Any, Iterable

from data.processing.gold_labels.benchmark_dataset import (
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)
from data.processing.gold_labels.conditioned_benchmark import (
    NO_REPORTED_CONDITION,
    TASK_DIRECTORIES,
    TDC_ROOT,
)
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


ROOT = Path(__file__).resolve().parents[3]
VERSION = "v1"
RELEASE_ID = "tdc_v1"
CONTRACT = "tdc_conditioned_benchmark.v1"
SPLITS = ("train", "valid", "test")
SPLIT_PRECEDENCE = {"train": 0, "valid": 1, "test": 2}
MAIN_COMMIT = "8972834103833c18a8d269a92f898bad07f7e5ab"
AMES_COMMIT = "f7af9885bc0cdfa2969b52db0065c94bd1f00567"

TASKS: dict[str, dict[str, Any]] = {
    "bbb_martins": {
        "dataset": "BBB_Martins",
        "target": "TDC BBB penetration label; 1 is BBB-positive and 0 is BBB-negative.",
        "format": "jsonl_splits",
        "root": "data/gold_labels/legacy/processed/BBB_Martins",
        "upstream_root": "data/processed/BBB_Martins",
        "upstream": MAIN_COMMIT,
    },
    "bioavailability_ma": {
        "dataset": "Bioavailability_Ma",
        "target": "TDC oral bioavailability label; 1 is F >= 20% and 0 is F < 20%.",
        "format": "jsonl_splits",
        "root": "data/gold_labels/legacy/processed/Bioavailability_Ma",
        "upstream_root": "data/processed/Bioavailability_Ma",
        "upstream": MAIN_COMMIT,
    },
    "skin_reaction": {
        "dataset": "Skin_Reaction",
        "target": "TDC skin-reaction label; 1 is reaction-positive and 0 is reaction-negative.",
        "format": "jsonl_splits",
        "root": "data/gold_labels/legacy/processed/Skin_Reaction",
        "upstream_root": "data/processed/Skin_Reaction",
        "upstream": MAIN_COMMIT,
    },
    "ames": {
        "dataset": "AMES",
        "target": "TDC Ames mutagenicity label; 1 is mutagenic and 0 is non-mutagenic.",
        "format": "csv_splits",
        "root": "data/raw/tdc/Ames/v1",
        "upstream": AMES_COMMIT,
        "source_location": (
            "/vast/projects/myatskar/design-documents/joseph/"
            "therapeutic-tuning/data/raw/original/AMES"
        ),
        "source_names": {"train": "train.csv", "valid": "val.csv", "test": "test.csv"},
    },
    "dili": {
        "dataset": "DILI",
        "target": "TDC drug-induced liver injury label; 1 is DILI-positive and 0 is DILI-negative.",
        "format": "tsv",
        "root": "data/raw/tdc/DILI/v1",
        "file": "dili.tab",
        "upstream_root": "data/starling_data/new_tasks_gold_audit/tdc_comparison",
        "upstream": MAIN_COMMIT,
    },
    "carcinogens": {
        "dataset": "Carcinogens_Lagunin",
        "target": "TDC carcinogenicity label; 1 is carcinogenic and 0 is non-carcinogenic.",
        "format": "tsv",
        "root": "data/raw/tdc/Carcinogens/v1",
        "file": "carcinogens_lagunin.tab",
        "upstream_root": "data/starling_data/new_tasks_gold_audit/tdc_comparison",
        "upstream": MAIN_COMMIT,
    },
}


def _binary_label(value: Any) -> int:
    numeric = float(value)
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"TDC label is not binary: {value!r}")
    return int(numeric)


def _source_files(spec: dict[str, Any]) -> list[tuple[str | None, Path]]:
    root = ROOT / spec["root"]
    if spec["format"] == "jsonl_splits":
        return [(split, root / f"{split}.jsonl") for split in SPLITS]
    if spec["format"] == "csv_splits":
        return [(split, root / f"{split}.csv") for split in SPLITS]
    return [(None, root / spec["file"])]


def _read_tabular(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _source_rows(task: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = TASKS[task]
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for split, path in _source_files(spec):
        if not path.is_file():
            raise FileNotFoundError(path)
        source = read_jsonl(path) if path.suffix == ".jsonl" else _read_tabular(
            path, "\t" if path.suffix == ".tab" else ","
        )
        files.append({
            "path": str(path.relative_to(ROOT)),
            **(
                {
                    "upstream_path": str(
                        Path(spec["upstream_root"])
                        / (f"{split}.jsonl" if split else spec["file"])
                    )
                }
                if spec.get("upstream_root")
                else {}
            ),
            **(
                {
                    "source_path": str(
                        Path(spec["source_location"])
                        / spec["source_names"][str(split)]
                    )
                }
                if spec.get("source_location")
                else {}
            ),
            "sha256": sha256_file(path),
            "rows": len(source),
            "split": split,
        })
        for index, row in enumerate(source):
            rows.append({
                "source_order": len(rows),
                "source_split": split,
                "source_record_id": str(
                    row.get("Drug_ID") or f"{split or 'all'}:{index:06d}"
                ),
                "drug": str(row.get("Drug") or row.get("drug") or "").strip(),
                "Y": row.get("Y"),
            })
    return rows, files


def _normalized_rows(source: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, rejected = [], []
    for row in source:
        try:
            label = _binary_label(row["Y"])
            identity = normalize_molecule_identity(row["drug"])
        except (TypeError, ValueError) as exc:
            rejected.append({**row, "drop_reason": f"invalid_label:{exc}"})
            continue
        if identity.status != "ok" or not identity.parent_smiles:
            rejected.append({**row, "drop_reason": "invalid_or_unresolved_smiles"})
            continue
        identity_dict = identity.to_dict()
        accepted.append({
            **row,
            "Y": label,
            "molecule_identity": identity_dict,
            "molecule_identity_key": identity.parent_inchi_key or identity.parent_smiles,
            "drug": identity.parent_smiles,
            "bemis_murcko_scaffold": bemis_murcko_scaffold(identity.parent_smiles),
        })
    return accepted, rejected


def _collapse_parents(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["molecule_identity_key"]].append(row)
    accepted, conflicts = [], []
    for parent_rows in sorted(grouped.values(), key=lambda values: min(r["source_order"] for r in values)):
        labels = sorted({row["Y"] for row in parent_rows})
        if len(labels) != 1:
            conflicts.append({
                "molecule_identity_key": parent_rows[0]["molecule_identity_key"],
                "labels": labels,
                "source_record_ids": [row["source_record_id"] for row in parent_rows],
                "drop_reason": "conflicting_parent_labels",
            })
            continue
        first = min(parent_rows, key=lambda row: row["source_order"])
        accepted.append({
            **first,
            "source_record_count": len(parent_rows),
            "source_record_ids": [row["source_record_id"] for row in parent_rows],
            "source_splits": sorted(
                {row["source_split"] for row in parent_rows if row["source_split"]},
                key=SPLIT_PRECEDENCE.get,
            ),
        })
    return accepted, conflicts


def _repair_provided_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row["split"] = max(row["source_splits"], key=SPLIT_PRECEDENCE.get)
    by_scaffold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["bemis_murcko_scaffold"]:
            by_scaffold[row["bemis_murcko_scaffold"]].append(row)
    repairs = []
    for scaffold_rows in by_scaffold.values():
        selected = max((row["split"] for row in scaffold_rows), key=SPLIT_PRECEDENCE.get)
        for row in scaffold_rows:
            if row["split"] != selected:
                repairs.append({
                    "molecule_identity_key": row["molecule_identity_key"],
                    "bemis_murcko_scaffold": row["bemis_murcko_scaffold"],
                    "from_split": row["split"],
                    "to_split": selected,
                    "reason": "cross_split_scaffold_precedence",
                })
                row["split"] = selected
    return repairs


def _tdc_scaffold_split(rows: list[dict[str, Any]]) -> None:
    scaffolds: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(rows):
        scaffolds[row["bemis_murcko_scaffold"]].add(index)
    train_size, valid_size = int(len(rows) * 0.7), int(len(rows) * 0.1)
    test_size = len(rows) - train_size - valid_size
    large, small = [], []
    for indices in scaffolds.values():
        (large if len(indices) > valid_size / 2 or len(indices) > test_size / 2 else small).append(indices)
    random = Random(42)
    random.shuffle(large)
    random.shuffle(small)
    assigned = {"train": [], "valid": [], "test": []}
    for indices in large + small:
        split = (
            "train" if len(assigned["train"]) + len(indices) <= train_size
            else "valid" if len(assigned["valid"]) + len(indices) <= valid_size
            else "test"
        )
        assigned[split].extend(indices)
    for split, indices in assigned.items():
        for index in indices:
            rows[index]["split"] = split


def _benchmark_row(task: str, row: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{task}\0{row['molecule_identity_key']}".encode("utf-8")
    ).hexdigest()[:20].upper()
    label = int(row["Y"])
    return {
        "drug": row["drug"],
        "molecule_identity_key": row["molecule_identity_key"],
        "molecule_identity": row["molecule_identity"],
        "bemis_murcko_scaffold": row["bemis_murcko_scaffold"],
        "condition_scope": "none_reported",
        "condition_group": NO_REPORTED_CONDITION,
        "condition_atoms": [],
        "label_counts": {str(label): row["source_record_count"]},
        "source_record_count": row["source_record_count"],
        "source_ids": [f"TDC:{TASKS[task]['dataset']}"],
        "source_record_ids": row["source_record_ids"],
        "source_pmids": [],
        "label_methods": {"tdc_external_binary_label.v1": row["source_record_count"]},
        "raw_value_examples": [str(label)],
        "context_examples": [],
        "majority_label": label,
        "majority_record_count": row["source_record_count"],
        "minority_record_count": 0,
        "agreement_fraction": 1.0,
        "vote_unit": "external_dataset_label",
        "record_support_eligible": False,
        "label_source": "tdc",
        "Y": label,
        "label_decision": "unanimous",
        "split": row["split"],
        "split_policy": CONTRACT,
        "benchmark_row_id": f"TDC_{task.upper()}_{digest}",
    }


def _minimal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (
        "drug", "Y", "condition_group", "condition_scope", "molecule_identity_key",
        "bemis_murcko_scaffold", "benchmark_row_id",
    )}


def _validate_splits(by_split: dict[str, list[dict[str, Any]]]) -> None:
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        parents_left = {row["molecule_identity_key"] for row in by_split[left]}
        parents_right = {row["molecule_identity_key"] for row in by_split[right]}
        scaffolds_left = {row["bemis_murcko_scaffold"] for row in by_split[left] if row["bemis_murcko_scaffold"]}
        scaffolds_right = {row["bemis_murcko_scaffold"] for row in by_split[right] if row["bemis_murcko_scaffold"]}
        if parents_left & parents_right or scaffolds_left & scaffolds_right:
            raise ValueError(f"TDC split leakage remains between {left} and {right}")
    if any({row["Y"] for row in rows} != {0, 1} for rows in by_split.values()):
        raise ValueError("every TDC split must contain both binary labels")


def build_task(task: str, *, output_root: Path = TDC_ROOT) -> dict[str, Any]:
    source, files = _source_rows(task)
    normalized, rejected = _normalized_rows(source)
    collapsed, conflicts = _collapse_parents(normalized)
    repairs = _repair_provided_splits(collapsed) if TASKS[task]["format"].endswith("splits") else []
    if not TASKS[task]["format"].endswith("splits"):
        _tdc_scaffold_split(collapsed)
    detailed = [_benchmark_row(task, row) for row in collapsed]
    by_split = {split: [row for row in detailed if row["split"] == split] for split in SPLITS}
    for rows in by_split.values():
        rows.sort(key=lambda row: (row["molecule_identity_key"], row["benchmark_row_id"]))
    _validate_splits(by_split)

    directory = TASK_DIRECTORIES[task]
    destination = output_root / directory / VERSION / "scaffold"
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        write_jsonl_atomic(destination / f"{split}.jsonl", [_minimal(row) for row in rows])
        write_jsonl_atomic(destination / f"{split}_molecule_condition_labels.jsonl", rows)
    write_jsonl_atomic(destination / "heldout_molecule_condition_labels.jsonl", by_split["valid"] + by_split["test"])
    write_jsonl_atomic(destination / "rejected_source_rows.jsonl", rejected)
    write_jsonl_atomic(destination / "conflicting_parent_labels.jsonl", conflicts)
    write_jsonl_atomic(destination / "split_repairs.jsonl", repairs)

    raw_root = ROOT / "data/raw/tdc" / directory / "v1"
    source_manifest = {
        "contract": "tdc_raw_source.v1",
        "task": task,
        "dataset": TASKS[task]["dataset"],
        "target_definition": TASKS[task]["target"],
        "upstream_commit": TASKS[task]["upstream"],
        **(
            {"source_location": TASKS[task]["source_location"]}
            if TASKS[task].get("source_location")
            else {}
        ),
        "files": files,
    }
    write_json_atomic(raw_root / "source_manifest.json", source_manifest)
    outputs = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "summary.json"}
    }
    summary = {
        "task": task,
        "task_directory": directory,
        "benchmark": "tdc",
        "version": VERSION,
        "release_id": RELEASE_ID,
        "contract": CONTRACT,
        "status": "complete",
        "target_definition": TASKS[task]["target"],
        "source_rows": len(source),
        "normalized_rows": len(normalized),
        "accepted_source_rows": sum(row["source_record_count"] for row in collapsed),
        "accepted_parents": len(detailed),
        "rejected_source_rows": len(rejected),
        "conflicting_source_rows": sum(
            len(row["source_record_ids"]) for row in conflicts
        ),
        "conflicting_parents": len(conflicts),
        "split_repairs": len(repairs),
        "splits": {
            split: {"n": len(rows), "positive": sum(row["Y"] for row in rows)}
            for split, rows in by_split.items()
        },
        "source_manifest": str(raw_root.relative_to(ROOT) / "source_manifest.json"),
        "source_manifest_sha256": sha256_file(raw_root / "source_manifest.json"),
        "outputs": outputs,
        "current_pointer_changed": False,
    }
    write_json_atomic(destination / "summary.json", summary)
    write_json_atomic(destination / "manifest.json", {**summary, "summary_sha256": sha256_file(destination / "summary.json")})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("all", *TASKS), default=["all"])
    parser.add_argument("--output-root", type=Path, default=TDC_ROOT)
    args = parser.parse_args(argv)
    tasks = list(TASKS) if args.tasks == ["all"] else list(dict.fromkeys(args.tasks))
    results = {task: build_task(task, output_root=args.output_root) for task in tasks}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
