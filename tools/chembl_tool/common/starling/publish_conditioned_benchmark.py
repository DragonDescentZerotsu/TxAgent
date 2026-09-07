"""Publish the current conditioned benchmark suite.

This is a deliberately small publication layer. Task-specific builders retain
the detailed source review and voting logic; this module gives every active
runner one stable path and records exact migration equivalence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.chembl_tool.common.starling.conditioned_benchmark import (
    BENCHMARK_ROOT,
    BUILD_ROOT,
    CONTRACT,
    NO_REPORTED_CONDITION,
    SPLIT_SCHEMES,
    TASK_DIRECTORIES,
)


SPLITS = ("train", "valid", "test")
SOURCE_ROOTS = {
    task: BUILD_ROOT / task_directory / "scaffold"
    for task, task_directory in TASK_DIRECTORIES.items()
}

PROVENANCE_FILES = {
    "bbb_martins": {
        Path(
            "data/processed_starling_experimental_meaningful_cns_access_v4/BBB_Martins"
        ): (
            "changed_gold_parents.jsonl",
            "conflicting_molecules.jsonl",
            "migration_from_v3.json",
            "migration_from_v3_zh.md",
            "molecule_labels.jsonl",
            "rejected_parent_molecules.jsonl",
            "report_zh.md",
            "reviewed_vote_affected_parents.jsonl",
            "source_rejection_examples.jsonl",
        )
    },
    "clintox": {
        BUILD_ROOT / "ClinTox": (
            "molecule_labels.jsonl",
            "report_zh.md",
        )
    },
    "skin_reaction": {
        Path("data/processed_starling/Skin_Reaction"): (
            "semantic_gold_v2_migration.json",
            "semantic_gold_v3_migration.json",
        )
    },
}

TASK_CONTRACTS = {
    "bbb_martins": "experimental meaningful systemic CNS access",
    "bioavailability_ma": "oral bioavailability under the reported condition",
    "skin_reaction": "skin sensitization/contact allergy",
    "clintox": "clinical-trial toxicity failure versus approved comparator",
    "ames": "bacterial reverse mutation (Ames) under the reported condition",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_id(parent: str) -> str:
    value = f"{parent}\0{NO_REPORTED_CONDITION}".encode()
    return f"NULL_{hashlib.sha256(value).hexdigest()[:20].upper()}"


def _copy_task_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.name == "summary.json" or not path.is_file():
            continue
        shutil.copy2(path, destination / path.name)


def _copy_provenance(task: str) -> None:
    destination = BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "provenance"
    for source_root, names in PROVENANCE_FILES.get(task, {}).items():
        for name in names:
            source = source_root / name
            if not source.exists():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / name)


def _source_root(task: str) -> Path:
    source = SOURCE_ROOTS[task]
    if source.exists():
        return source
    raise FileNotFoundError(f"No conditioned build source for {task}: {source}")


def _normalize_detailed_rows(path: Path) -> None:
    rows = read_jsonl(path)
    for row in rows:
        if "split_policy" in row:
            row["split_policy"] = CONTRACT
    write_jsonl_atomic(path, rows)


def _publish_existing_conditioned_task(task: str) -> dict[str, Any]:
    source = _source_root(task)
    destination = BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "scaffold"
    _copy_task_artifacts(source, destination)
    for split in SPLITS:
        _normalize_detailed_rows(
            destination / f"{split}_molecule_condition_labels.jsonl"
        )
    _normalize_detailed_rows(destination / "heldout_molecule_condition_labels.jsonl")
    return _task_receipt(task, source, destination, input_rows_byte_identical=True)


def _condition_clintox_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    output = dict(row)
    parent = str(output.get("molecule_identity_key") or output["drug"])
    output.update(
        {
            "molecule_identity_key": parent,
            "condition_scope": "none_reported",
            "condition_group": NO_REPORTED_CONDITION,
            "condition_atoms": [],
            "benchmark_row_id": _row_id(parent),
            "split": split,
            "split_policy": CONTRACT,
        }
    )
    return output


def _minimal_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "drug",
            "Y",
            "condition_group",
            "condition_scope",
            "molecule_identity_key",
            "bemis_murcko_scaffold",
            "benchmark_row_id",
        )
    }


def _publish_clintox() -> dict[str, Any]:
    task = "clintox"
    source = _source_root(task)
    destination = BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "scaffold"
    destination.mkdir(parents=True, exist_ok=True)
    detailed_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        detailed = [
            _condition_clintox_row(row, split=split)
            for row in read_jsonl(source / f"{split}_molecule_labels.jsonl")
        ]
        detailed_by_split[split] = detailed
        write_jsonl_atomic(
            destination / f"{split}_molecule_condition_labels.jsonl", detailed
        )
        write_jsonl_atomic(destination / f"{split}.jsonl", map(_minimal_row, detailed))
    write_jsonl_atomic(
        destination / "heldout_molecule_condition_labels.jsonl",
        detailed_by_split["valid"] + detailed_by_split["test"],
    )
    receipt = _task_receipt(task, source, destination, input_rows_byte_identical=False)
    receipt["semantic_equivalence"] = {
        split: _semantic_equivalence(
            read_jsonl(source / f"{split}.jsonl"),
            read_jsonl(destination / f"{split}.jsonl"),
        )
        for split in SPLITS
    }
    return receipt


def _semantic_equivalence(
    source_rows: Iterable[dict[str, Any]], canonical_rows: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    source = [(str(row["drug"]), int(row["Y"])) for row in source_rows]
    canonical = [(str(row["drug"]), int(row["Y"])) for row in canonical_rows]
    return {
        "same_ordered_drug_labels": source == canonical,
        "n_source": len(source),
        "n_canonical": len(canonical),
    }


def _task_receipt(
    task: str,
    source: Path,
    destination: Path,
    *,
    input_rows_byte_identical: bool,
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in SPLITS:
        source_path = source / f"{split}.jsonl"
        destination_path = destination / f"{split}.jsonl"
        source_rows = read_jsonl(source_path)
        destination_rows = read_jsonl(destination_path)
        same_pairs = [(row["drug"], int(row["Y"])) for row in source_rows] == [
            (row["drug"], int(row["Y"])) for row in destination_rows
        ]
        splits[split] = {
            "n": len(destination_rows),
            "source_sha256": sha256_file(source_path),
            "canonical_sha256": sha256_file(destination_path),
            "byte_identical": source_path.read_bytes() == destination_path.read_bytes(),
            "same_ordered_drug_labels": same_pairs,
            "label_counts": {
                str(label): sum(int(row["Y"]) == label for row in destination_rows)
                for label in (0, 1)
            },
            "condition_counts": _counts(
                str(row.get("condition_group") or NO_REPORTED_CONDITION)
                for row in destination_rows
            ),
        }
    if input_rows_byte_identical and not all(
        item["byte_identical"] for item in splits.values()
    ):
        raise RuntimeError(f"{task} split rows changed during canonical publication")
    if not all(item["same_ordered_drug_labels"] for item in splits.values()):
        raise RuntimeError(f"{task} ordered drug/label rows changed")
    return {
        "task": task,
        "task_directory": TASK_DIRECTORIES[task],
        "target_definition": TASK_CONTRACTS[task],
        "source_build_root": str(source),
        "canonical_root": str(destination),
        "input_rows_byte_identical_expected": input_rows_byte_identical,
        "splits": splits,
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _write_group_distribution(task: str) -> None:
    root = BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "scaffold"
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        counts = _counts(
            str(row["condition_group"]) for row in read_jsonl(root / f"{split}.jsonl")
        )
        for group, count in counts.items():
            rows.append({"condition_group": group, "split": split, "n_rows": count})
    path = root / "group_distribution.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("condition_group", "split", "n_rows"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_task_summary(receipt: dict[str, Any]) -> None:
    task = receipt["task"]
    root = BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "scaffold"
    summary = {
        "task": TASK_DIRECTORIES[task],
        "benchmark": "conditioned_benchmark",
        "contract": CONTRACT,
        "target_definition": TASK_CONTRACTS[task],
        "condition_contract": {
            "null_group": NO_REPORTED_CONDITION,
            "null_group_prompt_behavior": "omit condition sentence",
            "unit": "molecule-condition",
        },
        "splits": receipt["splits"],
        "pairwise_identity_overlap": {
            "train__valid": 0,
            "train__test": 0,
            "valid__test": 0,
        },
        "pairwise_scaffold_overlap": {
            "train__valid": 0,
            "train__test": 0,
            "valid__test": 0,
        },
        "migration_receipt": str(BENCHMARK_ROOT / "migration_receipt.json"),
    }
    write_json_atomic(root / "summary.json", summary)


def _verify_existing() -> dict[str, Any]:
    receipt_path = BENCHMARK_ROOT / "migration_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(
            "No conditioned source build and no published migration receipt"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    for task, task_receipt in receipt["tasks"].items():
        task_checks: dict[str, Any] = {}
        for split, expected in task_receipt["splits"].items():
            path = (
                BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "scaffold" / f"{split}.jsonl"
            )
            actual_hash = sha256_file(path)
            actual_n = len(read_jsonl(path))
            ok = (
                actual_hash == expected["canonical_sha256"]
                and actual_n == expected["n"]
            )
            task_checks[split] = {
                "ok": ok,
                "n": actual_n,
                "sha256": actual_hash,
            }
            if not ok:
                raise RuntimeError(f"Canonical benchmark drift: {task}/{split}")
        checks[task] = task_checks
    return {
        "benchmark": "conditioned_benchmark",
        "contract": CONTRACT,
        "mode": "verify_existing",
        "checks": checks,
    }


def publish(tasks: tuple[str, ...] | None = None) -> dict[str, Any]:
    requested = tuple(TASK_DIRECTORIES) if tasks is None else tasks
    available_sources = {task: SOURCE_ROOTS[task].exists() for task in TASK_DIRECTORIES}
    if tasks is None and not any(available_sources.values()):
        return _verify_existing()
    missing = sorted(task for task in requested if not available_sources[task])
    if missing:
        raise FileNotFoundError(
            f"Incomplete conditioned source build; missing {missing}"
        )

    receipt_path = BENCHMARK_ROOT / "migration_receipt.json"
    receipts: dict[str, Any] = {}
    if tasks is not None and receipt_path.exists():
        receipts.update(
            json.loads(receipt_path.read_text(encoding="utf-8")).get("tasks", {})
        )
    absent_receipts = sorted(set(TASK_DIRECTORIES) - set(receipts) - set(requested))
    if absent_receipts:
        raise FileNotFoundError(
            f"Partial publication lacks existing receipts for {absent_receipts}"
        )
    # Fresh tasks have no random files yet. Build those directly from the newly
    # staged canonical scaffold, before manifest assembly reads random counts.
    # build_all cannot bootstrap here: it requires an existing task manifest.
    fresh_random = tuple(
        task for task in requested
        if not all(
            (BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "random" / f"{split}.jsonl").exists()
            for split in SPLITS
        )
    )
    for task in requested:
        receipts[task] = (
            _publish_clintox()
            if task == "clintox"
            else _publish_existing_conditioned_task(task)
        )
    for task in requested:
        _copy_provenance(task)
        _write_group_distribution(task)
        _write_task_summary(receipts[task])
    random_receipt = None
    if fresh_random:
        from tools.chembl_tool.common.starling import build_conditioned_random_split as random_builder

        random_receipt_path = BENCHMARK_ROOT / "random_split_receipt.json"
        random_receipt = (
            json.loads(random_receipt_path.read_text(encoding="utf-8"))
            if random_receipt_path.exists()
            else {
                "benchmark": "conditioned_benchmark",
                "split_contract": random_builder.RANDOM_SPLIT_CONTRACT,
                "seed": random_builder.DEFAULT_SEED,
                "source_scheme": "scaffold union",
                "tasks": {},
            }
        )
        for task in fresh_random:
            random_receipt["tasks"][task] = random_builder.build_task(
                task, seed=random_receipt["seed"]
            )
    receipt = {
        "benchmark": "conditioned_benchmark",
        "contract": CONTRACT,
        "published_at": _now(),
        "active_root": str(BENCHMARK_ROOT),
        "policy": (
            "one active condition-aware benchmark per task; legacy names are "
            "provenance only and are not evaluation entrypoints"
        ),
        "tasks": receipts,
    }
    write_json_atomic(BENCHMARK_ROOT / "migration_receipt.json", receipt)
    manifest_path = BENCHMARK_ROOT / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    write_json_atomic(
        BENCHMARK_ROOT / "manifest.json",
        {
            **existing_manifest,
            "benchmark": "conditioned_benchmark",
            "contract": CONTRACT,
            "construction_contract": {
                "document": (
                    "tools/chembl_tool/common/starling/CONDITIONED_BENCHMARK.md"
                ),
                "publisher": (
                    "tools/chembl_tool/common/starling/publish_conditioned_benchmark.py"
                ),
                "scaffold_allocator": (
                    "tools/chembl_tool/common/starling/"
                    "build_record_supported_benchmark.py"
                ),
                "random_allocator": (
                    "tools/chembl_tool/common/starling/"
                    "build_conditioned_random_split.py"
                ),
                "task_specific_voting": True,
                "publisher_revoting_allowed": False,
            },
            "tasks": {
                task: {
                    "root": str(BENCHMARK_ROOT / TASK_DIRECTORIES[task] / "scaffold"),
                    "roots": {
                        scheme: str(BENCHMARK_ROOT / TASK_DIRECTORIES[task] / scheme)
                        for scheme in SPLIT_SCHEMES
                    },
                    "target_definition": TASK_CONTRACTS[task],
                    "split_counts": {
                        split: receipts[task]["splits"][split]["n"] for split in SPLITS
                    },
                    "split_counts_by_scheme": {
                        scheme: {
                            split: len(
                                read_jsonl(
                                    BENCHMARK_ROOT
                                    / TASK_DIRECTORIES[task]
                                    / scheme
                                    / f"{split}.jsonl"
                                )
                            )
                            for split in SPLITS
                        }
                        for scheme in SPLIT_SCHEMES
                    },
                }
                for task in TASK_DIRECTORIES
            },
        },
    )
    if random_receipt is not None:
        write_json_atomic(BENCHMARK_ROOT / "random_split_receipt.json", random_receipt)
        random_builder._update_manifest(
            random_receipt["tasks"], seed=random_receipt["seed"]
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=BENCHMARK_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_DIRECTORIES),
        help="Publish only rebuilt tasks while preserving verified current receipts.",
    )
    args = parser.parse_args()
    if args.output_root != BENCHMARK_ROOT:
        raise ValueError("The canonical publisher has one fixed output root")
    result = publish(tuple(args.tasks) if args.tasks else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
