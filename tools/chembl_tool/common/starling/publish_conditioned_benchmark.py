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
import os
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
    "dili": "reported clinically meaningful human DILI outcome under the stated exposure and population; a negative is not universal absence of DILI risk",
    "carcinogens": "reported any-site carcinogenic hazard under the reported species and exposure",
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
    if set(requested) & {'dili', 'carcinogens'}:
        raise ValueError("DILI/Carcinogens require --reviewed-release; generic publication "
                         "cannot replace their frozen Starling-only cohort")
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


def publish_reviewed_release(task: str, release_path: Path) -> dict[str, Any]:
    """Promote a fully validated source release without revoting or resplitting."""
    from tools.chembl_tool.common.build_runtime import sha256_file
    from tools.chembl_tool.common.starling.build_conditioned_random_split import _union_hash
    release = json.loads(release_path.read_text())
    if release['status'] != 'records_indices_and_card_links_validated':
        raise ValueError('Source release is not fully validated')
    if release['heldout_prefilter_levels'] != [1]:
        raise ValueError('Reviewed release must use the shared L1-only prefilter')
    source = Path(release['benchmark_root'])
    if source.name != TASK_DIRECTORIES[task]: raise ValueError('Release task mismatch')
    root = release_path.parent
    receipt_path = root/'canonical_publication.json'
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        for path, digest in previous['published_files'].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError('Published release drift: '+path)
        return previous
    for path, digest in release['validation_inputs'].items():
        if sha256_file(Path(path)) != digest: raise ValueError('Stale validation: '+path)
    for key, file in [('source_gold_sha256',Path(release['source_gold'])),
                      ('records_sha256',Path(release['records']))]:
        if sha256_file(file) != release[key]: raise ValueError('Stale release: '+str(file))
    unions = []
    for scheme in SPLIT_SCHEMES:
        rows = {s:read_jsonl(source/scheme/f'{s}_molecule_condition_labels.jsonl') for s in SPLITS}
        unions.append(_union_hash(r for values in rows.values() for r in values))
        parents = [{r['molecule_identity_key'] for r in values} for values in rows.values()]
        if any(parents[i]&parents[j] for i in range(3) for j in range(i)):
            raise ValueError('Parent overlap')
        expected = read_jsonl(root/'heldout'/f'{scheme}.jsonl')
        cohort = rows['valid']+rows['test']
        key = lambda r:(r['benchmark_row_id'],r['drug'],r['Y'])
        if sorted(map(key,expected)) != sorted(map(key,cohort)):
            raise ValueError('Heldout query cohort changed')
        index = Path(release['indices'][scheme]['path'])
        check=json.loads((index/'identity_validation.json').read_text())
        if check['status']!='passed' or check['heldout_prefilter_levels']!=[1]:
            raise ValueError('Index validation failed')
        if check['manifest_sha256']!=sha256_file(index/'manifest.json'):
            raise ValueError('Index manifest changed')
        for name,key in [('assay_neighbor_index.pkl','index_sha256'),('assay_molecule_evidence.jsonl','evidence_sha256')]:
            if sha256_file(index/name)!=release['indices'][scheme][key]:raise ValueError('Index artifact changed')
    if unions[0] != unions[1]: raise ValueError('Split schemes use different labels')
    destination = BENCHMARK_ROOT/TASK_DIRECTORIES[task]
    snapshot = root/'previous_active'
    if snapshot.exists(): raise ValueError('Inspect incomplete publication snapshot: '+str(snapshot))
    snapshot.mkdir()
    shutil.copytree(destination,snapshot/'benchmark')
    registries = [BENCHMARK_ROOT/n for n in ('manifest.json','migration_receipt.json','random_split_receipt.json')]
    retrieval_registry=Path('tools/chembl_tool/paper_experiments/current_starling_retrieval.json')
    registries.append(retrieval_registry)
    for p in registries: shutil.copy2(p,snapshot/p.name)
    # Replace whole split directories so stale audit files cannot appear current.
    shutil.rmtree(destination)
    destination.mkdir()
    for scheme in SPLIT_SCHEMES:
        shutil.copytree(source/scheme,destination/scheme,ignore=shutil.ignore_patterns('.*','*.lock'))
        shutil.copy2(root/'heldout'/f'{scheme}.jsonl',destination/scheme/'heldout_molecule_condition_labels.jsonl')
    runtime=Path('outputs/paper/starling_conditioned_assay_family_curve_v1')
    for source_dir,target,backup in [(Path(release['catalog']),runtime/'family_catalogs'/task,snapshot/'catalog'),
        *[(Path(release['indices'][s]['path']),runtime/'indices'/task/s,snapshot/('index_'+s)) for s in SPLIT_SCHEMES]]:
        if target.exists(): target.rename(backup)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source_dir,target,copy_function=os.link)
    counts={s:{split:len(read_jsonl(destination/s/f'{split}.jsonl')) for split in SPLITS} for s in SPLIT_SCHEMES}
    manifest=json.loads(registries[0].read_text())
    manifest['tasks'][task]={
        'root':str(destination/'scaffold'),'roots':{s:str(destination/s) for s in SPLIT_SCHEMES},
        'target_definition':release.get('target_definition',TASK_CONTRACTS[task]),'split_counts':counts['scaffold'],'split_counts_by_scheme':counts,
        'source_gold_root':str(Path(release['source_gold']).parent),'cohort':'starling_only_reviewed',
        'identity_contract':'new_task_tautomer_identity.v2','publication_receipt':str(receipt_path),
        'evaluation_status':'fresh_evaluation_pending','retrieval_validation':str(release_path),
        'test_previously_used_for_validation':release.get('test_previously_used_for_validation',True),
        'evaluation_policy':release.get('evaluation_policy','Frozen method; valid/test concurrent by explicit user instruction; no test-driven tuning.')}
    write_json_atomic(registries[0],manifest)
    write_json_atomic(destination/'manifest.json',manifest['tasks'][task])
    migration=json.loads(registries[1].read_text())
    migration['tasks'][task]={**_task_receipt(task,source/'scaffold',destination/'scaffold',input_rows_byte_identical=True),
        'publication_receipt':str(receipt_path),'cohort':'starling_only_reviewed',
        'target_definition':manifest['tasks'][task]['target_definition']}
    write_json_atomic(registries[1],migration)
    random=json.loads(registries[2].read_text())
    random['tasks'][task]={**json.loads((destination/'random/summary.json').read_text()),
        'root':str(destination/'random'),'same_molecule_condition_rows_and_labels':True,
        'source_union_sha256':unions[0],'random_union_sha256':unions[1],'publication_receipt':str(receipt_path)}
    write_json_atomic(registries[2],random)
    registry=json.loads(retrieval_registry.read_text());entry=registry['tasks'][task]
    entry.update(canonical_records_sha256=release['records_sha256'],overlay_records_sha256=release['records_sha256'],
        source_records=release['records'],catalog_sha256=sha256_file(Path(release['catalog'])/'family_assays.jsonl'),
        filter_scope_field=release['filter_scope_field'],filter_scope_value=release['filter_scope_value'])
    for scheme in SPLIT_SCHEMES:
        for key in ('index_sha256','evidence_sha256'):entry['indices'][scheme][key]=release['indices'][scheme][key]
    for entry in registry.get('source_releases',{}).values():
        if entry['manifest']==str(release_path):entry.update(canonical_experiment_defaults_changed=True,publication_receipt=str(receipt_path))
    write_json_atomic(retrieval_registry,registry)
    result={'status':'passed','task':task,'release':str(release_path),'release_sha256':sha256_file(release_path),
        'published_at':_now(),'previous_active':str(snapshot),'split_counts':counts,
        'heldout_metadata_only_enrichment':True,'gold_and_query_labels_unchanged':True,
        'unchanged_relative_to':'validated_staged_release, not the previous active cohort',
        'published_files':{str(p):sha256_file(p) for s in SPLIT_SCHEMES for p in (destination/s).iterdir() if p.is_file()}}
    write_json_atomic(receipt_path,result)
    return result


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
    parser.add_argument('--reviewed-release',type=Path,help='Promote one validated source/gold/index release atomically by artifact, without rebuilding labels')
    args = parser.parse_args()
    if args.output_root != BENCHMARK_ROOT:
        raise ValueError("The canonical publisher has one fixed output root")
    if args.reviewed_release:
        if not args.tasks or len(args.tasks)!=1:parser.error('--reviewed-release requires one task')
        print(json.dumps(publish_reviewed_release(args.tasks[0],args.reviewed_release),indent=2))
        return 0
    result = publish(tuple(args.tasks) if args.tasks else None)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
