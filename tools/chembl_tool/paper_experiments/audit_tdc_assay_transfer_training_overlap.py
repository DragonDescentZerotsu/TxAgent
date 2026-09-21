"""Audit TDC query exposure in the optimizer data of active assay rerankers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

import pyarrow.parquet as pq

from predict.retrieval.policies import bemis_murcko_scaffold
from predict.utils.json import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/assay_transfer_leakage/tdc_active_checkpoint_overlap_v1"
EXPECTED_QUERY_COUNTS = {
    "bbb_martins": {"valid": 197, "test": 530},
    "bioavailability_ma": {"valid": 64, "test": 128},
    "skin_reaction": {"valid": 40, "test": 82},
}
TASK_DIRS = {
    "bbb_martins": "BBB_Martins",
    "bioavailability_ma": "Bioavailability_Ma",
    "skin_reaction": "Skin_Reaction",
}
DIRECT_MODELS = {
    "bbb_martins": {
        "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-bbb-martins-best",
        "revision": "7678f7a1c43932f7612cfd49a8f4872d6e2f4cab",
        "checkpoint_step": 120,
        "dataset": "jiosephlee/context-conditioned-molecule-transfer-v10.3-bbb-martins-mixed-continuous-intern",
        "dataset_revision": "72fe16d4112cc9b1f7c1ea5199160ef8a05d372c",
    },
    "bioavailability_ma": {
        "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-0-2-bioavailability-ma-BEST",
        "revision": "29cc74f02df160b1f153edb700da66d1c30ef4ed",
        "checkpoint_step": 160,
        "dataset": "jiosephlee/context-conditioned-molecule-transfer-v10.3.0.2-bioavailability-ma-mixed-continuous-intern",
        "dataset_revision": "local:f39939f383c22937fed65ff34cdab461f9838ccda6ec91415795e01274ceb13f",
    },
    "skin_reaction": {
        "model": "jiosephlee/assay-transfer-tool-soft-v9.0.2-skin-reaction-mixed-continuous",
        "revision": "e4e894af28151760d2041275c5ecf136971c3925",
        "dataset": "jiosephlee/context-conditioned-molecule-transfer-v9.0.2-skin-reaction-mixed-continuous-intern",
        "dataset_revision": "94d0f6a7abc6f63f0f40b8b269343ca6ecda503f",
    },
}
DIRECT_CACHE_PROFILES = {
    "bbb_martins": "ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1",
    "bioavailability_ma": "ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1",
    "skin_reaction": "ranked_level_retrieval_tdc_v1_assay_skin_v9_v1",
}
LATER_MODELS = {
    "bbb_martins": {
        "L2": ("bbb_v24_1/hf/L2", "jiosephlee/assay-transfer-record-level-v24-1-bbb-martins-l2-intern", "b7c278103235fe7528ab11b6d7aca9f9ec3bdea2"),
        "L3": ("bbb_v24_1/hf/L3", "jiosephlee/assay-transfer-record-level-v24-1-bbb-martins-l3-intern", "a903fc6395ae25cad76d337d9dbae1548a6b5b51"),
        "L4": ("bbb_v24_1/hf/L4", "jiosephlee/assay-transfer-record-level-v24-1-bbb-martins-l4-intern", "e208d6f50982333c0df0f40291d08bd181e5a1db"),
    },
    "bioavailability_ma": {
        "L2": ("oral_v25/hf/L2", "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l2-intern", "b07ad9337a7f97a75db5770b521fd5a8779f8620"),
        "L3": ("oral_v25/hf/L3", "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l3-intern", "b6901b3e7eafca06835a2b9d5b60135ca2432521"),
        "L4": ("oral_v25/hf/L4", "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l4-intern", "2c14fa0784e03125a89d4d52636023b7e3cd07e2"),
        "L6": ("oral_v25/hf/L6", "jiosephlee/assay-transfer-record-level-v25-bioavailability-ma-l6-intern", "7df75e575fb60c297f02b4b21c75b470ae1ec261"),
    },
    "skin_reaction": {
        "L2+L3": ("v27/skin_reaction/hf/combined", "jiosephlee/assay-transfer-record-level-v27-skin-reaction-combined-intern", "1fa33cf3a7df7b6bd07a3b959f37502f951fc8ed"),
    },
}
LATER_CACHE_PROFILES = {
    "bbb_martins": "ranked_level_retrieval_tdc_v1_indirect_v1",
    "bioavailability_ma": "ranked_level_retrieval_tdc_v1_indirect_v1",
    "skin_reaction": "ranked_level_retrieval_skin_v27_tdc_v1",
}
@dataclass
class ParentDetail:
    query_occurrences: int = 0
    retrieval_occurrences: int = 0
    sources: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)


@dataclass
class TrainingProfile:
    pair_rows: int = 0
    query_parents: set[str] = field(default_factory=set)
    retrieval_parents: set[str] = field(default_factory=set)
    query_scaffolds: set[str] = field(default_factory=set)
    retrieval_scaffolds: set[str] = field(default_factory=set)
    details: dict[str, ParentDetail] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sqlite_queries(version_path: Path) -> list[dict[str, str]]:
    version = _read_json(version_path)
    database = version_path.with_name(str(version["database"]))
    if sha256_file(database) != version["database_sha256"]:
        raise ValueError(f"Query database hash mismatch: {database}")
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute(
            "SELECT benchmark_row_id,drug,query_parent_id,query_parent_smiles "
            "FROM queries ORDER BY benchmark_row_id"
        ).fetchall()
    finally:
        connection.close()
    return [dict(zip(("benchmark_row_id", "drug", "parent_id", "parent_smiles"), row)) for row in rows]


def _load_queries(task: str, split: str, cache_root: Path) -> list[dict[str, Any]]:
    path = REPO_ROOT / f"data/gold_labels/TDC/{TASK_DIRS[task]}/v1/scaffold/{split}_molecule_condition_labels.jsonl"
    version_path = cache_root / f"scaffold/{split}/L1/VERSION.json"
    version = _read_json(version_path)
    if sha256_file(path) != version["inputs"]["query_sha256"]:
        raise ValueError(f"TDC query input differs from cache: {path}")
    frozen = {row["benchmark_row_id"]: row for row in _sqlite_queries(version_path)}
    queries = _read_jsonl(path)
    if len(queries) != EXPECTED_QUERY_COUNTS[task][split] or len(frozen) != len(queries):
        raise ValueError(f"Unexpected TDC query count: {task}/{split}")
    if len({row["molecule_identity_key"] for row in queries}) != len(queries):
        raise ValueError(f"TDC query parents are not unique: {task}/{split}")
    for row in queries:
        cached = frozen.get(str(row["benchmark_row_id"]))
        identity = row["molecule_identity"]
        scaffold = str(row.get("bemis_murcko_scaffold") or "")
        if cached is None or cached != {
            "benchmark_row_id": str(row["benchmark_row_id"]),
            "drug": str(row["drug"]),
            "parent_id": str(row["molecule_identity_key"]),
            "parent_smiles": str(identity["parent_smiles"]),
        }:
            raise ValueError(f"TDC cache query differs from source: {task}/{split}/{row['benchmark_row_id']}")
        if scaffold != bemis_murcko_scaffold(str(identity["parent_smiles"])):
            raise ValueError(f"Stored TDC scaffold differs from renderer policy: {row['benchmark_row_id']}")
    return queries


def _scan_training(path: Path, target_parents: set[str], *, include_scaffolds: bool) -> TrainingProfile:
    schema = pq.read_schema(path)
    columns = ["query_parent_id", "retrieval_parent_id", "source_id"]
    if "level_family" in schema.names:
        columns.append("level_family")
    if include_scaffolds:
        columns.extend(("query_scaffold", "retrieval_scaffold"))
    profile = TrainingProfile()
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=131_072):
        data = batch.to_pydict()
        profile.pair_rows += batch.num_rows
        families = data.get("level_family", [None] * batch.num_rows)
        for index in range(batch.num_rows):
            source = str(data["source_id"][index] or "")
            family = str(families[index] or "")
            for role in ("query", "retrieval"):
                parent = str(data[f"{role}_parent_id"][index])
                getattr(profile, f"{role}_parents").add(parent)
                if include_scaffolds:
                    scaffold = str(data[f"{role}_scaffold"][index] or "")
                    if scaffold:
                        getattr(profile, f"{role}_scaffolds").add(scaffold)
                if parent not in target_parents:
                    continue
                detail = profile.details.setdefault(parent, ParentDetail())
                setattr(detail, f"{role}_occurrences", getattr(detail, f"{role}_occurrences") + 1)
                if source:
                    detail.sources.add(source)
                if family:
                    detail.families.add(family)
    return profile


def _merge_profiles(profiles: list[TrainingProfile]) -> TrainingProfile:
    merged = TrainingProfile()
    for profile in profiles:
        merged.pair_rows += profile.pair_rows
        for name in ("query_parents", "retrieval_parents", "query_scaffolds", "retrieval_scaffolds"):
            getattr(merged, name).update(getattr(profile, name))
        for parent, detail in profile.details.items():
            target = merged.details.setdefault(parent, ParentDetail())
            target.query_occurrences += detail.query_occurrences
            target.retrieval_occurrences += detail.retrieval_occurrences
            target.sources.update(detail.sources)
            target.families.update(detail.families)
    return merged


def _training_rows(stage: str, level: str, task: str, profile: TrainingProfile, path: Path) -> dict[str, Any]:
    return {
        "task": task,
        "stage": stage,
        "level": level,
        "train_pair_rows": profile.pair_rows,
        "query_parent_count": len(profile.query_parents),
        "retrieval_parent_count": len(profile.retrieval_parents),
        "parent_union_count": len(profile.query_parents | profile.retrieval_parents),
        "query_nonempty_scaffold_count": len(profile.query_scaffolds),
        "retrieval_nonempty_scaffold_count": len(profile.retrieval_scaffolds),
        "nonempty_scaffold_union_count": len(profile.query_scaffolds | profile.retrieval_scaffolds),
        "train_parquet": str(path),
        "train_parquet_sha256": sha256_file(path),
    }


def _member_rows(
    task: str,
    split: str,
    stage: str,
    level: str,
    queries: list[dict[str, Any]],
    profile: TrainingProfile,
) -> list[dict[str, Any]]:
    train_parents = profile.query_parents | profile.retrieval_parents
    train_scaffolds = profile.query_scaffolds | profile.retrieval_scaffolds
    rows = []
    for query in queries:
        parent = str(query["molecule_identity_key"])
        scaffold = str(query.get("bemis_murcko_scaffold") or "")
        parent_overlap = parent in train_parents
        scaffold_overlap = stage == "direct" and bool(scaffold) and scaffold in train_scaffolds
        if not parent_overlap and not scaffold_overlap:
            continue
        detail = profile.details.get(parent, ParentDetail())
        rows.append({
            "task": task,
            "split": split,
            "stage": stage,
            "level": level,
            "benchmark_row_id": str(query["benchmark_row_id"]),
            "parent_id": parent,
            "bemis_murcko_scaffold": scaffold,
            "gold_label": int(query["Y"]),
            "parent_overlap": int(parent_overlap),
            "scaffold_overlap": int(scaffold_overlap),
            "query_role_occurrences": detail.query_occurrences,
            "retrieval_role_occurrences": detail.retrieval_occurrences,
            "training_source_ids": ",".join(sorted(detail.sources)),
            "training_level_families": ",".join(sorted(detail.families)),
        })
    return rows


def _summary_rows(
    task: str,
    split: str,
    stage: str,
    level: str,
    queries: list[dict[str, Any]],
    profile: TrainingProfile,
    members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_members = [row for row in members if row["parent_overlap"]]
    query_scaffolds = [str(row.get("bemis_murcko_scaffold") or "") for row in queries]
    rows = [{
        "task": task,
        "split": split,
        "stage": stage,
        "level": level,
        "identity_kind": "parent",
        "query_count": len(queries),
        "eligible_query_count": len(queries),
        "unique_query_identity_count": len({row["molecule_identity_key"] for row in queries}),
        "unique_training_identity_count": len(profile.query_parents | profile.retrieval_parents),
        "overlap_query_count": len(parent_members),
        "overlap_unique_identity_count": len({row["parent_id"] for row in parent_members}),
        "overlap_fraction": len(parent_members) / len(queries),
        "overlap_gold_0_count": sum(row["gold_label"] == 0 for row in parent_members),
        "overlap_gold_1_count": sum(row["gold_label"] == 1 for row in parent_members),
    }]
    if stage == "direct":
        scaffold_members = [row for row in members if row["scaffold_overlap"]]
        eligible = sum(bool(value) for value in query_scaffolds)
        rows.append({
            "task": task,
            "split": split,
            "stage": stage,
            "level": level,
            "identity_kind": "bemis_murcko_scaffold_nonempty",
            "query_count": len(queries),
            "eligible_query_count": eligible,
            "unique_query_identity_count": len({value for value in query_scaffolds if value}),
            "unique_training_identity_count": len(profile.query_scaffolds | profile.retrieval_scaffolds),
            "overlap_query_count": len(scaffold_members),
            "overlap_unique_identity_count": len({row["bemis_murcko_scaffold"] for row in scaffold_members}),
            "overlap_fraction": len(scaffold_members) / eligible if eligible else 0.0,
            "overlap_gold_0_count": sum(row["gold_label"] == 0 for row in scaffold_members),
            "overlap_gold_1_count": sum(row["gold_label"] == 1 for row in scaffold_members),
        })
    return rows


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _validate_direct_training_input(
    task: str,
    arguments: dict[str, Any],
    expected: dict[str, Any],
    dataset_root: Path,
    train_path: Path,
) -> None:
    if arguments["dataset_split"] != "train":
        raise ValueError(f"Direct optimizer split is not train: {task}")
    if task == "bioavailability_ma":
        local_data = Path(str(arguments["local_data"])).resolve()
        if local_data != dataset_root.resolve():
            raise ValueError(f"Direct optimizer local dataset differs from audit input: {task}")
        if expected["dataset_revision"] != f"local:{sha256_file(train_path)}":
            raise ValueError(f"Direct optimizer local dataset hash differs from active cache: {task}")
        return
    if (
        arguments["hub_dataset_id"] != expected["dataset"]
        or arguments["hub_dataset_revision"] != expected["dataset_revision"]
    ):
        raise ValueError(f"Direct optimizer Hub dataset differs from active cache: {task}")


def _git_receipt(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip())
    return {"path": str(path), "commit": commit, "dirty": dirty}


def _render_report(overlap: list[dict[str, Any]]) -> str:
    parent_overlap = {
        (row["task"], row["split"], row["stage"], row["level"]): row
        for row in overlap if row["identity_kind"] == "parent"
    }
    scaffold_overlap = {
        (row["task"], row["split"]): row
        for row in overlap if row["identity_kind"] == "bemis_murcko_scaffold_nonempty"
    }
    lines = [
        "# Active assay-transfer optimizer overlap with TDC v1",
        "",
        "## Conclusion",
        "",
        "The active assay-transfer checkpoints are not molecule-disjoint from the TDC test queries. "
        "This is a high-confidence molecular-exposure finding, not by itself proof that a checkpoint "
        "memorized or used a TDC target label.",
        "",
        "Gold 0 and Gold 1 below are the TDC v1 benchmark `Y` values of the exposed queries. "
        "The table does not use optional labels stored in optimizer pair rows.",
        "",
        "## TDC test summary",
        "",
        "| Task | Direct parent overlap (Gold 0 / 1) | Direct scaffold overlap (Gold 0 / 1) | Any L2+ parent overlap (Gold 0 / 1) |",
        "|---|---:|---:|---:|",
    ]
    for task in ("bbb_martins", "bioavailability_ma", "skin_reaction"):
        direct_parent = parent_overlap[(task, "test", "direct", "L1")]
        direct_scaffold = scaffold_overlap[(task, "test")]
        later_parent = parent_overlap[(task, "test", "later", "ANY_L2_PLUS")]
        lines.append(
            f"| {task} | {direct_parent['overlap_query_count']}/{direct_parent['query_count']} "
            f"({direct_parent['overlap_gold_0_count']} / {direct_parent['overlap_gold_1_count']}) | "
            f"{direct_scaffold['overlap_query_count']}/{direct_scaffold['eligible_query_count']} "
            f"({direct_scaffold['overlap_gold_0_count']} / {direct_scaffold['overlap_gold_1_count']}) | "
            f"{later_parent['overlap_query_count']}/{later_parent['query_count']} "
            f"({later_parent['overlap_gold_0_count']} / {later_parent['overlap_gold_1_count']}) |"
        )
    lines.extend([
        "",
        "Every exposed query has a TDC Gold label. Counts in parentheses are `Gold 0 / Gold 1`.",
        "",
        "## Artifact guide",
        "",
        "- `overlap_summary.tsv`: denominators, unique identities, affected queries, and rates.",
        "- `overlap_members.tsv`: every exposed query with its TDC Gold label, roles, and sources.",
        "- `training_distribution.tsv`: optimizer-row and unique-identity distributions.",
        "- `checkpoint_lineage.tsv` and `manifest.json`: pinned models, datasets, hashes, and repositories.",
        "",
        "## Interpretation boundary",
        "",
        "These TDC results must not be described as molecule-disjoint or fully out-of-distribution. "
        "A separate causal or retraining study would be required to estimate how much this exposure changed performance.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--starling-root", type=Path, default=REPO_ROOT.parent / "starling_assay_transfer")
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT.parent / "therapeutic-tuning")
    parser.add_argument("--skin-v9-dataset-root", type=Path)
    args = parser.parse_args(argv)

    cache_base = REPO_ROOT / "data/caches/assay_reranking/active"
    query_roots = {
        task: cache_base / profile / task for task, profile in DIRECT_CACHE_PROFILES.items()
    }
    queries = {
        (task, split): _load_queries(task, split, query_roots[task])
        for task in TASK_DIRS for split in ("valid", "test")
    }
    target_parents = {
        task: {
            str(row["molecule_identity_key"])
            for split in ("valid", "test") for row in queries[(task, split)]
        }
        for task in TASK_DIRS
    }

    direct_roots = {
        "bbb_martins": args.starling_root / "assay_transfer/context_conditioned/artifacts/v10/hf/v10_3/BBB_Martins/mixed_continuous",
        "bioavailability_ma": args.starling_root / "assay_transfer/context_conditioned/artifacts/v10/hf/v10_3_0_2/Bioavailability_Ma/mixed_continuous",
    }
    if args.skin_v9_dataset_root:
        direct_roots["skin_reaction"] = args.skin_v9_dataset_root
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        direct_roots["skin_reaction"] = (
            hf_home / "hub/datasets--jiosephlee--context-conditioned-molecule-transfer-v9.0.2-skin-reaction-mixed-continuous-intern"
            "/snapshots/94d0f6a7abc6f63f0f40b8b269343ca6ecda503f"
        )
    training_manifests = {
        "bbb_martins": args.training_root / "results/sft/train/2026-09-15/intern-s1-mini_context-conditioned-molecule-transfer-v10.3-pinned-bbb-martins-mixed-continuous-soft-f1at5-b32x4-ga1-lr2e-5-packed-10ep/manifest.json",
        "bioavailability_ma": args.training_root / "results/sft/train/2026-09-16/intern-s1-mini_context-conditioned-molecule-transfer-v10.3.0.2-bioavailability-ma-mixed-continuous-soft-f1at5-b32x4-ga1-lr2e-5-packed-10ep/manifest.json",
        "skin_reaction": args.training_root / "results/sft/train/2026-09-05/intern-s1-mini_context-conditioned-molecule-transfer-v9-official-pinned-skin-reaction-mixed-continuous-soft-f1at5-b32x4-ga1-lr2e-5-packed-15ep/manifest.json",
    }

    overlap_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    later_profiles: dict[str, list[TrainingProfile]] = defaultdict(list)

    for task in TASK_DIRS:
        expected = DIRECT_MODELS[task]
        release_index_path = query_roots[task] / "RELEASE_INDEX.json"
        release_index = _read_json(release_index_path)
        if any(release_index["model"].get(key) != expected.get(key) for key in ("model", "revision", "checkpoint_step")):
            raise ValueError(f"Active direct model differs from audit contract: {task}")
        training_manifest = _read_json(training_manifests[task])
        dataset_root = direct_roots[task]
        dataset_manifest_path = dataset_root / "manifest.json"
        dataset_manifest = _read_json(dataset_manifest_path)
        train_path = dataset_root / "train/data.parquet"
        if sha256_file(train_path) != dataset_manifest["artifacts"]["train/data.parquet"]:
            raise ValueError(f"Direct training Parquet differs from manifest: {task}")
        _validate_direct_training_input(
            task, training_manifest["arguments"], expected, dataset_root, train_path
        )
        profile = _scan_training(train_path, target_parents[task], include_scaffolds=True)
        distribution_rows.append(_training_rows("direct", "L1", task, profile, train_path))
        lineage_rows.append({
            "task": task,
            "stage": "direct",
            "level": "L1",
            "cache_profile": DIRECT_CACHE_PROFILES[task],
            "model": expected["model"],
            "model_revision": expected["revision"],
            "checkpoint_step": expected.get("checkpoint_step", ""),
            "dataset": expected["dataset"],
            "dataset_revision": expected["dataset_revision"],
            "train_parquet": str(train_path),
            "train_parquet_sha256": sha256_file(train_path),
            "dataset_manifest": str(dataset_manifest_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "training_manifest": str(training_manifests[task]),
            "training_manifest_sha256": sha256_file(training_manifests[task]),
        })
        for split in ("valid", "test"):
            members = _member_rows(task, split, "direct", "L1", queries[(task, split)], profile)
            summaries = _summary_rows(task, split, "direct", "L1", queries[(task, split)], profile, members)
            member_rows.extend(members)
            overlap_rows.extend(summaries)

    record_root = args.starling_root / "assay_transfer/record_level/artifacts"
    for task, levels in LATER_MODELS.items():
        cache_root = cache_base / LATER_CACHE_PROFILES[task] / task
        for level, (relative, dataset, dataset_revision) in levels.items():
            dataset_root = record_root / relative
            dataset_manifest_path = dataset_root / "manifest.json"
            dataset_manifest = _read_json(dataset_manifest_path)
            train_path = dataset_root / "train/data.parquet"
            if sha256_file(train_path) != dataset_manifest["artifacts"]["train/data.parquet"]:
                raise ValueError(f"Later-level training Parquet differs from manifest: {task}/{level}")
            cache_level = "L2" if task == "skin_reaction" else level
            models = []
            for split in ("valid", "test"):
                version_path = cache_root / f"scaffold/{split}/{cache_level}/VERSION.json"
                model = _read_json(version_path)["model"]
                if model.get("dataset") != dataset or model.get("dataset_revision") != dataset_revision:
                    raise ValueError(f"Later-level dataset differs from active cache: {task}/{level}/{split}")
                models.append(model)
            if models[0] != models[1]:
                raise ValueError(f"Valid/test later-level models differ: {task}/{level}")
            profile = _scan_training(train_path, target_parents[task], include_scaffolds=False)
            later_profiles[task].append(profile)
            distribution_rows.append(_training_rows("later", level, task, profile, train_path))
            lineage_rows.append({
                "task": task,
                "stage": "later",
                "level": level,
                "cache_profile": LATER_CACHE_PROFILES[task],
                "model": models[0]["model"],
                "model_revision": models[0]["revision"],
                "checkpoint_step": models[0].get("checkpoint_step", ""),
                "dataset": dataset,
                "dataset_revision": dataset_revision,
                "train_parquet": str(train_path),
                "train_parquet_sha256": sha256_file(train_path),
                "dataset_manifest": str(dataset_manifest_path),
                "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
                "training_manifest": "",
                "training_manifest_sha256": "",
            })
            for split in ("valid", "test"):
                members = _member_rows(task, split, "later", level, queries[(task, split)], profile)
                summaries = _summary_rows(task, split, "later", level, queries[(task, split)], profile, members)
                member_rows.extend(members)
                overlap_rows.extend(summaries)

    for task, profiles in later_profiles.items():
        merged = _merge_profiles(profiles)
        for split in ("valid", "test"):
            members = _member_rows(task, split, "later", "ANY_L2_PLUS", queries[(task, split)], merged)
            summaries = _summary_rows(
                task, split, "later", "ANY_L2_PLUS", queries[(task, split)], merged, members
            )
            member_rows.extend(members)
            overlap_rows.extend(summaries)

    sort_key = lambda row: (row["task"], row.get("split", ""), row["stage"], row["level"], row.get("benchmark_row_id", ""), row.get("identity_kind", ""))
    for rows in (overlap_rows, member_rows, distribution_rows, lineage_rows):
        rows.sort(key=sort_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(args.output_dir / "overlap_summary.tsv", overlap_rows)
    _write_tsv(args.output_dir / "overlap_members.tsv", member_rows)
    _write_tsv(args.output_dir / "training_distribution.tsv", distribution_rows)
    _write_tsv(args.output_dir / "checkpoint_lineage.tsv", lineage_rows)
    _write_text(args.output_dir / "report.md", _render_report(overlap_rows))
    outputs = [
        "overlap_summary.tsv", "overlap_members.tsv", "training_distribution.tsv",
        "checkpoint_lineage.tsv", "report.md",
    ]
    input_paths = {
        Path(row[field])
        for row in lineage_rows
        for field in ("train_parquet", "dataset_manifest", "training_manifest")
        if row[field]
    }
    for task in TASK_DIRS:
        input_paths.add(query_roots[task] / "RELEASE_INDEX.json")
        for split in ("valid", "test"):
            input_paths.add(
                REPO_ROOT
                / f"data/gold_labels/TDC/{TASK_DIRS[task]}/v1/scaffold/{split}_molecule_condition_labels.jsonl"
            )
            input_paths.add(query_roots[task] / f"scaffold/{split}/L1/VERSION.json")
        later_root = cache_base / LATER_CACHE_PROFILES[task] / task
        input_paths.add(later_root / "RELEASE_INDEX.json")
        for level in LATER_MODELS[task]:
            cache_level = "L2" if task == "skin_reaction" else level
            for split in ("valid", "test"):
                input_paths.add(later_root / f"scaffold/{split}/{cache_level}/VERSION.json")
    manifest = {
        "schema_version": "tdc_assay_transfer_training_overlap.v1",
        "status": "complete",
        "benchmark": "tdc_v1",
        "scope": "active_direct_and_assay_scored_later_level_optimizer_train",
        "label_contract": "tdc_v1_benchmark_y",
        "repositories": {
            "txagent": _git_receipt(REPO_ROOT),
            "starling_assay_transfer": _git_receipt(args.starling_root),
            "therapeutic_tuning": _git_receipt(args.training_root),
        },
        "inputs": {str(path): sha256_file(path) for path in sorted(input_paths)},
        "outputs": {name: sha256_file(args.output_dir / name) for name in outputs},
    }
    _write_text(
        args.output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
