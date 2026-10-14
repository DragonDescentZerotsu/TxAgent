"""Publish TDC-v2 L2+ caches with all-or-nothing assay ranking per query and level.

Exact v1 score keys may be reused across the old valid/test boundary. No model
inference is performed; a query-level cache miss selects the full Morgan order.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import sqlite3

from predict.utils.json import sha256_file

from . import build_ranked_uid_retrieval as builder
from . import build_safety_v27_ranked_retrieval as safety
from . import build_skin_v27_ranked_retrieval as skin
from . import build_tdc_indirect_ranked_retrieval as indirect
from .runtime import cache_profile_root


TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction", "ames", "dili", "carcinogens")
POLICY = {
    "schema_version": "tdc_v2_query_pure.v1",
    "scope": "query_level",
    "complete_order": "assay_transfer",
    "incomplete_order": "morgan",
    "incomplete_prompt_scores": "hidden",
}
SOURCE_PROFILES = {
    "bbb_martins": "ranked_level_retrieval_tdc_v1_indirect_v2",
    "bioavailability_ma": "ranked_level_retrieval_tdc_v1_indirect_v2",
    "skin_reaction": "flat_v5/tdc_v1/skin_reaction/l2plus/assay_transfer/v27/training_candidate_copy_shared_parent100_v2",
    "ames": "flat_v5/tdc_v1/ames/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent40_partial_snapshot_v1",
    "dili": "flat_v5/tdc_v1/dili/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
    "carcinogens": "flat_v5/tdc_v1/carcinogens/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
}
PROFILE = "flat_v5/tdc_v2/{task}/l2plus/query_pure/reused_v1/original"
DILI_PARENT_SUCCESSOR = "flat_v5/tdc_v2/dili/l2plus/query_pure/parent_disjoint_v2/original"
AMES_PARENT100_SUCCESSOR = "flat_v5/tdc_v2/ames/l2plus/query_pure/parent100_v2/original"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _configure(
    task: str, *, dili_parent_successor: bool = False, ames_parent100: bool = False,
) -> tuple[Path, Path, Path]:
    if dili_parent_successor and task != "dili":
        raise ValueError("Parent-disjoint successor is DILI-specific")
    if ames_parent100 and task != "ames":
        raise ValueError("Parent-100 successor is AMES-specific")
    source_root = cache_profile_root(SOURCE_PROFILES[task])
    source_index_path = source_root / task / "RELEASE_INDEX.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    if source_index.get("status") != "complete" or source_index.get("task_id") != task:
        raise ValueError(f"Incomplete source cache: {source_index_path}")
    evidence_manifest = (source_index_path.parent / source_index["evidence"]["manifest"]).resolve()
    if sha256_file(evidence_manifest) != source_index["evidence"]["manifest_sha256"]:
        raise ValueError(f"Source projection manifest changed: {evidence_manifest}")
    if task in {"bbb_martins", "bioavailability_ma"}:
        indirect._configure()
    elif task == "skin_reaction":
        skin._configure("tdc", training_template=True)
        builder.BASE_RANKING_ROOT = None
    else:
        safety._configure(task, "tdc", 40 if task == "ames" and not ames_parent100 else 100,
                          snapshot=task == "ames" and not ames_parent100)
    builder.PROFILE = (DILI_PARENT_SUCCESSOR if dili_parent_successor else
                       AMES_PARENT100_SUCCESSOR if ames_parent100 else PROFILE.format(task=task))
    builder.FILTER_PARENT_RELATIONS = dili_parent_successor
    builder.QUERY_BENCHMARK = "tdc-v2"
    builder.LABEL_RELEASE = {"benchmark": "tdc", "version": "v2"}
    # Frozen query/candidate rows carry exact scores without rerendering prompts.
    builder.SCORING_FAMILY = "morgan_only"
    builder.SCORE_REUSE_ROOTS = ()
    builder.TRUST_PREDECESSOR_ROWS = False
    output_root = cache_profile_root(builder.PROFILE)
    return source_root, evidence_manifest, output_root


def _transplant_frozen_scores(
    task: str, subset: str, level: str, connection: sqlite3.Connection,
    manifest: dict, source_root: Path,
) -> None:
    sources = []
    with ExitStack() as stack:
        for source_subset in ("valid", "test"):
            path = source_root / task / "scaffold" / source_subset / level / "VERSION.json"
            old = json.loads(path.read_text(encoding="utf-8"))
            database = path.parent / old["database"]
            if (old.get("status") != "complete"
                    or old["inputs"]["evidence_manifest_sha256"]
                    != manifest["inputs"]["evidence_manifest_sha256"]
                    or sha256_file(database) != old["database_sha256"]):
                raise ValueError(f"Frozen score source changed: {path}")
            reader = stack.enter_context(sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True))
            identities = {}
            for query_id, drug, parent, smiles in reader.execute("SELECT * FROM queries"):
                key = (drug, parent, smiles)
                if key in identities:
                    raise ValueError(f"Duplicate frozen query identity: {task}/{level}/{key}")
                identities[key] = query_id
            sources.append((reader, identities, old, path))
        if sources[0][2]["model"] != sources[1][2]["model"]:
            raise ValueError(f"Frozen score models differ: {task}/{level}")
        reused = 0
        for query_id, drug, parent, smiles in connection.execute("SELECT * FROM queries"):
            new_rows = connection.execute(
                "SELECT item_id,parent_id,morgan_rank FROM rankings "
                "WHERE benchmark_row_id=? ORDER BY morgan_rank", (query_id,),
            ).fetchall()
            candidates = []
            for reader, identities, _, _ in sources:
                old_id = identities.get((drug, parent, smiles))
                if old_id is None:
                    continue
                old_rows = reader.execute(
                    "SELECT item_id,parent_id,morgan_rank,assay_transfer_score,"
                    "assay_rank,score_key FROM rankings WHERE benchmark_row_id=? ORDER BY morgan_rank",
                    (old_id,),
                ).fetchall()
                if ([row[:3] for row in old_rows] != new_rows
                        or any(row[3] is None or row[4] is None or row[5] is None for row in old_rows)
                        or sorted(row[4] for row in old_rows) != list(range(1, len(old_rows) + 1))):
                    continue
                candidates.append(old_rows)
            if not candidates:
                continue
            if any(candidate != candidates[0] for candidate in candidates[1:]):
                raise ValueError(f"Conflicting frozen score rows: {task}/{level}/{query_id}")
            connection.executemany(
                "UPDATE rankings SET assay_transfer_score=?,assay_rank=?,score_key=? "
                "WHERE benchmark_row_id=? AND item_id=?",
                [(row[3], row[4], row[5], query_id, row[0]) for row in candidates[0]],
            )
            reused += len(candidates[0])
        manifest["model"] = sources[0][2]["model"]
        manifest["score_counts"] = {"exact_reuse": reused}
        manifest["score_reuse_sources"] = [
            {"path": str(path.resolve()), "manifest_sha256": sha256_file(path),
             "database_sha256": old["database_sha256"]}
            for _, _, old, path in sources
        ]


def _finalize(task: str, subset: str, level: str, output_root: Path, source_root: Path) -> dict:
    directory = output_root / task / "scaffold" / subset / level
    path = directory / "VERSION.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] == "complete" and "query_pure_summary" in manifest:
        return manifest["query_pure_summary"]
    if manifest["status"] not in {"complete", "prepared", "ready_to_finalize"}:
        raise ValueError(f"Unfinalizable cache: {path}")
    database = directory / manifest["database"]
    with sqlite3.connect(database) as connection:
        _transplant_frozen_scores(task, subset, level, connection, manifest, source_root)
        census = {"queries": 0, "assay_pure_queries": 0, "morgan_pure_queries": 0,
                  "assay_pure_records": 0, "morgan_pure_records": 0}
        for query_id, in connection.execute("SELECT benchmark_row_id FROM queries"):
            total, scored = connection.execute(
                "SELECT COUNT(*),COUNT(assay_transfer_score) FROM rankings WHERE benchmark_row_id=?",
                (query_id,),
            ).fetchone()
            if total < 1:
                raise ValueError(f"Empty candidate universe: {query_id}/{level}")
            census["queries"] += 1
            if scored == total:
                ranks = [row[0] for row in connection.execute(
                    "SELECT assay_rank FROM rankings WHERE benchmark_row_id=? ORDER BY assay_rank",
                    (query_id,),
                )]
                if ranks != list(range(1, total + 1)):
                    raise ValueError(f"Incomplete assay order: {query_id}/{level}")
                census["assay_pure_queries"] += 1
                census["assay_pure_records"] += total
            else:
                connection.execute(
                    "UPDATE rankings SET assay_transfer_score=NULL,assay_rank=NULL,score_key=NULL "
                    "WHERE benchmark_row_id=?", (query_id,),
                )
                census["morgan_pure_queries"] += 1
                census["morgan_pure_records"] += total
            manifest["query_counts"][query_id].update({
                "scored_records": total if scored == total else 0,
                "missing_scores": 0 if scored == total else total,
            })
        manifest["score_coverage"] = "query_pure"
        manifest["query_pure_policy"] = POLICY
        manifest["query_pure_summary"] = census
        manifest["score_counts"].pop("pending", None)
        builder._write_complete(
            connection, database, task=task, subset=subset, level=level,
            identity={key: value for key, value in manifest.items() if key not in {
                "schema_version", "status", "database", "content_id", "database_sha256",
                "ranking_rows_sha256", "prompts",
            }}, target=directory,
        )
    prompt_path = directory / "prompts.parquet"
    if prompt_path.exists():
        prompt_path.unlink()
    return census


def _validate(task: str, output_root: Path, evidence_manifest: Path) -> dict:
    index_path = output_root / task / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (index.get("status") != "complete" or index.get("query_pure_policy") != POLICY
            or index.get("label_release") != {"benchmark": "tdc", "version": "v2"}
            or sha256_file(evidence_manifest) != index["evidence"]["manifest_sha256"]):
        raise ValueError(f"Incomplete query-pure release: {index_path}")
    report = {}
    for subset in ("valid", "test"):
        report[subset] = {}
        for level in builder.TASK_LEVELS[task]:
            entry = index["splits"][subset]["levels"][level]
            manifest_path = index_path.parent / entry["manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            database = manifest_path.parent / manifest["database"]
            if (sha256_file(manifest_path) != entry["manifest_sha256"]
                    or sha256_file(database) != manifest["database_sha256"]
                    or manifest["query_pure_policy"] != POLICY
                    or manifest["inputs"]["evidence_manifest_sha256"] != sha256_file(evidence_manifest)):
                raise ValueError(f"Published hashes differ: {task}/{subset}/{level}")
            with sqlite3.connect(database) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ValueError(f"SQLite integrity failed: {task}/{subset}/{level}")
                census = {"queries": 0, "assay_pure_queries": 0, "morgan_pure_queries": 0,
                          "assay_pure_records": 0, "morgan_pure_records": 0}
                for query_id, in connection.execute("SELECT benchmark_row_id FROM queries"):
                    values = connection.execute(
                        "SELECT COUNT(*),COUNT(assay_rank),COUNT(assay_transfer_score),"
                        "COUNT(score_key),COUNT(DISTINCT morgan_rank),"
                        "COUNT(DISTINCT assay_rank),MIN(morgan_rank),MAX(morgan_rank),"
                        "MIN(assay_rank),MAX(assay_rank) FROM rankings WHERE benchmark_row_id=?",
                        (query_id,),
                    ).fetchone()
                    total, ranked, scored, keyed, unique_morgan, unique_assay, first_morgan, last_morgan, first_assay, last_assay = values
                    if not total or unique_morgan != total or first_morgan != 1 or last_morgan != total:
                        raise ValueError(f"Incomplete Morgan order: {query_id}/{level}")
                    census["queries"] += 1
                    if (ranked, scored, keyed, unique_assay, first_assay, last_assay) == (total, total, total, total, 1, total):
                        census["assay_pure_queries"] += 1
                        census["assay_pure_records"] += total
                    elif ranked == scored == keyed == 0:
                        census["morgan_pure_queries"] += 1
                        census["morgan_pure_records"] += total
                    else:
                        raise ValueError(f"Mixed ranking: {query_id}/{level}")
                if census != manifest["query_pure_summary"]:
                    raise ValueError(f"Query-pure census differs: {task}/{subset}/{level}")
                report[subset][level] = census
    return report


def build(task: str, *, dili_parent_successor: bool = False, ames_parent100: bool = False) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unsupported TDC task: {task}")
    source_root, evidence_manifest, output_root = _configure(
        task, dili_parent_successor=dili_parent_successor, ames_parent100=ames_parent100,
    )
    task_root = output_root / task
    if (task_root / "RELEASE_INDEX.json").exists():
        return _validate(task, output_root, evidence_manifest)
    for subset in ("valid", "test"):
        shared_path = task_root / "scaffold" / subset / "PARENT_UNIVERSE.json"
        if builder.SHARED_LATER_PARENT_UNIVERSE and not shared_path.exists():
            builder.prepare_shared_parent_universe(task, subset, output_root, evidence_manifest)
        for level in builder.TASK_LEVELS[task]:
            path = task_root / "scaffold" / subset / level / "VERSION.json"
            if not path.exists():
                builder.prepare_level(task, subset, level, output_root, evidence_manifest)
            _finalize(task, subset, level, output_root, source_root)
    index = builder.write_index(task, output_root, evidence_manifest)
    index.update({
        "ranking_modes": ["morgan", "query-pure"],
        "score_coverage": "query_pure",
        "selection_coverage": "complete",
        "query_pure_policy": POLICY,
        "score_source_release_index": str((source_root / task / "RELEASE_INDEX.json").resolve()),
        "score_source_release_index_sha256": sha256_file(source_root / task / "RELEASE_INDEX.json"),
        **({"parent_relation_filter": "parent_disjoint",
            "identity_normalizer_version": builder.normalize_molecule_identity("C").normalizer_version}
           if dili_parent_successor else {}),
    })
    index_path = task_root / "RELEASE_INDEX.json"
    temporary = index_path.with_name(".RELEASE_INDEX.query_pure.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, index_path)
    return _validate(task, output_root, evidence_manifest)


def report() -> Path:
    """Summarize verified query-level purity and pin every source index."""
    output = REPO_ROOT / "outputs/analysis/tdc/tdc_v2_original_query_pure_cache_v1"
    if output.exists():
        raise FileExistsError(f"Refusing to replace analysis: {output}")
    source_hashes = {}
    rows = []
    for task in TASKS:
        l1_method = "assay_transfer" if task in {"bioavailability_ma", "skin_reaction"} else "query_pure"
        l1_index_path = cache_profile_root(
            f"flat_v5/tdc_v2/{task}/l1/{l1_method}/reused_task_best_v1/original"
        ) / "RELEASE_INDEX.json"
        _, _, l2_root = _configure(task)
        l2_index_path = l2_root / task / "RELEASE_INDEX.json"
        indexes = {
            "L1": json.loads(l1_index_path.read_text(encoding="utf-8")),
            "later": json.loads(l2_index_path.read_text(encoding="utf-8")),
        }
        if (indexes["L1"].get("status") != "complete"
                or indexes["later"].get("status") != "complete"
                or indexes["later"].get("query_pure_policy") != POLICY):
            raise ValueError(f"Unpublished query-pure cache: {task}")
        source_hashes[task] = {
            "L1": {"path": str(l1_index_path), "sha256": sha256_file(l1_index_path)},
            "later": {"path": str(l2_index_path), "sha256": sha256_file(l2_index_path)},
        }
        for subset in ("valid", "test"):
            states: dict[str, dict[str, str]] = {}
            for level in ("L1", *builder.TASK_LEVELS[task]):
                role = "L1" if level == "L1" else "later"
                index_path = l1_index_path if role == "L1" else l2_index_path
                entry = indexes[role]["splits"][subset]["levels"][level]
                manifest_path = index_path.parent / entry["manifest"]
                if sha256_file(manifest_path) != entry["manifest_sha256"]:
                    raise ValueError(f"Cache manifest changed: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if level == "L1":
                    states[level] = {
                        query_id: ("assay" if values.get("assay_candidate_records") == values["morgan_candidate_records"]
                                   else "morgan")
                        for query_id, values in manifest["query_counts"].items()
                    }
                else:
                    states[level] = {
                        query_id: ("assay" if values["scored_records"] == values["candidate_records"]
                                   else "morgan")
                        for query_id, values in manifest["query_counts"].items()
                    }
            query_ids = set(states["L1"])
            if any(set(values) != query_ids for values in states.values()):
                raise ValueError(f"Query IDs differ between levels: {task}/{subset}")
            later_levels = tuple(level for level in states if level != "L1")
            states["L2plus"] = {
                query_id: (
                    "assay" if all(states[level][query_id] == "assay" for level in later_levels)
                    else "morgan" if all(states[level][query_id] == "morgan" for level in later_levels)
                    else "mixed"
                ) for query_id in query_ids
            }
            states["all_levels"] = {
                query_id: (
                    "assay" if states["L1"][query_id] == states["L2plus"][query_id] == "assay"
                    else "morgan" if states["L1"][query_id] == states["L2plus"][query_id] == "morgan"
                    else "mixed"
                ) for query_id in query_ids
            }
            for level, values in states.items():
                counts = {name: list(values.values()).count(name) for name in ("morgan", "assay", "mixed")}
                rows.append({
                    "task": task, "split": subset, "scope": level,
                    "queries": len(values),
                    **{f"{name}_queries": counts[name] for name in counts},
                    **{f"{name}_pct": f"{100 * counts[name] / len(values):.6f}" for name in counts},
                })
    for subset in ("valid", "test"):
        for scope in ("L2plus", "all_levels"):
            selected = [row for row in rows if row["split"] == subset and row["scope"] == scope]
            total = sum(row["queries"] for row in selected)
            counts = {name: sum(row[f"{name}_queries"] for row in selected)
                      for name in ("morgan", "assay", "mixed")}
            rows.append({
                "task": "ALL_SIX", "split": subset, "scope": scope, "queries": total,
                **{f"{name}_queries": counts[name] for name in counts},
                **{f"{name}_pct": f"{100 * counts[name] / total:.6f}" for name in counts},
            })
    # A new DILI retrieval world may exceed the selected semantic-score map.
    import pyarrow.parquet as pq
    from optimization.semantic_buckets.inputs import resolve_semantic_bucket_artifacts

    semantic = resolve_semantic_bucket_artifacts("dili", "CURRENT")
    weighted = set(pq.read_table(
        semantic.record_relevance_rankings, columns=["source_row_uid"]
    )["source_row_uid"].to_pylist())
    dili_index_path = Path(source_hashes["dili"]["later"]["path"])
    dili_index = json.loads(dili_index_path.read_text(encoding="utf-8"))
    semantic_rows = []
    world = set()
    for subset in ("valid", "test"):
        for level, entry in dili_index["splits"][subset]["levels"].items():
            manifest_path = dili_index_path.parent / entry["manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with sqlite3.connect(manifest_path.parent / manifest["database"]) as connection:
                uids = [uid for uid, in connection.execute("SELECT item_id FROM rankings")]
            unique = set(uids)
            world.update(unique)
            semantic_rows.append({
                "split": subset, "level": level, "occurrences": len(uids),
                "missing_occurrences": sum(uid not in weighted for uid in uids),
                "unique_uids": len(unique), "missing_unique_uids": len(unique - weighted),
            })
    semantic_rows.append({
        "split": "ALL", "level": "ALL",
        "occurrences": sum(row["occurrences"] for row in semantic_rows),
        "missing_occurrences": sum(row["missing_occurrences"] for row in semantic_rows),
        "unique_uids": len(world), "missing_unique_uids": len(world - weighted),
    })
    output.mkdir(parents=True)
    tsv = output / "purity.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "semantic_coverage.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(semantic_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(semantic_rows)
    summary = [row for row in rows if row["scope"] in {"L2plus", "all_levels"}]
    lines = [
        "# TDC-v2 original-split query-pure cache restoration", "",
        "A query-level panel is assay-pure only when every candidate has its exact frozen score. "
        "Otherwise the whole panel is Morgan-pure with assay values hidden. "
        "The L2plus and all-levels rows are pure only if every included level agrees; "
        "otherwise it is mixed across levels.", "",
        "BBB and Bioavailability have no frozen assay scores at L5, so their L5 "
        "and whole-prompt classifications cannot be assay-pure.", "",
        "The published AMES predecessor is a partial snapshot; its test L4/L5 "
        "panels are incomplete and therefore Morgan-pure here.", "",
        "| Task | Split | Scope | Queries | Morgan-pure | Assay-pure | Mixed across levels |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['task']} | {row['split']} | {row['scope']} | {row['queries']} | "
            f"{row['morgan_queries']} ({row['morgan_pct']}%) | "
            f"{row['assay_queries']} ({row['assay_pct']}%) | "
            f"{row['mixed_queries']} ({row['mixed_pct']}%) |"
        )
    lines += [
        "", "See `purity.tsv` for every task, split, and level.", "",
        f"DILI semantic-score coverage: {len(world - weighted)} of {len(world)} unique "
        "candidate UIDs are absent from the selected sidecar. See `semantic_coverage.tsv` "
        "for split/level counts. A reviewed semantic successor is required before "
        "using that sidecar for this new world.", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "tdc_v2_query_pure_analysis.v1",
        "sources": source_hashes,
        "semantic_sidecar": {
            "path": str(semantic.record_relevance_rankings),
            "sha256": sha256_file(semantic.record_relevance_rankings),
        },
        "outputs": {name: sha256_file(output / name) for name in (
            "purity.tsv", "semantic_coverage.tsv", "report.md")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--dili-parent-successor", action="store_true")
    parser.add_argument("--ames-parent100", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(report())
    elif args.task:
        print(json.dumps(build(args.task, dili_parent_successor=args.dili_parent_successor,
                               ames_parent100=args.ames_parent100), indent=2, sort_keys=True))
    else:
        parser.error("--task or --report is required")


if __name__ == "__main__":
    main()
