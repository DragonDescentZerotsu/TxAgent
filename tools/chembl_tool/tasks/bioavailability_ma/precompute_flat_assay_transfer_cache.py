"""Build, migrate, score, and verify the retrieval-agnostic assay-transfer cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.chembl_tool.common.experiment_retrieval import retrieve_experiment_view
from tools.chembl_tool.tasks.bioavailability_ma.assay_transfer_rerank import (
    ASSAY_TRANSFER_MODEL,
    ASSAY_TRANSFER_MODEL_REVISION,
    CATALOG_SCHEMA_VERSION,
    SCORING_CONTRACT_VERSION,
    AssayTransferCachedReranker,
    AssayTransferPromptRenderer,
    AssayTransferScoreCache,
    PromptScore,
    PromptTask,
    flat_score_key,
    prompt_task_from_dict,
    template_bundle_hash,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_assay_transfer_rerank_catalog import (
    _CandidateCollector,
    _canonical_json,
    _records_from_index_candidate,
)
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import STARLING
from tools.chembl_tool.tasks.bioavailability_ma.precompute_assay_transfer_rerank import (
    resolve_model_snapshot,
    run_spawned_workers,
)
from tools.chembl_tool.tasks.bioavailability_ma.retrieve_neighbors import load_index


BASE_DIR = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/assay_transfer_rerank"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "flat_v2"
DEFAULT_FOUR_SOURCE_INDEX = Path(
    "outputs/paper/molecular_evidence_agent/evidence/"
    "bioavailability_starling_four_source_full/starling_factor_neighbor_index.pkl"
)
DEFAULT_FIVE_SOURCE_INDEX = Path(
    "outputs/paper/molecular_evidence_agent/evidence/"
    "bioavailability_starling_five_source_full/starling_factor_neighbor_index.pkl"
)
LEGACY_CACHES = (BASE_DIR / "scores.sqlite3", BASE_DIR / "test/scores.sqlite3")
EXPECTED_CATALOG_RECORDS = 17_694
EXPECTED_LEGACY_ROWS = 35_833
EXPECTED_DEMAND = 39_093
EXPECTED_MISSES = 3_260
MANIFEST_SCHEMA_VERSION = "assay_transfer_candidate_manifest.flat.v2"
AUDIT_SCHEMA_VERSION = "assay_transfer_score_audit.flat.v2"


@dataclass(frozen=True)
class Condition:
    split: str
    source_setup: str
    identity_policy: str

    @property
    def condition_id(self) -> str:
        return f"{self.split}__{self.source_setup}__{self.identity_policy}"


CONDITIONS = tuple(
    Condition(split, source_setup, identity_policy)
    for split in ("validation", "test")
    for source_setup in ("four_source", "five_source")
    for identity_policy in ("operational", "parent_disjoint")
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.prepare:
        summary = prepare_flat_artifacts(args, output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.infer:
        summary = infer_missing_scores(args, output_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.verify:
        summary = verify_flat_artifacts(args, output_dir, strict=args.strict_retrieval)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def prepare_flat_artifacts(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    split_records = {
        "validation": _read_jsonl(Path(args.validation_input)),
        "test": _read_jsonl(Path(args.test_input)),
    }
    indices = {
        "four_source": load_index(Path(args.four_source_index)),
        "five_source": load_index(Path(args.five_source_index)),
    }

    catalog_by_id: dict[str, dict[str, Any]] = {}
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows, records = freeze_condition(
            condition=condition,
            query_records=split_records[condition.split],
            index=indices[condition.source_setup],
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            raw_pool_size=args.raw_pool_size,
            candidate_size=args.candidate_size,
        )
        for record in records:
            record_id = str(record["record_id"])
            previous = catalog_by_id.setdefault(record_id, record)
            if previous != record:
                raise ValueError(f"Conflicting payloads for assay record ID {record_id}")
        condition_rows[condition.condition_id] = rows
        record_links = sum(
            len(candidate["record_ids"])
            for row in rows
            for candidate in row["candidates"]
        )
        condition_summaries[condition.condition_id] = {
            "n_queries": len(split_records[condition.split]),
            "n_groups": len(rows),
            "n_candidates": sum(len(row["candidates"]) for row in rows),
            "n_record_links": record_links,
            "n_unique_record_ids": len(
                {
                    record_id
                    for row in rows
                    for candidate in row["candidates"]
                    for record_id in candidate["record_ids"]
                }
            ),
        }

    catalog_records = [catalog_by_id[key] for key in sorted(catalog_by_id)]
    records_digest = hashlib.sha256(
        "\n".join(_canonical_json(row) for row in catalog_records).encode("utf-8")
    ).hexdigest()
    catalog_version = f"{CATALOG_SCHEMA_VERSION}:{records_digest}"
    catalog_metadata = {
        "record_type": "catalog_metadata",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": catalog_version,
        "source_mode": "flat_five_source_condition_union",
        "template_hash": template_bundle_hash(),
        "n_records": len(catalog_records),
        "records_sha256": records_digest,
        "conditions": [condition.condition_id for condition in CONDITIONS],
    }
    _write_jsonl_stable(output_dir / "catalog.jsonl", [catalog_metadata, *catalog_records])

    for condition in CONDITIONS:
        rows = condition_rows[condition.condition_id]
        metadata = {
            "record_type": "manifest_metadata",
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "condition_id": condition.condition_id,
            "split": condition.split,
            "source_setup": condition.source_setup,
            "identity_policy": condition.identity_policy,
            "catalog_version": catalog_version,
            "index_path": str(
                Path(args.four_source_index)
                if condition.source_setup == "four_source"
                else Path(args.five_source_index)
            ),
            **condition_summaries[condition.condition_id],
        }
        _write_jsonl_stable(
            manifests_dir / f"{condition.condition_id}.jsonl", [metadata, *rows]
        )

    renderer = AssayTransferPromptRenderer()
    demand_by_key: dict[str, PromptTask] = {}
    audit_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for row in condition_rows[condition.condition_id]:
            for candidate_rank, candidate in enumerate(row["candidates"], start=1):
                for record_id in candidate["record_ids"]:
                    record = catalog_by_id[record_id]
                    task = _build_task(
                        renderer=renderer,
                        record=record,
                        query_smiles=str(row["query_smiles"]),
                        group_id=str(row["group_id"]),
                        molecule_id=str(candidate["molecule_id"]),
                        catalog_version=catalog_version,
                        model=args.model,
                        model_revision=args.model_revision,
                    )
                    previous = demand_by_key.setdefault(task.cache_key, task)
                    if previous.prompt_hash != task.prompt_hash or previous.prompt != task.prompt:
                        raise ValueError(f"Conflicting prompt payload for score key {task.cache_key}")
                    audit_rows.append(
                        {
                            "schema_version": AUDIT_SCHEMA_VERSION,
                            "condition_id": condition.condition_id,
                            "split": condition.split,
                            "source_setup": condition.source_setup,
                            "identity_policy": condition.identity_policy,
                            "query_index": row["query_index"],
                            "query_smiles": row["query_smiles"],
                            "group_id": row["group_id"],
                            "candidate_rank": candidate_rank,
                            "molecule_id": candidate["molecule_id"],
                            "record_id": record_id,
                            "prompt_hash": task.prompt_hash,
                            "score_key": task.cache_key,
                        }
                    )
    demand = [demand_by_key[key] for key in sorted(demand_by_key)]
    _write_jsonl_stable(output_dir / "demand.jsonl", [asdict(task) for task in demand])
    audit_rows.sort(
        key=lambda row: (
            row["condition_id"],
            row["query_index"],
            row["group_id"],
            row["candidate_rank"],
            row["record_id"],
        )
    )
    _write_jsonl_stable(output_dir / "audit_mapping.jsonl", audit_rows)

    cache = AssayTransferScoreCache(output_dir / "scores.sqlite3", mode="read_write")
    try:
        if args.skip_legacy_migration:
            migration = {"status": "skipped", "reason": "new model revision; no reusable legacy scores"}
            legacy_keys: set[str] = set()
        else:
            migration = migrate_legacy_scores(cache, LEGACY_CACHES)
            legacy_keys = {
                str(row[0])
                for row in cache.connection.execute(
                    "SELECT cache_key FROM prompt_scores WHERE score_origin='legacy_migration'"
                )
            }
    finally:
        cache.close()
    covered_after_migration = [task for task in demand if task.cache_key in legacy_keys]
    misses = [task for task in demand if task.cache_key not in legacy_keys]
    summary = {
        "status": "prepared",
        "output_dir": str(output_dir),
        "model": args.model,
        "model_revision": args.model_revision,
        "retrieval_cap": {
            "rerank_raw_pool_size": args.raw_pool_size,
            "rerank_candidate_size": args.candidate_size,
            "min_similarity": args.min_similarity,
            "top_k_per_group": args.top_k_per_group,
        },
        "n_catalog_records": len(catalog_records),
        "n_conditions": len(CONDITIONS),
        "n_audit_links": len(audit_rows),
        "n_unique_demand": len(demand),
        "n_covered_after_migration": len(covered_after_migration),
        "n_missing_after_migration": len(misses),
        "migration": migration,
        "conditions": condition_summaries,
    }
    if not args.skip_expected_guards:
        _require_expected_counts(summary)
    _write_json_stable(output_dir / "prepare_summary.json", summary)
    return summary


def freeze_condition(
    *,
    condition: Condition,
    query_records: list[dict[str, Any]],
    index: dict[str, Any],
    top_k_per_group: int,
    min_similarity: float,
    raw_pool_size: int,
    candidate_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selections: list[dict[str, Any]] = []
    for query_index, query_record in enumerate(query_records):
        query_smiles = str(query_record.get("drug") or "")
        if not query_smiles:
            raise ValueError(f"{condition.split} record {query_index} has no drug SMILES")
        retrieve_experiment_view(
            query_smiles,
            index,
            mode="full_mechanism",
            config=STARLING,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
            neighbor_identity_policy=condition.identity_policy,
            reranker=_CandidateCollector(query_index, selections),
            rerank_raw_pool_size=raw_pool_size,
            rerank_candidate_size=candidate_size,
        )

    records_by_id: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for selection in selections:
        candidates = []
        for candidate in selection["candidates"]:
            record_ids = []
            for record in _records_from_index_candidate(selection["group_id"], candidate):
                record_id = str(record["record_id"])
                previous = records_by_id.setdefault(record_id, record)
                if previous != record:
                    raise ValueError(f"Conflicting condition record payload for {record_id}")
                record_ids.append(record_id)
            candidates.append(
                {
                    "molecule_id": candidate["molecule_chembl_id"],
                    "canonical_smiles": candidate["canonical_smiles"],
                    "similarity": candidate["similarity"],
                    "structural_rank": candidate["structural_rank"],
                    "record_ids": sorted(set(record_ids)),
                }
            )
        rows.append(
            {
                "record_type": "candidate_group",
                "query_index": selection["query_index"],
                "query_smiles": selection["query_smiles"],
                "group_id": selection["group_id"],
                "raw_pool_size": raw_pool_size,
                "candidate_size": candidate_size,
                "candidates": candidates,
            }
        )
    rows.sort(key=lambda row: (int(row["query_index"]), str(row["group_id"])))
    return rows, [records_by_id[key] for key in sorted(records_by_id)]


def migrate_legacy_scores(
    cache: AssayTransferScoreCache, legacy_paths: Iterable[Path]
) -> dict[str, Any]:
    source_rows: list[tuple[Any, ...]] = []
    by_identity: dict[tuple[str, str, str, str, str], tuple[Any, ...]] = {}
    for path in legacy_paths:
        if not path.exists():
            raise FileNotFoundError(f"Legacy assay-transfer cache not found: {path}")
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                """
                SELECT prompt_hash, model, model_revision, scoring_contract_version,
                       template_hash, logp_transfer, logp_not_transfer,
                       transfer_probability
                FROM prompt_scores ORDER BY cache_key
                """
            ).fetchall()
        finally:
            connection.close()
        source_rows.extend(rows)
        for row in rows:
            identity = tuple(str(value) for value in row[:5])
            previous = by_identity.setdefault(identity, row)
            if previous != row:
                raise ValueError(f"Conflicting legacy score payload for {identity[0]}")

    tasks: list[PromptTask] = []
    scores: list[PromptScore] = []
    for identity, row in sorted(by_identity.items()):
        prompt_hash, model, revision, contract, template_hash = identity
        cache_key = flat_score_key(
            prompt_hash=prompt_hash,
            model=model,
            model_revision=revision,
            scoring_contract_version=contract,
            template_hash=template_hash,
        )
        tasks.append(
            PromptTask(
                cache_key=cache_key,
                prompt_hash=prompt_hash,
                prompt="",
                query_smiles="",
                group_id="",
                molecule_id="",
                record_id="",
                model=model,
                model_revision=revision,
                scoring_contract_version=contract,
                template_hash=template_hash,
                catalog_version="legacy_migration",
            )
        )
        scores.append(
            PromptScore(
                cache_key=cache_key,
                logp_transfer=float(row[5]),
                logp_not_transfer=float(row[6]),
                transfer_probability=float(row[7]),
            )
        )
    for offset in range(0, len(tasks), 500):
        cache.write_batch(
            tasks[offset : offset + 500],
            scores[offset : offset + 500],
            score_origin="legacy_migration",
        )

    observed = cache.lookup(tasks)
    for task, score in zip(tasks, scores, strict=True):
        if observed.get(task.cache_key) != score:
            raise ValueError(f"Legacy score changed during migration: {task.prompt_hash}")
    return {
        "n_source_rows": len(source_rows),
        "n_unique_rows": len(by_identity),
        "n_exactly_verified": len(observed),
        "legacy_paths": [str(path) for path in legacy_paths],
    }


def infer_missing_scores(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    demand = [prompt_task_from_dict(row) for row in _read_jsonl(output_dir / "demand.jsonl")]
    cache = AssayTransferScoreCache(output_dir / "scores.sqlite3", mode="read_write")
    try:
        covered = cache.lookup(demand)
        missing = [task for task in demand if task.cache_key not in covered]
        if len({task.cache_key for task in missing}) != len(missing):
            raise ValueError("Missing-score worklist contains duplicate model computations")
        if not missing:
            return {"status": "already_complete", "n_missing": 0, "n_scored": 0}
        snapshot_path, immutable_revision = resolve_model_snapshot(
            args.model,
            args.model_revision,
            force_download=args.force_model_download,
            local_files_only=args.local_files_only,
        )
        if immutable_revision != args.model_revision:
            raise ValueError(
                f"Resolved model revision {immutable_revision} does not match demand revision "
                f"{args.model_revision}"
            )
        demand_revisions = {task.model_revision for task in demand}
        if demand_revisions != {immutable_revision}:
            raise ValueError(
                f"Demand file mixes model revisions {sorted(demand_revisions)}; expected only "
                f"{immutable_revision}. Re-run --prepare with the intended --model-revision."
            )
        devices = _parse_devices(args.devices)
        summary = run_spawned_workers(
            missing,
            cache=cache,
            snapshot_path=snapshot_path,
            devices=devices,
            batch_size=args.batch_size,
            dtype="bfloat16",
        )
        final_covered = cache.lookup(demand)
    finally:
        cache.close()
    result = {
        "status": "complete",
        "n_missing_before_inference": len(missing),
        "n_unique_covered": len(final_covered),
        **summary,
    }
    _write_json_stable(output_dir / "inference_summary.json", result)
    return result


def verify_flat_artifacts(
    args: argparse.Namespace, output_dir: Path, *, strict: bool
) -> dict[str, Any]:
    catalog_rows = _read_jsonl(output_dir / "catalog.jsonl")
    metadata, records = catalog_rows[0], catalog_rows[1:]
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record["record_id"])
        previous = by_id.setdefault(record_id, record)
        if previous != record:
            raise ValueError(f"Conflicting catalog record ID {record_id}")
    demand = [prompt_task_from_dict(row) for row in _read_jsonl(output_dir / "demand.jsonl")]
    if len({task.cache_key for task in demand}) != len(demand):
        raise ValueError("Demand file contains duplicate flat score keys")
    cache = AssayTransferScoreCache(output_dir / "scores.sqlite3", mode="read_only")
    try:
        covered = cache.lookup(demand)
        origin_counts = dict(
            cache.connection.execute(
                "SELECT score_origin, COUNT(*) FROM prompt_scores GROUP BY score_origin"
            ).fetchall()
        )
        total_store_rows = int(
            cache.connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0]
        )
    finally:
        cache.close()
    missing = sorted({task.cache_key for task in demand} - set(covered))

    audit_rows = _read_jsonl(output_dir / "audit_mapping.jsonl")
    manifests: dict[str, list[dict[str, Any]]] = {}
    four_source_ids: set[str] = set()
    five_source_ids: set[str] = set()
    for condition in CONDITIONS:
        rows = _read_jsonl(output_dir / "manifests" / f"{condition.condition_id}.jsonl")
        manifest_metadata, candidate_rows = rows[0], rows[1:]
        if manifest_metadata["condition_id"] != condition.condition_id:
            raise ValueError(f"Manifest condition mismatch: {condition.condition_id}")
        manifests[condition.condition_id] = candidate_rows
        joined = {
            record_id
            for row in candidate_rows
            for candidate in row["candidates"]
            for record_id in candidate["record_ids"]
        }
        unknown = joined - set(by_id)
        if unknown:
            raise ValueError(f"Manifest {condition.condition_id} has unknown record IDs")
        if condition.source_setup == "four_source":
            four_source_ids.update(joined)
        else:
            five_source_ids.update(joined)
    fifth_source_catalog_ids = {
        record_id
        for record_id, record in by_id.items()
        if record.get("source_id") == "starling-labs/Oral_Bioavailability"
    }
    leaked = four_source_ids & fifth_source_catalog_ids
    if leaked:
        raise ValueError(f"Four-source manifests leaked {len(leaked)} fifth-source records")

    strict_summaries: dict[str, Any] = {}
    if strict:
        split_records = {
            "validation": _read_jsonl(Path(args.validation_input)),
            "test": _read_jsonl(Path(args.test_input)),
        }
        indices = {
            "four_source": load_index(Path(args.four_source_index)),
            "five_source": load_index(Path(args.five_source_index)),
        }
        audit_keys_by_condition: dict[str, set[str]] = {}
        for row in audit_rows:
            audit_keys_by_condition.setdefault(str(row["condition_id"]), set()).add(
                str(row["score_key"])
            )
        for condition in CONDITIONS:
            regenerated, regenerated_records = freeze_condition(
                condition=condition,
                query_records=split_records[condition.split],
                index=indices[condition.source_setup],
                top_k_per_group=args.top_k_per_group,
                min_similarity=args.min_similarity,
                raw_pool_size=args.raw_pool_size,
                candidate_size=args.candidate_size,
            )
            if regenerated != manifests[condition.condition_id]:
                raise ValueError(f"Strict manifest replay mismatch: {condition.condition_id}")
            regenerated_ids = {str(record["record_id"]) for record in regenerated_records}
            manifest_ids = {
                record_id
                for row in manifests[condition.condition_id]
                for candidate in row["candidates"]
                for record_id in candidate["record_ids"]
            }
            if regenerated_ids != manifest_ids:
                raise ValueError(f"Strict record join mismatch: {condition.condition_id}")

            reranker = AssayTransferCachedReranker(
                catalog_path=output_dir / "catalog.jsonl",
                cache_path=output_dir / "scores.sqlite3",
                cache_mode="read_only",
                model=args.model,
                model_revision=args.model_revision,
                candidate_manifest_path=(
                    output_dir / "manifests" / f"{condition.condition_id}.jsonl"
                ),
            )
            try:
                for query_record in split_records[condition.split]:
                    retrieve_experiment_view(
                        str(query_record["drug"]),
                        indices[condition.source_setup],
                        mode="full_mechanism",
                        config=STARLING,
                        top_k_per_group=args.top_k_per_group,
                        min_similarity=args.min_similarity,
                        neighbor_identity_policy=condition.identity_policy,
                        reranker=reranker,
                        rerank_raw_pool_size=args.raw_pool_size,
                        rerank_candidate_size=args.candidate_size,
                    )
                seen_keys = set(reranker.seen_tasks)
                expected_keys = audit_keys_by_condition[condition.condition_id]
                if seen_keys != expected_keys:
                    raise ValueError(f"Strict score-key preflight mismatch: {condition.condition_id}")
                strict_summaries[condition.condition_id] = {
                    "status": "complete",
                    "n_unique_prompt_scores": len(seen_keys),
                    "n_missing": len(reranker.missing_tasks),
                }
            finally:
                reranker.cache.close()

    summary = {
        "status": "complete" if not missing else "incomplete",
        "catalog_version": metadata["catalog_version"],
        "n_catalog_records": len(records),
        "n_catalog_unique_record_ids": len(by_id),
        "n_unique_demand": len(demand),
        "n_unique_covered": len(covered),
        "n_missing": len(missing),
        "n_store_rows": total_store_rows,
        "score_origin_counts": origin_counts,
        "n_audit_links": len(audit_rows),
        "n_four_source_record_ids": len(four_source_ids),
        "n_five_source_record_ids": len(five_source_ids),
        "n_fifth_source_records_in_four_source": len(leaked),
        "strict_retrieval": strict_summaries,
    }
    if missing:
        raise ValueError(f"Flat score cache still has {len(missing)} missing prompts")
    if len(covered) != len(demand) or total_store_rows != len(demand):
        raise ValueError("Flat score store does not have exactly one row per demanded prompt")
    if not args.skip_expected_guards:
        if len(records) != EXPECTED_CATALOG_RECORDS:
            raise ValueError(f"Expected {EXPECTED_CATALOG_RECORDS} catalog records, got {len(records)}")
        if len(demand) != EXPECTED_DEMAND:
            raise ValueError(f"Expected {EXPECTED_DEMAND} prompts, got {len(demand)}")
        if int(origin_counts.get("legacy_migration", 0)) != EXPECTED_LEGACY_ROWS:
            raise ValueError("Legacy migration row count changed")
        if int(origin_counts.get("inference", 0)) != EXPECTED_MISSES:
            raise ValueError("Inference row count does not equal the unique missing worklist")
    _write_json_stable(output_dir / "verification.json", summary)
    return summary


def _build_task(
    *,
    renderer: AssayTransferPromptRenderer,
    record: dict[str, Any],
    query_smiles: str,
    group_id: str,
    molecule_id: str,
    catalog_version: str,
    model: str = ASSAY_TRANSFER_MODEL,
    model_revision: str = ASSAY_TRANSFER_MODEL_REVISION,
) -> PromptTask:
    prompt = renderer.render(record, query_smiles)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_key = flat_score_key(
        prompt_hash=prompt_hash,
        model=model,
        model_revision=model_revision,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        template_hash=renderer.template_hash,
    )
    return PromptTask(
        cache_key=cache_key,
        prompt_hash=prompt_hash,
        prompt=prompt,
        query_smiles=query_smiles,
        group_id=group_id,
        molecule_id=molecule_id,
        record_id=str(record["record_id"]),
        model=model,
        model_revision=model_revision,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        template_hash=renderer.template_hash,
        catalog_version=catalog_version,
    )


def _require_expected_counts(summary: dict[str, Any]) -> None:
    expected = {
        "n_catalog_records": EXPECTED_CATALOG_RECORDS,
        "n_unique_demand": EXPECTED_DEMAND,
        "n_covered_after_migration": EXPECTED_LEGACY_ROWS,
        "n_missing_after_migration": EXPECTED_MISSES,
    }
    mismatches = {
        key: (expected_value, summary.get(key))
        for key, expected_value in expected.items()
        if summary.get(key) != expected_value
    }
    if summary["migration"]["n_source_rows"] != EXPECTED_LEGACY_ROWS:
        mismatches["migration.n_source_rows"] = (
            EXPECTED_LEGACY_ROWS,
            summary["migration"]["n_source_rows"],
        )
    if mismatches:
        raise ValueError(f"Flat cache preparation count mismatch: {mismatches}")


def _write_jsonl_stable(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _write_stable(path, payload)


def _write_json_stable(path: Path, payload: dict[str, Any]) -> None:
    _write_stable(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _write_stable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace non-identical flat artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_devices(value: str) -> list[int]:
    devices = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"Invalid device list: {value}")
    return devices


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--infer", action="store_true")
    actions.add_argument("--verify", action="store_true")
    parser.add_argument("--strict-retrieval", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--validation-input", default="data/processed/Bioavailability_Ma/valid.jsonl"
    )
    parser.add_argument("--test-input", default="data/processed/Bioavailability_Ma/test.jsonl")
    parser.add_argument("--four-source-index", default=str(DEFAULT_FOUR_SOURCE_INDEX))
    parser.add_argument("--five-source-index", default=str(DEFAULT_FIVE_SOURCE_INDEX))
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--raw-pool-size", type=int, default=100)
    parser.add_argument("--candidate-size", type=int, default=50)
    parser.add_argument("--model", default=ASSAY_TRANSFER_MODEL)
    parser.add_argument("--model-revision", default=ASSAY_TRANSFER_MODEL_REVISION)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force-model-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-legacy-migration",
        action="store_true",
        help="Do not import the v1 condition-scoped caches (required when building on a new model revision).",
    )
    parser.add_argument(
        "--skip-expected-guards",
        action="store_true",
        help="Skip the v1-specific catalog/demand/row count assertions (required for a new cap or model).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
