"""Canonical molecule-by-context record collapse for normalized Starling v7."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sqlite3
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.assay_transfer_measurements import (
    display_measurement_tuple,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.final_endpoint_pruning import (
    load_final_endpoint_pruning,
)
from tools.chembl_tool.common.starling.record_deduplication import (
    load_duplicate_lineage,
)
from tools.chembl_tool.common.starling.semantic_record_aggregation import (
    SemanticAggregationConfig,
    SemanticAggregationLimitReached,
    TEMPLATE_PATH,
    aggregate_semantic_groups,
    load_semantic_cache,
    render_prompt,
    validate_semantic_response,
)


COLLAPSE_VERSION = "starling_record_collapse.v6"
RECORDS_FILENAME = "records.parquet"
SEMANTIC_FILENAME = "semantic_aggregation.jsonl"
MANIFEST_FILENAME = "manifest.json"
DIRECT_RETRIEVAL_SOURCES = frozenset({"direct_vote", "direct_residual"})
UNCONDITIONED_AGREEMENT_THRESHOLD = 0.70
CONDITIONED_AGREEMENT_THRESHOLD = 0.60
_BASE_COLUMNS = {
    "canonical_record_id",
    "source_id",
    "source_name",
    "source_record_id",
    "source_row_number",
    "source_index",
    "canonical_smiles",
    "smiles",
    "retrieval_eligible",
    "group_id",
    "canonical_endpoint_name",
    "finite_scalar_value",
    "variation_value",
    "measurement_kind",
    "canonical_measurement_scale_id",
    "canonical_category_id",
    "canonical_category_rank",
    "canonical_measurement_text",
    "canonical_unit_text",
    "measurement_text",
    "unit_text",
    "support_text",
    "pmid",
    "doi",
    "confidence",
    "retrieval_source_id",
    "direct_vote_label",
    "direct_vote_unit_id",
    "condition_group",
    "condition_atoms",
    "condition_scope",
    "condition_key_status",
    "direct_group_id",
}
_PARENTHETICAL_SMILES_RE = re.compile(
    r"\(\s*SMILES\s*:\s*[^\s)]+\s*\)", re.IGNORECASE
)
_COMPOUND_SMILES_RE = re.compile(
    r"\bcompound\s+(?:identified\s+as\s+)?SMILES\s*:\s*\S+", re.IGNORECASE
)
_SMILES_RE = re.compile(r"\bSMILES\s*:\s*\S+", re.IGNORECASE)


def build_collapsed_record_stage(
    *,
    task_id: str,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    out_dir: str | Path,
    duplicate_lineage_path: str | Path | None = None,
    semantic_source_columns: Mapping[str, Sequence[str]] | None = None,
    direct_label_definition: str = "the task's direct binary outcome",
    preserved_columns: Sequence[str] = (),
    semantic_config: SemanticAggregationConfig | None = None,
    semantic_model: str | None = None,
    prior_semantic_paths: Sequence[str | Path] = (),
    defer_semantic_aggregation: bool = False,
    final_endpoint_pruning_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collapse retrieval-eligible rows by direct condition or indirect bucket."""
    source_path = Path(records_path)
    bucket_path = Path(pair_bucket_records_path)
    metadata_path = Path(pair_bucket_metadata_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    available = set(pq.read_schema(source_path).names)
    source_columns = semantic_source_columns or {}
    columns = _projection_columns(
        available,
        {
            field
            for fields in source_columns.values()
            for field in fields
        },
        preserved_columns,
    )
    empty_records_schema = _empty_collapsed_schema(
        source_path, bucket_path, set(preserved_columns)
    )
    required = {
        "canonical_record_id",
        "source_id",
        "canonical_smiles",
        "retrieval_eligible",
    }
    missing = required - available
    if missing:
        raise ValueError(f"collapse input lacks required columns: {sorted(missing)}")
    pruned_record_ids, pruning_input = (
        load_final_endpoint_pruning(
            final_endpoint_pruning_manifest_path,
            task_id=task_id,
            records_path=source_path,
            pair_bucket_records_path=bucket_path,
        )
        if final_endpoint_pruning_manifest_path is not None
        else (set(), None)
    )
    work_dir = tempfile.TemporaryDirectory(prefix="starling-record-collapse-")
    work_db = Path(work_dir.name) / "records.sqlite"
    connection = sqlite3.connect(work_db)
    _initialize_work_db(connection)
    stream_stats = _stream_joined_records(
        connection,
        records_path=source_path,
        bucket_path=bucket_path,
        columns=sorted(columns),
        pruned_record_ids=pruned_record_ids,
    )
    dedup_lineage = (
        load_duplicate_lineage(duplicate_lineage_path)
        if duplicate_lineage_path is not None
        else {}
    )
    semantic_stats = _materialize_semantic_responses(
        connection,
        task_id=task_id,
        semantic_config=semantic_config,
        semantic_model=semantic_model,
        prior_semantic_paths=prior_semantic_paths,
        defer_semantic_aggregation=defer_semantic_aggregation,
        direct_label_definition=direct_label_definition,
        semantic_source_columns=source_columns,
    )
    records_file = target / RECORDS_FILENAME
    semantic_file = target / SEMANTIC_FILENAME
    output_stats = _write_collapsed_outputs(
        connection,
        task_id=task_id,
        records_file=records_file,
        semantic_file=semantic_file,
        dedup_lineage=dedup_lineage,
        empty_records_schema=empty_records_schema,
        preserved_columns=preserved_columns,
        defer_semantic_aggregation=defer_semantic_aggregation,
        semantic_model=semantic_model,
    )
    connection.close()
    work_dir.cleanup()
    if not output_stats["pending_semantic_records_are_not_retrieval_eligible"]:
        raise ValueError("pending semantic records must be excluded from retrieval")
    manifest = {
        "version": COLLAPSE_VERSION,
        "task_id": task_id,
        "grouping_contract": {
            "direct_vote": ["canonical_smiles", "condition_group"],
            "direct_residual": [
                "canonical_smiles",
                "condition_group",
                "condition_key_status",
            ],
            "indirect": ["canonical_smiles", "pair_bucket_key"],
        },
        "aggregation_contract": {
            "continuous_absolute": "median",
            "controlled_categorical": "mode_with_full_counts_null_on_tie",
            "single_relative_or_unresolved_scalar": "numeric_passthrough",
            "free_text_or_multi_relative": "llm_loss_aware_summary",
            "direct_unconditioned_agreement_threshold": UNCONDITIONED_AGREEMENT_THRESHOLD,
            "direct_conditioned_agreement_threshold": CONDITIONED_AGREEMENT_THRESHOLD,
        },
        "semantic_aggregation": {
            "deferred": defer_semantic_aggregation,
            "requested_model": (
                semantic_config.model if semantic_config else semantic_model
            ),
            "endpoint": semantic_config.base_url if semantic_config else None,
            "models_in_artifact": output_stats["semantic_models"],
            "template": str(TEMPLATE_PATH),
            "template_sha256": file_sha256(TEMPLATE_PATH),
        },
        "deduplication_contract": {
            "performed_here": False,
            "input_lineage": str(duplicate_lineage_path or ""),
        },
        "inputs": {
            "records": {"path": str(source_path), "sha256": file_sha256(source_path)},
            "pair_bucket_records": {"path": str(bucket_path), "sha256": file_sha256(bucket_path)},
            "pair_bucket_metadata": {"path": str(metadata_path), "sha256": file_sha256(metadata_path)},
            "final_endpoint_pruning": pruning_input,
        },
        "summary": {
            "input_records": stream_stats["input_records"],
            "retrieval_eligible_input_records": stream_stats["eligible_records"],
            "deduplicated_input_records": stream_stats["eligible_records"],
            "collapsed_records": output_stats["collapsed_records"],
            "direct_mapping_records": stream_stats["direct_mapping_records"],
            "semantic_groups": semantic_stats["total"],
            "semantic_groups_completed": semantic_stats["completed"],
            "semantic_groups_pending": semantic_stats["pending"],
            "final_endpoint_pruning_excluded_records": len(pruned_record_ids),
            "excluded_reason_counts": stream_stats["excluded_reason_counts"],
            "aggregation_method_counts": output_stats[
                "aggregation_method_counts"
            ],
            "retrieval_source_counts": output_stats["retrieval_source_counts"],
            "source_records_represented": output_stats[
                "source_records_represented"
            ],
            "semantic_token_usage": output_stats["semantic_token_usage"],
        },
        "outputs": {
            RECORDS_FILENAME: file_sha256(records_file),
            SEMANTIC_FILENAME: file_sha256(semantic_file),
        },
        "validations": {
            "one_output_per_molecule_context": True,
            "direct_and_indirect_grouping_are_disjoint": True,
            "row_deduplication_performed_here": False,
            "no_canonical_claim_id": True,
            "configured_group_ids_preserved": output_stats[
                "configured_group_ids_preserved"
            ],
            "representative_context_discarded": True,
            "direct_records_are_not_assay_transferable": output_stats[
                "direct_records_are_not_assay_transferable"
            ],
            "pending_semantic_records_are_not_retrieval_eligible": output_stats[
                "pending_semantic_records_are_not_retrieval_eligible"
            ],
        },
    }
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _initialize_work_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE eligible (
            record_id TEXT PRIMARY KEY,
            group_key TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX eligible_group ON eligible(group_key, record_id);
        CREATE TABLE semantic (
            group_key TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        CREATE TABLE semantic_pending (
            group_key TEXT PRIMARY KEY,
            prompt_bytes INTEGER NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )


def _stream_joined_records(
    connection: sqlite3.Connection,
    *,
    records_path: Path,
    bucket_path: Path,
    columns: Sequence[str],
    pruned_record_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    record_batches = pq.ParquetFile(records_path).iter_batches(
        batch_size=5_000, columns=list(columns)
    )
    bucket_batches = pq.ParquetFile(bucket_path).iter_batches(batch_size=5_000)
    input_records = 0
    eligible_records = 0
    excluded: Counter[str] = Counter()
    seen_pruned: set[str] = set()
    sentinel = object()
    for record_batch, bucket_batch in itertools.zip_longest(
        record_batches, bucket_batches, fillvalue=sentinel
    ):
        if record_batch is sentinel or bucket_batch is sentinel:
            raise ValueError("Stage-03/Stage-04 batch coverage differs before collapse")
        records = record_batch.to_pylist()
        buckets = bucket_batch.to_pylist()
        if len(records) != len(buckets):
            raise ValueError("Stage-03/Stage-04 batch lengths differ before collapse")
        for record, sidecar in zip(records, buckets, strict=True):
            if str(record.get("canonical_record_id") or "") != str(
                sidecar.get("canonical_record_id") or ""
            ):
                raise ValueError("Stage-03/Stage-04 row order differs before collapse")
            record.update(
                {key: value for key, value in sidecar.items() if key != "canonical_record_id"}
            )
        for record in records:
            input_records += 1
            record_id = str(record["canonical_record_id"])
            if not record.get("retrieval_source_id"):
                record["retrieval_source_id"] = "indirect"
            if not bool(record.get("retrieval_eligible")):
                excluded["retrieval_ineligible"] += 1
                continue
            if not _text(record.get("canonical_smiles")):
                excluded["unresolved_structure"] += 1
                continue
            if record_id in pruned_record_ids:
                excluded["final_endpoint_pruning"] += 1
                seen_pruned.add(record_id)
                continue
            if (
                record.get("retrieval_source_id") == "indirect"
                and not _text(record.get("pair_bucket_key"))
            ):
                raise ValueError(
                    "retrieval-eligible indirect record has no pair bucket: "
                    + record_id
                )
            group_key = _collapse_group_key(record)
            connection.execute(
                "INSERT INTO eligible(record_id, group_key, payload) VALUES (?, ?, ?)",
                (
                    record_id,
                    group_key,
                    _json_dump(record),
                ),
            )
            eligible_records += 1
        connection.commit()
    missing_pruned = pruned_record_ids - seen_pruned
    if missing_pruned:
        raise ValueError(
            "final endpoint pruning references records that did not reach collapse: "
            f"{sorted(missing_pruned)[:10]}"
        )
    return {
        "input_records": input_records,
        "eligible_records": eligible_records,
        "direct_mapping_records": 0,
        "excluded_reason_counts": dict(sorted(excluded.items())),
    }


def _materialize_semantic_responses(
    connection: sqlite3.Connection,
    *,
    task_id: str,
    semantic_config: SemanticAggregationConfig | None,
    semantic_model: str | None,
    prior_semantic_paths: Sequence[str | Path],
    defer_semantic_aggregation: bool,
    direct_label_definition: str,
    semantic_source_columns: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    cache_paths: tuple[str | Path | None, ...] = (
        *prior_semantic_paths,
        semantic_config.cache_path if semantic_config else None,
    )
    prior = {} if defer_semantic_aggregation else load_semantic_cache(cache_paths)
    expected_model = semantic_config.model if semantic_config else semantic_model
    batch_size = max(32, semantic_config.workers if semantic_config else 32)
    semantic_count = 0

    for group_key, group in _iter_retained_groups(connection):
        if _aggregation_method(group) != "semantic_llm":
            continue
        payload = _semantic_payload(
            task_id,
            group_key,
            group,
            direct_label_definition=direct_label_definition,
            semantic_source_columns=semantic_source_columns,
        )
        prompt_sha256 = hashlib.sha256(render_prompt(payload).encode()).hexdigest()
        cached = prior.get(group_key)
        if (
            cached
            and cached.get("prompt_sha256") == prompt_sha256
            and cached.get("requested_model") == expected_model
        ):
            validate_semantic_response(cached.get("response"))
            connection.execute(
                "INSERT OR REPLACE INTO semantic VALUES (?, ?)",
                (group_key, _json_dump(cached)),
            )
            semantic_count += 1
            continue
        rendered = render_prompt(payload)
        connection.execute(
            "INSERT INTO semantic_pending VALUES (?, ?, ?, ?)",
            (
                group_key,
                len(rendered.encode("utf-8")),
                prompt_sha256,
                _json_dump(payload),
            ),
        )
    connection.commit()
    pending_count = connection.execute(
        "SELECT COUNT(*) FROM semantic_pending"
    ).fetchone()[0]
    if defer_semantic_aggregation:
        return {
            "total": semantic_count + pending_count,
            "completed": semantic_count,
            "pending": pending_count,
        }
    if pending_count and semantic_config is None:
        raise RuntimeError(
            f"{pending_count} semantic collapse group(s) require LLM aggregation; "
            "provide semantic aggregation endpoint settings"
        )
    cursor = connection.execute(
        "SELECT group_key, payload FROM semantic_pending "
        "ORDER BY prompt_bytes, group_key"
    )
    new_count = 0
    while True:
        remaining = (
            semantic_config.max_new_groups - new_count
            if semantic_config and semantic_config.max_new_groups is not None
            else batch_size
        )
        if remaining <= 0:
            break
        batch = cursor.fetchmany(min(batch_size, remaining))
        if not batch:
            break
        rows = aggregate_semantic_groups(
            {str(group_key): json.loads(payload) for group_key, payload in batch},
            config=semantic_config,
            prior_paths=(),
        )
        for row in rows:
            connection.execute(
                "INSERT OR REPLACE INTO semantic VALUES (?, ?)",
                (str(row["group_id"]), _json_dump(row)),
            )
        semantic_count += len(rows)
        new_count += len(rows)
        connection.commit()
    if new_count < pending_count:
        raise SemanticAggregationLimitReached(
            f"semantic aggregation pilot completed {new_count} new group(s); "
            f"{pending_count - new_count} remain and Stage 06 was not published"
        )
    connection.commit()
    return {
        "total": semantic_count,
        "completed": semantic_count,
        "pending": 0,
    }


def _write_collapsed_outputs(
    connection: sqlite3.Connection,
    *,
    task_id: str,
    records_file: Path,
    semantic_file: Path,
    dedup_lineage: Mapping[str, Sequence[Mapping[str, str]]],
    empty_records_schema: pa.Schema,
    preserved_columns: Sequence[str],
    defer_semantic_aggregation: bool,
    semantic_model: str | None,
) -> dict[str, Any]:
    records_jsonl = records_file.with_suffix(".jsonl")
    method_counts: Counter[str] = Counter()
    retrieval_counts: Counter[str] = Counter()
    informativeness_counts: Counter[str] = Counter()
    source_records_represented = 0
    collapsed_ids: set[str] = set()
    direct_transfer_valid = True
    configured_groups_preserved = True
    pending_retrieval_valid = True
    with records_jsonl.open("w", encoding="utf-8") as handle:
        for group_key, group in _iter_retained_groups(connection):
            method = _aggregation_method(group)
            semantic_row = connection.execute(
                "SELECT payload FROM semantic WHERE group_key=?", (group_key,)
            ).fetchone()
            collapsed = _aggregate_group(
                task_id,
                group_key,
                group,
                method=method,
                semantic=json.loads(semantic_row[0]) if semantic_row else None,
                dedup_lineage=dedup_lineage,
                preserved_columns=preserved_columns,
                allow_pending_semantic=defer_semantic_aggregation,
            )
            collapsed_id = str(collapsed["canonical_record_id"])
            if not collapsed_id or collapsed_id in collapsed_ids:
                raise ValueError("collapsed canonical record IDs must be nonempty and unique")
            collapsed_ids.add(collapsed_id)
            method_counts[method] += 1
            retrieval_counts[str(collapsed["retrieval_source_id"])] += 1
            if collapsed.get("direct_label_informativeness"):
                informativeness_counts[
                    str(collapsed["direct_label_informativeness"])
                ] += 1
            source_records_represented += int(collapsed["source_record_count"])
            if collapsed["retrieval_source_id"] in DIRECT_RETRIEVAL_SOURCES:
                direct_transfer_valid &= not bool(collapsed["assay_transfer_eligible"])
            configured_groups_preserved &= not str(collapsed["group_id"]).startswith(
                ("direct_vote.", "direct_residual.")
            )
            if collapsed["aggregation_status"] == "pending_semantic_aggregation":
                pending_retrieval_valid &= not bool(collapsed["retrieval_eligible"])
            handle.write(_json_dump(collapsed) + "\n")
    semantic_usage: Counter[str] = Counter()
    semantic_models: Counter[str] = Counter()
    with semantic_file.open("w", encoding="utf-8") as handle:
        for (payload,) in connection.execute(
            "SELECT payload FROM semantic ORDER BY group_key"
        ):
            row = json.loads(payload)
            for key, value in (row.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    semantic_usage[str(key)] += int(value)
            semantic_models[
                f"{row.get('requested_model') or 'unknown'} -> "
                f"{row.get('served_model') or 'unknown'}"
            ] += 1
            handle.write(_json_dump(row) + "\n")
        for group_key, prompt_sha256, payload in connection.execute(
            "SELECT group_key, prompt_sha256, payload FROM semantic_pending "
            "WHERE group_key NOT IN (SELECT group_key FROM semantic) "
            "ORDER BY group_key"
        ):
            row = json.loads(payload)
            handle.write(
                _json_dump(
                    {
                        "group_id": group_key,
                        "status": "pending_semantic_aggregation",
                        "requested_model": semantic_model,
                        "prompt_sha256": prompt_sha256,
                        "template_sha256": file_sha256(TEMPLATE_PATH),
                        "canonical_source_record_ids": row[
                            "canonical_source_record_ids"
                        ],
                    }
                )
                + "\n"
            )
    _jsonl_to_parquet(
        records_jsonl, records_file, empty_schema=empty_records_schema
    )
    records_jsonl.unlink()
    return {
        "collapsed_records": len(collapsed_ids),
        "aggregation_method_counts": dict(sorted(method_counts.items())),
        "retrieval_source_counts": dict(sorted(retrieval_counts.items())),
        "direct_label_informativeness_counts": dict(
            sorted(informativeness_counts.items())
        ),
        "source_records_represented": source_records_represented,
        "semantic_token_usage": dict(sorted(semantic_usage.items())),
        "semantic_models": dict(sorted(semantic_models.items())),
        "direct_records_are_not_assay_transferable": direct_transfer_valid,
        "configured_group_ids_preserved": configured_groups_preserved,
        "pending_semantic_records_are_not_retrieval_eligible": pending_retrieval_valid,
    }


def _iter_retained_groups(
    connection: sqlite3.Connection,
) -> Any:
    cursor = connection.execute(
        "SELECT group_key, payload FROM eligible ORDER BY group_key, record_id"
    )
    for group_key, sqlite_group in itertools.groupby(cursor, key=lambda row: row[0]):
        yield str(group_key), [json.loads(row[1]) for row in sqlite_group]


def _jsonl_to_parquet(
    jsonl_path: Path,
    parquet_path: Path,
    *,
    empty_schema: pa.Schema | None = None,
) -> None:
    schemas: list[pa.Schema] = []
    for rows in _jsonl_chunks(jsonl_path):
        schemas.append(pa.Table.from_pylist(rows).schema)
    if not schemas:
        schema = empty_schema or pa.schema([])
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in schema],
                schema=schema,
            ),
            parquet_path,
            compression="zstd",
        )
        return
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    with pq.ParquetWriter(
        parquet_path,
        schema,
        compression="zstd",
        compression_level=3,
    ) as writer:
        for rows in _jsonl_chunks(jsonl_path):
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _jsonl_chunks(path: Path, size: int = 5_000) -> Any:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= size:
                yield rows
                rows = []
    if rows:
        yield rows


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _projection_columns(
    available: set[str],
    semantic_columns: set[str],
    preserved_columns: Sequence[str],
) -> set[str]:
    return (_BASE_COLUMNS | semantic_columns | set(preserved_columns)) & available


def _empty_collapsed_schema(
    records_path: Path, bucket_path: Path, preserved_columns: set[str]
) -> pa.Schema:
    fields: dict[str, pa.Field] = {}
    source_schema = pq.read_schema(records_path)
    bucket_schema = pq.read_schema(bucket_path)
    for name in sorted(preserved_columns):
        if name in source_schema.names:
            fields[name] = source_schema.field(name)
    for name in (
        "pair_bucket_key",
        "canonical_pair_fields_json",
        "assay_transfer_eligible",
        "assay_transfer_ineligibility_reason",
    ):
        if name in bucket_schema.names:
            fields.setdefault(name, bucket_schema.field(name))
    appended = {
        "canonical_record_id": pa.string(),
        "collapsed_record_id": pa.string(),
        "source_record_id": pa.string(),
        "source_id": pa.string(),
        "canonical_smiles": pa.string(),
        "group_id": pa.string(),
        "collapse_group_key": pa.string(),
        "retrieval_source_id": pa.string(),
        "endpoint_name": pa.string(),
        "canonical_endpoint_name": pa.string(),
        "canonical_measurement_text": pa.string(),
        "canonical_unit_text": pa.string(),
        "finite_scalar_value": pa.float64(),
        "display_measurement_text": pa.string(),
        "display_scalar_value": pa.float64(),
        "display_unit_text": pa.string(),
        "display_transform_id": pa.string(),
        "measurement_kind": pa.string(),
        "canonical_measurement_scale_id": pa.string(),
        "canonical_category_id": pa.string(),
        "canonical_category_rank": pa.float64(),
        "retrieval_eligible": pa.bool_(),
        "aggregation_method": pa.string(),
        "aggregation_status": pa.string(),
        "direct_label_informativeness": pa.string(),
        "aggregate_counts_json": pa.string(),
        "source_record_count": pa.int64(),
        "deduplicated_source_record_count": pa.int64(),
        "canonical_source_record_ids_json": pa.string(),
        "source_ids_json": pa.string(),
        "pmids_json": pa.string(),
        "dois_json": pa.string(),
        "aggregate_min": pa.float64(),
        "aggregate_max": pa.float64(),
        "aggregate_q1": pa.float64(),
        "aggregate_q3": pa.float64(),
        "condition_group": pa.string(),
        "condition_scope": pa.string(),
        "condition_key_status": pa.string(),
        "condition_atoms_json": pa.string(),
    }
    for name, data_type in appended.items():
        fields.setdefault(name, pa.field(name, data_type))
    return pa.schema(fields.values())


def _collapse_group_key(record: Mapping[str, Any]) -> str:
    source = str(record["retrieval_source_id"])
    smiles = str(record["canonical_smiles"])
    if source == "direct_vote":
        identity = [source, smiles, str(record["condition_group"])]
    elif source == "direct_residual":
        identity = [
            source,
            smiles,
            str(record["condition_group"]),
            str(record["condition_key_status"]),
        ]
    else:
        identity = [source, smiles, str(record["pair_bucket_key"])]
    return hashlib.sha256("\0".join(identity).encode()).hexdigest()


def _aggregation_method(group: Sequence[Mapping[str, Any]]) -> str:
    if str(group[0]["retrieval_source_id"]) == "direct_vote":
        return "direct_binary_vote"
    if all(_text(row.get("canonical_category_id")) for row in group):
        return "categorical_mode"
    if all(
        str(row.get("measurement_kind") or "") == "continuous"
        and _finite(row.get("finite_scalar_value"))
        and str(row.get("canonical_unit_text") or "")
        not in {"relative-scalar", "unresolved-scalar", "free-text"}
        for row in group
    ):
        return "continuous_median"
    if len(group) == 1 and _finite(group[0].get("finite_scalar_value")):
        return "single_record_passthrough"
    return "semantic_llm"


def _aggregate_group(
    task_id: str,
    group_key: str,
    group: list[dict[str, Any]],
    *,
    method: str,
    semantic: Mapping[str, Any] | None,
    dedup_lineage: Mapping[str, Sequence[Mapping[str, str]]],
    preserved_columns: Sequence[str],
    allow_pending_semantic: bool = False,
) -> dict[str, Any]:
    raw_ids: list[str] = []
    source_ids = {
        str(row.get("source_id")) for row in group if row.get("source_id")
    }
    pmids = {_text(row.get("pmid")) for row in group if _text(row.get("pmid"))}
    dois = {_text(row.get("doi")) for row in group if _text(row.get("doi"))}
    for row in group:
        record_id = str(row["canonical_record_id"])
        raw_ids.append(record_id)
        for duplicate in dedup_lineage.get(record_id, ()):
            raw_ids.append(str(duplicate["canonical_record_id"]))
            if duplicate_source := str(duplicate.get("source_id") or ""):
                source_ids.add(duplicate_source)
            if duplicate.get("pmid"):
                pmids.add(str(duplicate["pmid"]))
    retrieval_source = str(group[0]["retrieval_source_id"])
    group_ids = {
        _text(row.get("direct_group_id") or row.get("group_id")) for row in group
    }
    if len(group_ids) != 1 or not next(iter(group_ids)):
        raise ValueError(f"collapse group spans configured group IDs: {group_key}")
    group_id = next(iter(group_ids))
    smiles = _one_value(group, "canonical_smiles", group_key)
    pair_context = _pair_context(group[0]) if retrieval_source == "indirect" else {}
    output = {
        field: _common_value(group, field)
        for field in preserved_columns
        if any(field in row for row in group)
    }
    output.update(pair_context)
    output.update(
        {
            "canonical_smiles": smiles,
            "group_id": group_id,
            "source_id": (
                next(iter(source_ids)) if len(source_ids) == 1 else "multi_source"
            ),
            "endpoint_name": (
                group_id
                if retrieval_source in DIRECT_RETRIEVAL_SOURCES
                else _one_value(group, "canonical_endpoint_name", group_key)
            ),
            "canonical_endpoint_name": (
                group_id
                if retrieval_source in DIRECT_RETRIEVAL_SOURCES
                else _one_value(group, "canonical_endpoint_name", group_key)
            ),
            "canonical_unit_text": (
                "binary_outcome"
                if method == "direct_binary_vote"
                else _common_value(group, "canonical_unit_text")
            ),
            "measurement_kind": _common_value(group, "measurement_kind"),
            "canonical_measurement_scale_id": _common_value(
                group, "canonical_measurement_scale_id"
            ),
            "pair_bucket_key": (
                _one_value(group, "pair_bucket_key", group_key)
                if retrieval_source == "indirect"
                else None
            ),
            "canonical_pair_fields_json": (
                _common_value(group, "canonical_pair_fields_json")
                if retrieval_source == "indirect"
                else None
            ),
        }
    )
    status = "valid"
    counts: dict[str, int] = {}
    output.update(
        {
            "finite_scalar_value": None,
            "canonical_category_id": None,
            "canonical_category_rank": None,
            "canonical_measurement_text": None,
            "aggregate_min": None,
            "aggregate_max": None,
            "aggregate_q1": None,
            "aggregate_q3": None,
            "direct_label_informativeness": None,
        }
    )
    if method == "direct_binary_vote":
        vote_labels: dict[str, int] = {}
        for row in group:
            vote_unit = str(
                row.get("direct_vote_unit_id") or row["canonical_record_id"]
            )
            label = int(row["direct_vote_label"])
            prior = vote_labels.get(vote_unit)
            if prior is not None and prior != label:
                raise ValueError(f"direct vote unit has conflicting labels: {vote_unit}")
            vote_labels[vote_unit] = label
            for duplicate in dedup_lineage.get(str(row["canonical_record_id"]), ()):
                duplicate_unit = str(
                    duplicate.get("direct_vote_unit_id")
                    or duplicate["canonical_record_id"]
                )
                duplicate_label = duplicate.get("direct_vote_label")
                if duplicate_label not in {0, 1}:
                    continue
                prior = vote_labels.get(duplicate_unit)
                if prior is not None and prior != int(duplicate_label):
                    raise ValueError(
                        f"direct vote unit has conflicting labels: {duplicate_unit}"
                    )
                vote_labels[duplicate_unit] = int(duplicate_label)
        counts = dict(
            sorted(
                Counter(str(label) for label in vote_labels.values()).items()
            )
        )
        n0, n1 = counts.get("0", 0), counts.get("1", 0)
        threshold = (
            CONDITIONED_AGREEMENT_THRESHOLD
            if str(group[0].get("condition_scope") or "") == "external"
            else UNCONDITIONED_AGREEMENT_THRESHOLD
        )
        agreement = max(n0, n1) / len(vote_labels)
        if n0 == n1 or agreement < threshold:
            retrieval_source = "direct_residual"
            status = "conflict"
            value = None
        else:
            value = 1 if n1 > n0 else 0
        output["finite_scalar_value"] = float(value) if value is not None else None
        output["canonical_category_id"] = (
            "positive" if value == 1 else "negative" if value == 0 else None
        )
        output["canonical_category_rank"] = value
        output["canonical_measurement_text"] = (
            output["canonical_category_id"] if value is not None else None
        )
        output["measurement_kind"] = "binary"
        output["canonical_measurement_scale_id"] = "direct_binary_vote"
    elif method == "categorical_mode":
        counts = dict(
            sorted(Counter(str(row["canonical_category_id"]) for row in group).items())
        )
        maximum = max(counts.values())
        modes = sorted(key for key, count in counts.items() if count == maximum)
        value = modes[0] if len(modes) == 1 else None
        status = "valid" if value is not None else "conflict"
        output["canonical_category_id"] = value
        ranks = {
            str(row["canonical_category_id"]): row.get("canonical_category_rank")
            for row in group
        }
        output["canonical_category_rank"] = ranks.get(value) if value else None
        output["finite_scalar_value"] = (
            float(output["canonical_category_rank"])
            if _finite(output["canonical_category_rank"])
            else None
        )
        output["canonical_measurement_text"] = value
    elif method == "continuous_median":
        values = [float(row["finite_scalar_value"]) for row in group]
        median = float(statistics.median(values))
        output["finite_scalar_value"] = median
        output["canonical_measurement_text"] = format(median, ".12g")
        output["aggregate_min"] = min(values)
        output["aggregate_max"] = max(values)
        output["aggregate_q1"] = float(np.quantile(values, 0.25))
        output["aggregate_q3"] = float(np.quantile(values, 0.75))
    elif method == "single_record_passthrough":
        record = group[0]
        value = float(record["finite_scalar_value"])
        output["finite_scalar_value"] = value
        output["canonical_measurement_text"] = format(value, ".12g")
    else:
        if semantic is None:
            if not allow_pending_semantic:
                raise ValueError(
                    f"semantic group lacks a validated response: {group_key}"
                )
            status = "pending_semantic_aggregation"
        else:
            semantic_response = dict(semantic["response"])
            output["canonical_measurement_text"] = semantic_response.get("summary")
            output["direct_label_informativeness"] = semantic_response[
                "direct_label_informativeness"
            ]
        output["measurement_kind"] = "semantic"
    output.update(display_measurement_tuple(output))
    collapsed_id = hashlib.sha256(
        f"{task_id}\0{group_key}".encode()
    ).hexdigest()
    transfer_reasons = sorted(
        {
            _text(row.get("assay_transfer_ineligibility_reason"))
            for row in group
            if _text(row.get("assay_transfer_ineligibility_reason"))
        }
    )
    transfer_eligible = bool(
        retrieval_source == "indirect"
        and all(bool(row.get("assay_transfer_eligible")) for row in group)
    )
    vote_units = {
        str(row.get("direct_vote_unit_id") or row["canonical_record_id"])
        for row in group
    } | {
        str(item.get("direct_vote_unit_id") or item["canonical_record_id"])
        for row in group
        for item in dedup_lineage.get(str(row["canonical_record_id"]), ())
    }
    output.update(
        {
            "canonical_record_id": collapsed_id,
            "collapsed_record_id": collapsed_id,
            "source_record_id": collapsed_id,
            "retrieval_source_id": retrieval_source,
            "collapse_group_key": group_key,
            "aggregation_method": method,
            "aggregation_status": status,
            "aggregate_counts_json": json.dumps(counts, sort_keys=True),
            "source_record_count": (
                len(vote_units)
                if method == "direct_binary_vote"
                else len(raw_ids)
            ),
            "deduplicated_source_record_count": len(group),
            "canonical_source_record_ids_json": json.dumps(sorted(raw_ids)),
            "source_ids_json": json.dumps(sorted(source_ids)),
            "pmids_json": json.dumps(sorted(pmids)),
            "dois_json": json.dumps(sorted(dois)),
            "retrieval_eligible": status != "pending_semantic_aggregation",
            "assay_transfer_eligible": (
                transfer_eligible and status != "pending_semantic_aggregation"
            ),
            "assay_transfer_ineligibility_reason": (
                "pending_semantic_aggregation"
                if status == "pending_semantic_aggregation"
                else None
                if transfer_eligible
                else transfer_reasons[0]
                if retrieval_source == "indirect" and len(transfer_reasons) == 1
                else "mixed_or_ineligible_source_records"
                if retrieval_source == "indirect"
                else "direct_or_contextual_outcome_not_assay_transferable"
            ),
        }
    )
    if retrieval_source in DIRECT_RETRIEVAL_SOURCES:
        output.update(
            {
                "condition_group": _one_value(group, "condition_group", group_key),
                "condition_scope": _common_value(group, "condition_scope"),
                "condition_key_status": _common_value(
                    group, "condition_key_status"
                ),
                "condition_atoms_json": json.dumps(
                    sorted(
                        {
                            str(atom)
                            for row in group
                            for atom in (row.get("condition_atoms") or ())
                        }
                    )
                ),
            }
        )
    return output


def _semantic_payload(
    task_id: str,
    group_key: str,
    group: Sequence[Mapping[str, Any]],
    *,
    direct_label_definition: str = "the task's direct binary outcome",
    semantic_source_columns: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    del task_id, group_key
    source_columns = semantic_source_columns or {}
    return {
        "direct_label_definition": direct_label_definition,
        "canonical_source_record_ids": sorted(
            str(row["canonical_record_id"]) for row in group
        ),
        "records": [
            _semantic_record(
                row,
                source_columns.get(str(row.get("source_id") or ""), ()),
            )
            for row in group
        ],
    }


def _semantic_record(
    record: Mapping[str, Any], context_columns: Sequence[str]
) -> dict[str, Any]:
    display = display_measurement_tuple(record)
    fields: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, key, value in (
        ("Molecule", "molecule_name", record.get("molecule_name")),
        ("Endpoint", "canonical_endpoint_name", record.get("canonical_endpoint_name")),
        ("Result", "display_measurement_text", display.get("display_measurement_text")),
        ("Unit", "display_unit_text", display.get("display_unit_text")),
    ):
        text = _prompt_text(value)
        if text:
            fields.append({"label": label, "value": text})
            seen.add(key)
    for key in context_columns:
        if key in seen:
            continue
        text = _prompt_text(record.get(key))
        if text:
            fields.append(
                {"label": str(key).replace("_", " ").capitalize(), "value": text}
            )
            seen.add(key)
    return {"fields": fields}


def _prompt_text(value: Any) -> str:
    """Remove structure identifiers from otherwise useful scientific text."""
    text = _text(value)
    if not text:
        return ""
    text = _PARENTHETICAL_SMILES_RE.sub("", text)
    text = _COMPOUND_SMILES_RE.sub("the tested compound", text)
    text = _SMILES_RE.sub("the tested compound", text)
    return " ".join(text.split())


def _pair_context(record: Mapping[str, Any]) -> Any:
    raw = record.get("canonical_pair_fields_json")
    try:
        return json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {"pair_bucket_key": record.get("pair_bucket_key")}


def _common_value(group: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = {_json_dump(row.get(field)): row.get(field) for row in group}
    return next(iter(values.values())) if len(values) == 1 else None


def _one_value(
    group: Sequence[Mapping[str, Any]], field: str, group_key: str
) -> Any:
    value = _common_value(group, field)
    if value in {None, ""}:
        raise ValueError(f"collapse group {group_key} spans or lacks {field}")
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "COLLAPSE_VERSION",
    "DIRECT_RETRIEVAL_SOURCES",
    "MANIFEST_FILENAME",
    "RECORDS_FILENAME",
    "SEMANTIC_FILENAME",
    "build_collapsed_record_stage",
]
