"""Bounded-memory row deduplication for canonical Starling v7 records."""

from __future__ import annotations

import itertools
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256


DEDUPLICATION_VERSION = "starling_record_deduplication.v6"
RECORDS_FILENAME = "records.parquet"
PAIR_BUCKET_RECORDS_FILENAME = "pair_bucket_records.parquet"
DIRECT_MAPPING_FILENAME = "direct_record_mapping.parquet"
DUPLICATES_FILENAME = "duplicates.parquet"
MANIFEST_FILENAME = "manifest.json"
DIRECT_RETRIEVAL_SOURCES = frozenset({"direct_vote", "direct_residual"})
TRUSTED_DIRECT_CONDITION_STATUSES = frozenset(
    {"selected_reviewed", "accepted_unselected", "none_reported"}
)
_PAIR_BUCKET_FIELD_TYPES = {
    "pair_bucket_key": pa.string(),
    "canonical_pair_fields_json": pa.string(),
    "assay_transfer_eligible": pa.bool_(),
    "assay_transfer_ineligibility_reason": pa.string(),
    "bucket_eligible": pa.bool_(),
    "bucket_exclusion_reason": pa.string(),
}
SUPPORT_JACCARD_MINIMUM = 0.80
_TOKEN = re.compile(r"[a-z0-9]+")
_CONTEXT_FIELDS = (
    "canonical_assay_context",
    "canonical_species_context",
    "canonical_reference_scope",
    "canonical_reference_basis",
    "canonical_biological_matrix",
    "qualifying_conditions",
    "species_or_population",
    "assay_or_test",
    "dose",
    "oral_dose",
    "dose_or_concentration",
    "formulation_or_solid_form",
    "formulation_vehicle",
    "exposure_time",
    "study_design",
    "light_conditions",
)
_CANDIDATE_FIELDS = frozenset(
    {
        "canonical_record_id",
        "canonical_claim_id",
        "canonical_claim_parent_identity_key",
        "canonical_claim_representative",
        "source_row_uid",
        "source_id",
        "source_smiles",
        "canonical_smiles",
        "group_id",
        "retrieval_source_id",
        "support_text",
        "pmid",
        "doi",
        "direct_vote_label",
        "direct_vote_unit_id",
        "condition_group",
        "measurement_kind",
        "canonical_measurement_text",
        "measurement_text",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "direct_residual_endpoint_name",
        "canonical_endpoint_name",
        "endpoint_name",
        "canonical_category_id",
        "canonical_category_rank",
        "finite_scalar_value",
        "canonical_pair_fields_json",
        "pair_bucket_key",
        "deduplication_context_id",
        "evidence_context_json",
        "assay_transfer_prebase_measurement_text",
        "assay_transfer_prebase_unit_text",
        *_CONTEXT_FIELDS,
    }
)


def build_deduplicated_record_stage(
    *,
    task_id: str,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    out_dir: str | Path,
    direct_mapping_builder: (
        Callable[[Sequence[Mapping[str, Any]]], list[dict[str, Any]]] | None
    ) = None,
    collapse_duplicates: bool = True,
) -> dict[str, Any]:
    """Publish aligned records, optionally applying the legacy late deduplicator."""
    source_path = Path(records_path)
    sidecar_path = Path(pair_bucket_records_path)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    work_dir = tempfile.TemporaryDirectory(prefix="starling-record-deduplication-")
    database = Path(work_dir.name) / "records.sqlite"
    connection = sqlite3.connect(database)
    _initialize(connection)
    input_records, mapping_records = _index_candidates(
        connection,
        records_path=source_path,
        sidecar_path=sidecar_path,
        direct_mapping_builder=direct_mapping_builder,
    )
    duplicate_counts = _deduplicate(connection) if collapse_duplicates else Counter()
    output_records = _write_retained_records(
        connection,
        records_path=source_path,
        sidecar_path=sidecar_path,
        records_output=target / RECORDS_FILENAME,
        sidecar_output=target / PAIR_BUCKET_RECORDS_FILENAME,
        direct_mapping_builder=direct_mapping_builder,
    )
    _write_sqlite_json_parquet(
        connection,
        query="SELECT payload FROM direct_mapping ORDER BY record_id",
        output=target / DIRECT_MAPPING_FILENAME,
    )
    _write_sqlite_json_parquet(
        connection,
        query="SELECT payload FROM duplicates ORDER BY discarded_record_id",
        output=target / DUPLICATES_FILENAME,
    )
    connection.close()
    work_dir.cleanup()
    sidecar_records = pq.ParquetFile(
        target / PAIR_BUCKET_RECORDS_FILENAME
    ).metadata.num_rows
    written_mappings = pq.ParquetFile(target / DIRECT_MAPPING_FILENAME).metadata.num_rows
    written_duplicates = pq.ParquetFile(target / DUPLICATES_FILENAME).metadata.num_rows
    validations = {
        "retained_plus_duplicates_equals_input": (
            output_records + written_duplicates == input_records
        ),
        "deduplicated_records_align_with_pair_sidecar": (
            sidecar_records == output_records
        ),
        "all_physical_direct_vote_units_preserved_in_mapping": (
            written_mappings == mapping_records
        ),
    }
    if not all(validations.values()):
        raise ValueError(f"row deduplication validation failed: {validations}")
    outputs = {
        name: file_sha256(target / name)
        for name in (
            RECORDS_FILENAME,
            PAIR_BUCKET_RECORDS_FILENAME,
            DIRECT_MAPPING_FILENAME,
            DUPLICATES_FILENAME,
        )
    }
    manifest = {
        "version": DEDUPLICATION_VERSION,
        "task_id": task_id,
        "contract": {
            "mode": (
                "legacy_late_deduplication"
                if collapse_duplicates
                else "authoritative_stage1_passthrough"
            ),
            "within_source": "exact_semantic_row",
            "cross_source": "equal_measurement_context_and_supported_claim",
            "paper_support_token_jaccard_minimum": SUPPORT_JACCARD_MINIMUM,
            "direct_partitions_kept_separate": True,
            "conflicting_direct_labels_merge": False,
            "direct_mapping_grain": "normalized_row_with_optional_canonical_claim_id",
            "direct_pair_bucket": "intrinsic_stage03_pair_bucket",
            "row_deletion_performed": collapse_duplicates,
        },
        "inputs": {
            "records": {"path": str(source_path), "sha256": file_sha256(source_path)},
            "pair_bucket_records": {
                "path": str(sidecar_path),
                "sha256": file_sha256(sidecar_path),
            },
        },
        "summary": {
            "input_records": input_records,
            "retained_records": output_records,
            "duplicates_removed": sum(duplicate_counts.values()),
            "duplicate_scope_counts": dict(sorted(duplicate_counts.items())),
            "direct_mapping_records": mapping_records,
        },
        "outputs": outputs,
        "validations": validations,
    }
    (target / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_duplicate_lineage(path: str | Path) -> dict[str, list[dict[str, str]]]:
    lineage: dict[str, list[dict[str, str]]] = defaultdict(list)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=5_000):
        for row in batch.to_pylist():
            lineage[str(row["retained_canonical_record_id"])].append(
                {
                    "canonical_record_id": str(row["discarded_canonical_record_id"]),
                    "source_id": str(row.get("discarded_source_id") or ""),
                    "pmid": _text(row.get("discarded_pmid")),
                    "direct_vote_unit_id": _text(
                        row.get("discarded_direct_vote_unit_id")
                    ),
                    "direct_vote_label": row.get("discarded_direct_vote_label"),
                }
            )
    return dict(lineage)


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE candidates (
            ordinal INTEGER PRIMARY KEY,
            record_id TEXT NOT NULL UNIQUE,
            dedup_key TEXT NOT NULL,
            payload TEXT NOT NULL,
            retained INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX candidates_dedup ON candidates(dedup_key, record_id);
        CREATE TABLE direct_mapping (
            record_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        CREATE TABLE duplicates (
            discarded_record_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        """
    )


def _aligned_batches(records_path: Path, sidecar_path: Path):
    records = pq.ParquetFile(records_path).iter_batches(batch_size=5_000)
    sidecars = pq.ParquetFile(sidecar_path).iter_batches(batch_size=5_000)
    sentinel = object()
    for record_batch, sidecar_batch in itertools.zip_longest(
        records, sidecars, fillvalue=sentinel
    ):
        if record_batch is sentinel or sidecar_batch is sentinel:
            raise ValueError("Stage-03/Stage-04 batch coverage differs before deduplication")
        record_rows, sidecar_rows = record_batch.to_pylist(), sidecar_batch.to_pylist()
        if len(record_rows) != len(sidecar_rows):
            raise ValueError("Stage-03/Stage-04 batch lengths differ before deduplication")
        for record, sidecar in zip(record_rows, sidecar_rows, strict=True):
            if str(record.get("canonical_record_id") or "") != str(
                sidecar.get("canonical_record_id") or ""
            ):
                raise ValueError("Stage-03/Stage-04 row order differs before deduplication")
        yield record_rows, sidecar_rows


def _index_candidates(
    connection: sqlite3.Connection,
    *,
    records_path: Path,
    sidecar_path: Path,
    direct_mapping_builder: Callable | None,
) -> tuple[int, int]:
    ordinal = 0
    mapping_count = 0
    for records, sidecars in _aligned_batches(records_path, sidecar_path):
        records_by_id = _unique_by_id(records)
        if any(not record.get("source_row_uid") for record in records):
            raise ValueError("Stage-2 records require source_row_uid")
        mappings = direct_mapping_builder(records) if direct_mapping_builder else []
        mapping_by_id = _unique_by_id(mappings)
        for mapping in mappings:
            mapping = {
                **mapping,
                "source_row_uid": str(
                    records_by_id[str(mapping["canonical_record_id"])].get(
                        "source_row_uid"
                    )
                    or ""
                ),
                "dedup_status": "retained",
                "dedup_retained_canonical_record_id": None,
            }
            connection.execute(
                "INSERT INTO direct_mapping VALUES (?, ?)",
                (str(mapping["canonical_record_id"]), _json_dump(mapping)),
            )
        mapping_count += len(mappings)
        for record, sidecar in zip(records, sidecars, strict=True):
            record_id = str(record.get("canonical_record_id") or "")
            if not record_id:
                raise ValueError("canonical_record_id must be nonempty")
            joined = _decorate(record, sidecar, mapping_by_id.get(record_id))
            candidate = _candidate_view(joined)
            connection.execute(
                "INSERT INTO candidates(ordinal, record_id, dedup_key, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    ordinal,
                    record_id,
                    _json_dump(_dedup_key(candidate)),
                    _json_dump(candidate),
                ),
            )
            ordinal += 1
        connection.commit()
    return ordinal, mapping_count


def _deduplicate(connection: sqlite3.Connection) -> Counter[str]:
    counts: Counter[str] = Counter()
    cursor = connection.execute(
        "SELECT dedup_key, payload FROM candidates ORDER BY dedup_key, record_id"
    )
    for _, sqlite_group in itertools.groupby(cursor, key=lambda row: row[0]):
        rows = [json.loads(row[1]) for row in sqlite_group]
        clusters: list[list[dict[str, Any]]] = []
        for row in sorted(rows, key=_representative_key):
            compatible = [
                cluster
                for cluster in clusters
                if all(_dedup_match(row, member) is not None for member in cluster)
            ]
            if compatible:
                compatible[0].append(row)
            else:
                clusters.append([row])
        for cluster in clusters:
            retained = min(cluster, key=_representative_key)
            retained_id = str(retained["canonical_record_id"])
            for discarded in cluster:
                if discarded is retained:
                    continue
                match = _dedup_match(retained, discarded)
                if match is None:
                    raise AssertionError("complete-link dedup cluster lost compatibility")
                discarded_id = str(discarded["canonical_record_id"])
                scope = (
                    "within_source"
                    if retained.get("source_id") == discarded.get("source_id")
                    else "cross_source"
                )
                audit = {
                    "retained_canonical_record_id": retained_id,
                    "discarded_canonical_record_id": discarded_id,
                    "source_row_uid": str(discarded.get("source_row_uid") or ""),
                    "retained_source_row_uid": str(
                        retained.get("source_row_uid") or ""
                    ),
                    "retained_source_id": str(retained.get("source_id") or ""),
                    "discarded_source_id": str(discarded.get("source_id") or ""),
                    "retrieval_source_id": str(retained["retrieval_source_id"]),
                    "canonical_smiles": str(retained.get("canonical_smiles") or ""),
                    "paper_id": _paper_id(retained),
                    "discarded_pmid": _text(discarded.get("pmid")),
                    "discarded_direct_vote_unit_id": _text(
                        discarded.get("direct_vote_unit_id")
                    ),
                    "discarded_direct_vote_label": discarded.get("direct_vote_label"),
                    "canonical_claim_id": _text(discarded.get("canonical_claim_id")),
                    "duplicate_scope": scope,
                    "support_token_jaccard": match[0],
                    "measurement_signature": _json_dump(
                        _measurement_signature(retained)
                    ),
                    "match_method": match[1],
                }
                connection.execute(
                    "UPDATE candidates SET retained=0 WHERE record_id=?", (discarded_id,)
                )
                mapping = connection.execute(
                    "SELECT payload FROM direct_mapping WHERE record_id=?", (discarded_id,)
                ).fetchone()
                if mapping:
                    value = json.loads(mapping[0])
                    value.update(
                        {
                            "dedup_status": f"{scope}_duplicate",
                            "dedup_retained_canonical_record_id": retained_id,
                        }
                    )
                    connection.execute(
                        "UPDATE direct_mapping SET payload=? WHERE record_id=?",
                        (_json_dump(value), discarded_id),
                    )
                connection.execute(
                    "INSERT INTO duplicates VALUES (?, ?)",
                    (discarded_id, _json_dump(audit)),
                )
                counts[scope] += 1
        connection.commit()
    return counts


def _write_retained_records(
    connection: sqlite3.Connection,
    *,
    records_path: Path,
    sidecar_path: Path,
    records_output: Path,
    sidecar_output: Path,
    direct_mapping_builder: Callable | None,
) -> int:
    record_writer = sidecar_writer = None
    retained = 0
    record_schema = _retained_record_schema(records_path, sidecar_path)
    sidecar_fields = {field.name: field for field in pq.read_schema(sidecar_path)}
    sidecar_fields.setdefault("source_row_uid", pa.field("source_row_uid", pa.string()))
    for name, data_type in _PAIR_BUCKET_FIELD_TYPES.items():
        if name not in sidecar_fields or pa.types.is_null(sidecar_fields[name].type):
            sidecar_fields[name] = pa.field(name, data_type)
    sidecar_schema = pa.schema(sidecar_fields.values())
    try:
        for records, sidecars in _aligned_batches(records_path, sidecar_path):
            mappings = direct_mapping_builder(records) if direct_mapping_builder else []
            mapping_by_id = _unique_by_id(mappings)
            record_ids = [str(record["canonical_record_id"]) for record in records]
            placeholders = ",".join("?" for _ in record_ids)
            retained_ids = {
                record_id
                for (record_id,) in connection.execute(
                    f"SELECT record_id FROM candidates "
                    f"WHERE retained=1 AND record_id IN ({placeholders})",
                    record_ids,
                )
            }
            kept_records: list[dict[str, Any]] = []
            kept_sidecars: list[dict[str, Any]] = []
            for record, sidecar in zip(records, sidecars, strict=True):
                record_id = str(record["canonical_record_id"])
                if record_id not in retained_ids:
                    continue
                decorated = _decorate(record, sidecar, mapping_by_id.get(record_id))
                kept_records.append(decorated)
                kept_sidecars.append(
                    {
                        **sidecar,
                        "source_row_uid": decorated.get("source_row_uid"),
                        **{
                            field: decorated.get(field)
                            for field in (
                                "pair_bucket_key",
                                "canonical_pair_fields_json",
                                "assay_transfer_eligible",
                                "assay_transfer_ineligibility_reason",
                                "bucket_eligible",
                                "bucket_exclusion_reason",
                            )
                        },
                    }
                )
            if not kept_records:
                continue
            record_table = pa.Table.from_pylist(kept_records, schema=record_schema)
            sidecar_table = pa.Table.from_pylist(kept_sidecars, schema=sidecar_schema)
            if record_writer is None:
                record_writer = pq.ParquetWriter(
                    records_output, record_schema, compression="zstd", compression_level=3
                )
                sidecar_writer = pq.ParquetWriter(
                    sidecar_output, sidecar_schema, compression="zstd", compression_level=3
                )
            record_writer.write_table(record_table)
            sidecar_writer.write_table(sidecar_table)
            retained += len(kept_records)
    finally:
        if record_writer is not None:
            record_writer.close()
            sidecar_writer.close()
    if record_writer is None:
        pq.write_table(pa.table({}), records_output, compression="zstd")
        pq.write_table(pa.table({}), sidecar_output, compression="zstd")
    return retained


def _retained_record_schema(records_path: Path, sidecar_path: Path) -> pa.Schema:
    fields = {field.name: field for field in pq.read_schema(records_path)}
    for field in pq.read_schema(sidecar_path):
        if field.name != "canonical_record_id":
            fields.setdefault(field.name, field)
    appended = {
        **_PAIR_BUCKET_FIELD_TYPES,
        "retrieval_source_id": pa.string(),
        "direct_vote_label": pa.int64(),
        "direct_vote_unit_id": pa.string(),
        "condition_group": pa.string(),
        "condition_atoms": pa.list_(pa.string()),
        "condition_scope": pa.string(),
        "condition_key_status": pa.string(),
        "direct_group_id": pa.string(),
        "direct_residual_endpoint_name": pa.string(),
    }
    for name, data_type in appended.items():
        if name not in fields or pa.types.is_null(fields[name].type):
            fields[name] = pa.field(name, data_type)
    return pa.schema(fields.values())


def _write_sqlite_json_parquet(
    connection: sqlite3.Connection, *, query: str, output: Path
) -> None:
    cursor = connection.execute(query)
    schemas: list[pa.Schema] = []
    while batch := cursor.fetchmany(5_000):
        rows = [json.loads(payload) for (payload,) in batch]
        schemas.append(pa.Table.from_pylist(rows).schema)
    if not schemas:
        pq.write_table(pa.table({}), output, compression="zstd")
        return
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    with pq.ParquetWriter(
        output, schema, compression="zstd", compression_level=3
    ) as writer:
        cursor = connection.execute(query)
        while batch := cursor.fetchmany(5_000):
            writer.write_table(
                pa.Table.from_pylist(
                    [json.loads(payload) for (payload,) in batch], schema=schema
                )
            )


def _decorate(
    record: Mapping[str, Any], sidecar: Mapping[str, Any], mapping: Mapping[str, Any] | None
) -> dict[str, Any]:
    output = dict(record)
    output.update({key: value for key, value in sidecar.items() if key != "canonical_record_id"})
    if mapping is None:
        output["retrieval_source_id"] = "indirect"
    else:
        condition_group = str(mapping["condition_group"])
        condition_status = str(mapping["condition_key_status"])
        output.update(
            {
                "retrieval_source_id": mapping["retrieval_source_id"],
                "direct_vote_label": mapping.get("direct_vote_label"),
                "direct_vote_unit_id": mapping.get("direct_vote_unit_id")
                or f"{record.get('source_id')}:row:{int(record.get('source_row_number') or 0)}",
                "condition_group": condition_group,
                "condition_atoms": mapping["condition_atoms"],
                "condition_scope": mapping["condition_scope"],
                "condition_key_status": condition_status,
                "direct_group_id": mapping.get("direct_group_id"),
                "direct_residual_endpoint_name": mapping.get(
                    "direct_residual_endpoint_name"
                ),
                "canonical_claim_id": mapping.get("canonical_claim_id"),
                "canonical_claim_parent_identity_key": mapping.get(
                    "canonical_claim_parent_identity_key"
                ),
                "canonical_claim_representative": mapping.get(
                    "canonical_claim_representative"
                ),
            }
        )
        base_pair_key = _text(output.get("pair_bucket_key"))
        if not base_pair_key:
            output["retrieval_eligible"] = False
            output["organization_status"] = "missing_pair_bucket"
            output["assay_transfer_eligible"] = False
            output["assay_transfer_ineligibility_reason"] = "missing_pair_bucket"
        elif condition_status not in TRUSTED_DIRECT_CONDITION_STATUSES:
            output["assay_transfer_eligible"] = False
            output["assay_transfer_ineligibility_reason"] = (
                "untrusted_direct_condition_key"
            )
        elif not bool(output.get("assay_transfer_eligible")) and not _text(
            output.get("assay_transfer_ineligibility_reason")
        ):
            output["assay_transfer_ineligibility_reason"] = (
                "ineligible_source_pair_bucket"
            )
    output["bucket_eligible"] = bool(output.get("assay_transfer_eligible"))
    output["bucket_exclusion_reason"] = output.get(
        "assay_transfer_ineligibility_reason"
    )
    return output


def _candidate_view(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in _CANDIDATE_FIELDS}


def _unique_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = _text(row.get("canonical_record_id"))
        if not key or key in output:
            raise ValueError("direct mapping canonical_record_id values must be unique")
        output[key] = row
    return output


def _dedup_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    if claim_id := _text(record.get("canonical_claim_id")):
        return "canonical_direct_claim", claim_id
    support = _normalized_text(record.get("support_text"))
    paper = _paper_id(record)
    return (
        str(record["retrieval_source_id"]),
        str(record.get("canonical_smiles") or ""),
        _dedup_context_key(record),
        _measurement_signature(record),
        paper or f"support:{support}",
    )


def _dedup_context_key(record: Mapping[str, Any]) -> str:
    if record["retrieval_source_id"] in DIRECT_RETRIEVAL_SOURCES:
        return str(record.get("condition_group") or "")
    raw = record.get("canonical_pair_fields_json")
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        payload = {"pair_bucket_key": record.get("pair_bucket_key")}
    if isinstance(payload, dict):
        payload.pop("source_id", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _measurement_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    kind = str(record.get("measurement_kind") or "")
    scale = _normalized_text(record.get("canonical_measurement_scale_id"))
    unit = _normalized_text(record.get("canonical_unit_text"))
    endpoint = _normalized_text(
        record.get("direct_residual_endpoint_name")
        or record.get("canonical_endpoint_name")
        or record.get("endpoint_name")
    )
    if record.get("direct_vote_label") in {0, 1}:
        value: Any = ("direct_label", int(record["direct_vote_label"]))
    elif record.get("canonical_category_id") is not None:
        value = (
            "category",
            str(record.get("canonical_category_id")),
            _json_number(record.get("canonical_category_rank")),
        )
    elif _finite(record.get("finite_scalar_value")):
        value = ("scalar", float(record["finite_scalar_value"]))
    else:
        value = (
            "text",
            _normalized_text(
                record.get("canonical_measurement_text") or record.get("measurement_text")
            ),
        )
    return kind, endpoint, unit, scale, value


def _dedup_match(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[float, str] | None:
    left_claim = _text(left.get("canonical_claim_id"))
    right_claim = _text(right.get("canonical_claim_id"))
    if left_claim and left_claim == right_claim:
        left_parent = _text(left.get("canonical_claim_parent_identity_key"))
        right_parent = _text(right.get("canonical_claim_parent_identity_key"))
        compatible = (
            left_parent
            and left_parent == right_parent
            and left.get("retrieval_source_id") == right.get("retrieval_source_id")
            and left.get("direct_vote_label") == right.get("direct_vote_label")
            and left.get("condition_group") == right.get("condition_group")
        )
        if not compatible:
            raise ValueError(
                f"canonical direct claim members disagree: {left_claim}"
            )
        return 1.0, "same_canonical_direct_claim"
    if not _text(left.get("canonical_smiles")) or not _text(right.get("canonical_smiles")):
        return None
    if _context_conflict(left, right):
        return None
    left_support = _normalized_text(left.get("support_text"))
    right_support = _normalized_text(right.get("support_text"))
    if str(left.get("source_id") or "") == str(right.get("source_id") or ""):
        if _within_source_identity(left) == _within_source_identity(right):
            return 1.0, "within_source_exact_semantic_row"
        return None
    left_paper, right_paper = _paper_id(left), _paper_id(right)
    if left_paper or right_paper:
        if not left_paper or left_paper != right_paper:
            return None
        similarity = _jaccard(left_support, right_support)
        return (
            (similarity, "same_paper_equal_measurement_similar_support")
            if similarity >= SUPPORT_JACCARD_MINIMUM
            else None
        )
    if left_support and left_support == right_support:
        return 1.0, "equal_measurement_identical_support_without_paper_id"
    return None


def _within_source_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    context = _text(record.get("deduplication_context_id")) or _json_context(
        record.get("evidence_context_json")
    )
    return (
        _text(record.get("source_smiles")),
        _text(record.get("group_id")),
        _measurement_signature(record),
        _text(record.get("assay_transfer_prebase_measurement_text")),
        _text(record.get("assay_transfer_prebase_unit_text")),
        context,
        _text(record.get("support_text")),
        _paper_id(record),
    )


def _json_context(value: Any) -> str:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return _text(value)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _context_conflict(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return any(
        (left_value := _normalized_text(left.get(field)))
        and (right_value := _normalized_text(right.get(field)))
        and left_value != right_value
        for field in _CONTEXT_FIELDS
    )


def _representative_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    coverage = sum(bool(_text(record.get(field))) for field in _CONTEXT_FIELDS)
    support_length = len(_normalized_text(record.get("support_text")))
    return (
        not bool(record.get("canonical_claim_representative")),
        -coverage,
        -support_length,
        str(record["canonical_record_id"]),
    )


def _paper_id(record: Mapping[str, Any]) -> str:
    pmid = _normalized_text(record.get("pmid"))
    if pmid:
        return f"pmid:{pmid}"
    doi = _normalized_text(record.get("doi"))
    return f"doi:{doi}" if doi else ""


def _jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(_TOKEN.findall(left)), set(_TOKEN.findall(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _normalized_text(value: Any) -> str:
    return " ".join(_TOKEN.findall(_text(value).casefold()))


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_number(value: Any) -> int | float | None:
    if not _finite(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "DEDUPLICATION_VERSION",
    "DIRECT_MAPPING_FILENAME",
    "DUPLICATES_FILENAME",
    "MANIFEST_FILENAME",
    "PAIR_BUCKET_RECORDS_FILENAME",
    "RECORDS_FILENAME",
    "build_deduplicated_record_stage",
    "load_duplicate_lineage",
]
