"""Shared builder for source-level family-purity overlays.

An overlay changes only ``group_id`` and appends explicit audit provenance. It
does not alter measurements, support text, molecule identity, retrieval
eligibility, or frozen benchmark labels. Task-specific classifiers decide the
new family; this module only performs the auditable rewrite.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    sha256_file,
    write_json_atomic,
)
from tools.chembl_tool.common.source_family_purity import (
    FamilyMove,
    audit_exact_voter_membership,
)
from tools.chembl_tool.common.starling.current_retrieval_artifacts import (
    current_records_path,
)
from tools.chembl_tool.tasks.skin_reaction.source_family_purity import (
    AOP_GROUP as SKIN_AOP_GROUP,
    DEFAULT_CONDITION_REVIEW as SKIN_CONDITION_REVIEW,
    DIRECT_GROUP as SKIN_DIRECT_GROUP,
    NEAR_DIRECT_GROUP as SKIN_NEAR_DIRECT_GROUP,
    PURITY_VERSION as SKIN_PURITY_VERSION,
    SourceRecordKey,
    canonical_partition as skin_canonical_partition,
    load_canonical_partition_decisions as load_skin_canonical_partition_decisions,
    load_voter_source_keys as load_skin_voter_source_keys,
    upstream_source_key as skin_upstream_source_key,
    vote_pure_family_move as skin_vote_pure_family_move,
)
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    strict_scope_exclusion_reason as skin_strict_scope_exclusion_reason,
)


PURITY_VERSION = "conditioned_source_family_purity.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/"
    "source_family_purity_v5"
)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class PuritySpec:
    task: str
    input_records: Path
    direct_group: str
    classify: Callable[[Mapping[str, Any]], str | "FamilyMove"]
    purity_version: str = PURITY_VERSION
    allow_move_from_direct: bool = False
    classifier_columns: tuple[str, ...] = ()


def _skin_spec(
    condition_review: Path = SKIN_CONDITION_REVIEW,
) -> tuple[PuritySpec, set[SourceRecordKey], Mapping[SourceRecordKey, Any]]:
    voters = load_skin_voter_source_keys(condition_review=condition_review)
    canonical_partitions = load_skin_canonical_partition_decisions()
    return (
        PuritySpec(
            task="skin_reaction",
            input_records=current_records_path("skin_reaction"),
            direct_group=SKIN_DIRECT_GROUP,
            classify=partial(
                skin_vote_pure_family_move,
                voter_source_keys=voters,
                canonical_partitions=canonical_partitions,
            ),
            purity_version=SKIN_PURITY_VERSION,
            allow_move_from_direct=True,
        ),
        voters,
        canonical_partitions,
    )


SPECS = {"skin_reaction": _skin_spec}

_PROVENANCE_FIELDS = (
    "source_family_purity_version",
    "source_family_original_group_id",
    "source_family_purity_reason",
)
_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("task", pa.string()),
        pa.field("source_row_index", pa.int64()),
        pa.field("source_id", pa.string()),
        pa.field("source_index", pa.int64()),
        pa.field("source_record_id", pa.string()),
        pa.field("original_group_id", pa.string()),
        pa.field("new_group_id", pa.string()),
        pa.field("purity_reason", pa.string()),
        pa.field("retrieval_eligible", pa.bool_()),
        pa.field("parent_smiles", pa.string()),
        pa.field("canonical_smiles", pa.string()),
        pa.field("canonical_endpoint_name", pa.string()),
        pa.field("canonical_measurement_text", pa.string()),
        pa.field("support_text", pa.string()),
    ]
)


def classify_family_move(
    spec: PuritySpec, record: Mapping[str, Any]
) -> tuple[str, str]:
    """Return ``(new_group, reason)``; an empty reason means no mutation."""

    original = _text(record.get("group_id"))
    if not original or (
        original == spec.direct_group and not spec.allow_move_from_direct
    ):
        return original, ""
    decision = spec.classify(record)
    if isinstance(decision, FamilyMove):
        return (
            (decision.new_group, decision.reason)
            if decision.new_group and decision.reason
            else (original, "")
        )
    return (spec.direct_group, decision) if decision else (original, "")


def build_overlay(
    spec: PuritySpec,
    output_dir: Path,
    *,
    batch_size: int = 10_000,
) -> dict[str, Any]:
    source = spec.input_records.resolve()
    parquet = pq.ParquetFile(source)
    if "group_id" not in parquet.schema.names:
        raise ValueError(f"{source} lacks required group_id")
    collisions = sorted(set(parquet.schema.names) & set(_PROVENANCE_FIELDS))
    if collisions:
        raise ValueError(
            f"Input already contains purity provenance fields: {collisions}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records.parquet"
    audit_path = output_dir / "moved_records.parquet"
    output_schema = (
        pa.schema(parquet.schema_arrow)
        .append(pa.field(_PROVENANCE_FIELDS[0], pa.string()))
        .append(pa.field(_PROVENANCE_FIELDS[1], pa.string()))
        .append(pa.field(_PROVENANCE_FIELDS[2], pa.string()))
    )

    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    original_group_counts: Counter[str] = Counter()
    moved = moved_eligible = row_offset = 0

    with (
        atomic_output_path(output_path) as output_tmp,
        atomic_output_path(audit_path) as audit_tmp,
    ):
        with (
            pq.ParquetWriter(output_tmp, output_schema, compression="zstd") as writer,
            pq.ParquetWriter(
                audit_tmp, _AUDIT_SCHEMA, compression="zstd"
            ) as audit_writer,
        ):
            for batch in parquet.iter_batches(batch_size=batch_size):
                if spec.classifier_columns:
                    missing = set(spec.classifier_columns) - set(batch.schema.names)
                    if missing:
                        raise ValueError(
                            f"classifier columns missing from source: {sorted(missing)}"
                        )
                    rows = batch.select(spec.classifier_columns).to_pylist()
                else:
                    rows = batch.to_pylist()
                new_groups: list[str | None] = []
                versions: list[str | None] = []
                originals: list[str | None] = []
                reasons: list[str | None] = []
                audit_rows: list[dict[str, Any]] = []
                for local_index, record in enumerate(rows):
                    original = _text(record.get("group_id"))
                    new_group, reason = classify_family_move(spec, record)
                    new_groups.append(new_group or None)
                    if not reason:
                        versions.append(None)
                        originals.append(None)
                        reasons.append(None)
                        continue
                    versions.append(spec.purity_version)
                    originals.append(original)
                    reasons.append(reason)
                    moved += 1
                    eligible = record.get("retrieval_eligible") is True
                    moved_eligible += int(eligible)
                    reason_counts[reason] += 1
                    source_counts[_text(record.get("source_id")) or "<missing>"] += 1
                    original_group_counts[original] += 1
                    audit_rows.append(
                        {
                            "task": spec.task,
                            "source_row_index": row_offset + local_index,
                            "source_id": _text(record.get("source_id")),
                            "source_index": _optional_int(record.get("source_index")),
                            "source_record_id": _first_text(
                                record,
                                "canonical_record_id",
                                "record_id",
                                "extraction_id",
                            ),
                            "original_group_id": original,
                            "new_group_id": new_group,
                            "purity_reason": reason,
                            "retrieval_eligible": eligible,
                            "parent_smiles": _text(record.get("parent_smiles")),
                            "canonical_smiles": _first_text(
                                record,
                                "canonical_smiles",
                                "representative_smiles",
                                "smiles",
                            ),
                            "canonical_endpoint_name": _text(
                                record.get("canonical_endpoint_name")
                            ),
                            "canonical_measurement_text": _text(
                                record.get("canonical_measurement_text")
                            ),
                            "support_text": _text(record.get("support_text")),
                        }
                    )

                group_index = batch.schema.get_field_index("group_id")
                group_field = batch.schema.field(group_index)
                rewritten = batch.set_column(
                    group_index,
                    group_field,
                    pa.array(new_groups, type=group_field.type, from_pandas=True),
                )
                rewritten = (
                    rewritten.append_column(
                        _PROVENANCE_FIELDS[0], pa.array(versions, type=pa.string())
                    )
                    .append_column(
                        _PROVENANCE_FIELDS[1], pa.array(originals, type=pa.string())
                    )
                    .append_column(
                        _PROVENANCE_FIELDS[2], pa.array(reasons, type=pa.string())
                    )
                )
                writer.write_batch(rewritten)
                if audit_rows:
                    audit_writer.write_table(
                        pa.Table.from_pylist(audit_rows, schema=_AUDIT_SCHEMA)
                    )
                row_offset += len(rows)

    manifest = {
        "purity_version": spec.purity_version,
        "task": spec.task,
        "direct_group": spec.direct_group,
        "input_records": _project_path(source),
        "input_records_sha256": sha256_file(source),
        "output_records": str(output_path.resolve()),
        "output_records_sha256": sha256_file(output_path),
        "moved_records": str(audit_path.resolve()),
        "moved_records_sha256": sha256_file(audit_path),
        "n_source_rows": parquet.metadata.num_rows,
        "n_output_rows": row_offset,
        "n_moved_rows": moved,
        "n_moved_retrieval_eligible_rows": moved_eligible,
        "moved_by_reason": dict(sorted(reason_counts.items())),
        "moved_by_source_id": dict(sorted(source_counts.items())),
        "moved_by_original_group": dict(sorted(original_group_counts.items())),
        "mutated_fields": ["group_id"],
        "appended_audit_fields": list(_PROVENANCE_FIELDS),
        "gold_labels_modified": False,
    }
    if row_offset != parquet.metadata.num_rows:
        raise RuntimeError(
            f"Row-count mismatch: read {row_offset}, expected {parquet.metadata.num_rows}"
        )
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def finalize_skin_manifest(
    output_dir: Path,
    voter_source_keys: set[SourceRecordKey],
    canonical_partitions: Mapping[SourceRecordKey, Any],
    condition_review: Path,
) -> dict[str, Any]:
    """Add exact voter and semantic-family publication gates for Skin."""

    output_path = output_dir / "records.parquet"
    parquet = pq.ParquetFile(output_path)
    rows_for_membership = (
        row
        for batch in parquet.iter_batches(
            columns=["group_id", "source_id", "source_row_number", "retrieval_eligible"]
        )
        for row in batch.to_pylist()
    )
    membership = audit_exact_voter_membership(
        rows_for_membership,
        voter_source_keys,
        direct_group=SKIN_DIRECT_GROUP,
        record_id=skin_upstream_source_key,
    )

    family_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    mismatch_count = source_rows_audited = 0
    strict_scope_violations: Counter[str] = Counter()
    strict_scope_examples: list[dict[str, Any]] = []
    mismatch_examples: list[dict[str, Any]] = []
    semantic_rows = (
        row for batch in parquet.iter_batches() for row in batch.to_pylist()
    )
    for row in semantic_rows:
        key = skin_upstream_source_key(row)
        if key is None:
            continue
        source_rows_audited += 1
        decision = skin_canonical_partition(row, canonical_partitions)
        if decision is None:
            raise RuntimeError(
                f"Skin semantic audit lacks canonical decision for {key!r}"
            )
        actual = _text(row.get("group_id"))
        strict_reason = skin_strict_scope_exclusion_reason(row)
        if strict_reason and actual in {
            SKIN_DIRECT_GROUP,
            SKIN_NEAR_DIRECT_GROUP,
            SKIN_AOP_GROUP,
        }:
            strict_scope_violations[strict_reason] += 1
            if len(strict_scope_examples) < 20:
                strict_scope_examples.append(
                    {
                        "source_record_key": f"{key[0]}:{key[1]}",
                        "group_id": actual,
                        "strict_scope_exclusion_reason": strict_reason,
                        "canonical_assay_context": _text(
                            row.get("canonical_assay_context")
                        ),
                        "canonical_endpoint_name": _text(
                            row.get("canonical_endpoint_name")
                        ),
                    }
                )
        replay = skin_vote_pure_family_move(
            row, voter_source_keys, canonical_partitions
        )
        expected = replay.new_group or actual
        family_counts[actual] += 1
        partition_counts[decision.partition] += 1
        if actual != expected:
            mismatch_count += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {
                        "source_record_key": f"{key[0]}:{key[1]}",
                        "partition": decision.partition,
                        "partition_reason": decision.reason,
                        "expected_group": expected,
                        "actual_group": actual,
                    }
                )
    semantic_gate = {
        "skin_level_semantic_purity": (
            mismatch_count == 0 and not strict_scope_violations
        ),
        "n_source_rows_audited": source_rows_audited,
        "family_counts": dict(sorted(family_counts.items())),
        "canonical_partition_counts": dict(sorted(partition_counts.items())),
        "n_semantic_family_mismatches": mismatch_count,
        "semantic_family_mismatch_examples": mismatch_examples,
        "n_strict_target_scope_violations": sum(strict_scope_violations.values()),
        "strict_target_scope_violation_counts": dict(
            sorted(strict_scope_violations.items())
        ),
        "strict_target_scope_violation_examples": strict_scope_examples,
    }
    if mismatch_count or strict_scope_violations:
        raise RuntimeError(f"Skin semantic-family gate failed: {semantic_gate}")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "l1_contract": (
                "exact source-record membership in the current voter ledger; "
                "retrieval L1 additionally intersects retrieval eligibility and "
                "excludes held-out parents"
            ),
            "level_contract": {
                "L1": "actual base or accepted condition-review voter records only",
                "L2": (
                    "canonical observed final sensitization outcomes that did not vote, "
                    "plus predicted or defined-approach overall sensitization classifications"
                ),
                "L3": (
                    "experimental or predicted sensitization mechanisms, including explicit "
                    "AOP key events and sensitization-anchored unspecified mechanisms"
                ),
                "excluded": (
                    "photo/light-dependent evidence, irritation-only outcomes, non-contact "
                    "cutaneous adverse reactions, empty/no-evidence, unrelated, and truly "
                    "unresolved records are absent from all three levels"
                ),
            },
            "voter_counts": {
                "n_voter_source_keys": len(voter_source_keys),
                "condition_review": str(condition_review.resolve()),
                "condition_review_sha256": sha256_file(condition_review),
            },
            "hard_gates": {**membership, **semantic_gate},
        }
    )
    write_json_atomic(manifest_path, manifest)
    return manifest


def _first_text(record: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = _text(record.get(field))
        if value:
            return value
    return ""


def _optional_int(value: Any) -> int | None:
    if value is None or not _text(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return " ".join(str(value).split())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *SPECS], default="all")
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--input-records",
        type=Path,
        help="Override the canonical records path when exactly one task is selected.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20_000,
        help="Parquet batch size; 20,000 reproduces the current Skin overlay bytes.",
    )
    parser.add_argument(
        "--skin-condition-review", type=Path, default=SKIN_CONDITION_REVIEW
    )
    args = parser.parse_args(argv)
    if args.input_records and args.task == "all":
        parser.error("--input-records requires one explicit --task")
    tasks = list(SPECS) if args.task == "all" else [args.task]
    for task in tasks:
        spec, voters, canonical_partitions = SPECS[task](args.skin_condition_review)
        if args.input_records:
            spec = replace(spec, input_records=args.input_records)
        output_dir = Path(args.output_root) / task
        manifest = build_overlay(
            spec,
            output_dir,
            batch_size=args.batch_size,
        )
        manifest = finalize_skin_manifest(
            output_dir, voters, canonical_partitions, args.skin_condition_review
        )
        print(
            f"{task}: moved {manifest['n_moved_rows']:,} rows "
            f"({manifest['n_moved_retrieval_eligible_rows']:,} retrieval eligible)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
