"""Shared builder for source-level family-purity overlays.

An overlay changes only ``group_id`` and appends explicit audit provenance. It
does not alter measurements, support text, molecule identity, retrieval
eligibility, or frozen benchmark labels. Task-specific classifiers decide the
new family; this module only performs the auditable rewrite.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    sha256_file,
    write_json_atomic,
)
from tools.chembl_tool.common.source_family_purity import FamilyMove
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.canonical_starling_source import (
    direct_outcome_reason,
)


PURITY_VERSION = "conditioned_source_family_purity.v1"
SKIN_AOP_GROUP = "Mechanism.sensitization_aop"
SKIN_DIRECT_GROUP = "Direct.skin_reaction"
DEFAULT_ROOT = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/source_overlays/"
    "source_family_purity_v1"
)


@dataclass(frozen=True)
class PuritySpec:
    task: str
    input_records: Path
    direct_group: str
    classify: Callable[[Mapping[str, Any]], str | "FamilyMove"]
    purity_version: str = PURITY_VERSION
    allow_move_from_direct: bool = False
    classifier_columns: tuple[str, ...] = ()


def _skin_reason(record: Mapping[str, Any]) -> str:
    if _text(record.get("group_id")) != SKIN_AOP_GROUP:
        return ""
    return direct_outcome_reason(record)


SPECS = {
    "skin_reaction": PuritySpec(
        task="skin_reaction",
        input_records=Path(
            "/data1/joseph/TxAgent/outputs/chembl_tool/tasks/skin_reaction/"
            "evidence_library/starling_normalized_v7/03_records/records.parquet"
        ),
        direct_group=SKIN_DIRECT_GROUP,
        classify=_skin_reason,
    ),
}

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


def classify_family_move(spec: PuritySpec, record: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(new_group, reason)``; an empty reason means no mutation."""

    original = _text(record.get("group_id"))
    if not original or (original == spec.direct_group and not spec.allow_move_from_direct):
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
        raise ValueError(f"Input already contains purity provenance fields: {collisions}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "records.parquet"
    audit_path = output_dir / "moved_records.parquet"
    output_schema = pa.schema(parquet.schema_arrow).append(
        pa.field(_PROVENANCE_FIELDS[0], pa.string())
    ).append(pa.field(_PROVENANCE_FIELDS[1], pa.string())).append(
        pa.field(_PROVENANCE_FIELDS[2], pa.string())
    )

    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    original_group_counts: Counter[str] = Counter()
    moved = moved_eligible = row_offset = 0

    with atomic_output_path(output_path) as output_tmp, atomic_output_path(
        audit_path
    ) as audit_tmp:
        with pq.ParquetWriter(output_tmp, output_schema, compression="zstd") as writer, pq.ParquetWriter(
            audit_tmp, _AUDIT_SCHEMA, compression="zstd"
        ) as audit_writer:
            for batch in parquet.iter_batches(batch_size=batch_size):
                if spec.classifier_columns:
                    missing = set(spec.classifier_columns) - set(batch.schema.names)
                    if missing:
                        raise ValueError(f"classifier columns missing from source: {sorted(missing)}")
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
                                record, "canonical_smiles", "representative_smiles", "smiles"
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
                rewritten = rewritten.append_column(
                    _PROVENANCE_FIELDS[0], pa.array(versions, type=pa.string())
                ).append_column(
                    _PROVENANCE_FIELDS[1], pa.array(originals, type=pa.string())
                ).append_column(
                    _PROVENANCE_FIELDS[2], pa.array(reasons, type=pa.string())
                )
                writer.write_batch(rewritten)
                if audit_rows:
                    audit_writer.write_table(pa.Table.from_pylist(audit_rows, schema=_AUDIT_SCHEMA))
                row_offset += len(rows)

    manifest = {
        "purity_version": spec.purity_version,
        "task": spec.task,
        "direct_group": spec.direct_group,
        "input_records": str(source),
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
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args(argv)
    tasks = list(SPECS) if args.task == "all" else [args.task]
    for task in tasks:
        spec = SPECS[task]
        manifest = build_overlay(
            spec,
            Path(args.output_root) / task,
            batch_size=args.batch_size,
        )
        print(
            f"{task}: moved {manifest['n_moved_rows']:,} rows "
            f"({manifest['n_moved_retrieval_eligible_rows']:,} retrieval eligible)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
