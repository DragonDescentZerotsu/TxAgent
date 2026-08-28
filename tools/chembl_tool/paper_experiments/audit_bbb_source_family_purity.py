"""Audit every BBB retrieval source row against the frozen family contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    atomic_output_path,
    sha256_file,
    write_json_atomic,
)
from tools.chembl_tool.tasks.bbb_martins.source_family_purity import (
    BBBSourceFamilyClassifier,
    CLASSIFIER_COLUMNS,
    DEFAULT_REVIEW_LEDGER,
    DIRECT_GROUP,
    EFFLUX_GROUP,
    INFLUX_GROUP,
    NEAR_DIRECT_GROUP,
    PASSIVE_GROUP,
    _explicit_efflux,
    _explicit_influx,
    _explicit_passive,
    _searchable,
    is_explicit_prediction_record,
    is_mdck_record,
    is_pampa_record,
    load_gold_vote_source_indices,
    load_near_direct_reviews,
)


DEFAULT_RECORDS = Path(
    "outputs/paper/starling_conditioned_assay_family_curve_v1/"
    "source_overlays/bbb_source_family_purity_v4/records.parquet"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RECORDS.parent / "purity_audit"
AUDIT_VERSION = "bbb_source_family_row_audit.v1"
_LEDGER_SCHEMA = pa.schema(
    [
        pa.field("source_row_index", pa.int64()),
        pa.field("source_index", pa.int64()),
        pa.field("source_record_id", pa.string()),
        pa.field("retrieval_eligible", pa.bool_()),
        pa.field("original_group_id", pa.string()),
        pa.field("assigned_group_id", pa.string()),
        pa.field("assignment_reason", pa.string()),
        pa.field("canonical_endpoint_name", pa.string()),
        pa.field("bbb_transport_label", pa.string()),
        pa.field("canonical_transport_mechanism", pa.string()),
    ]
)


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def audit(records_path: Path, review_path: Path, output_dir: Path) -> dict[str, Any]:
    gold_indices = load_gold_vote_source_indices()
    classifier = BBBSourceFamilyClassifier(
        load_near_direct_reviews(review_path), gold_indices
    )
    parquet = pq.ParquetFile(records_path)
    required = tuple(
        dict.fromkeys(
            (
                *CLASSIFIER_COLUMNS,
                "source_family_original_group_id",
                "source_family_purity_reason",
            )
        )
    )
    missing = sorted(set(required) - set(parquet.schema.names))
    if missing:
        raise ValueError(f"records missing audit columns: {missing}")

    counts: Counter[str] = Counter()
    eligible_counts: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    violations: Counter[str] = Counter()
    row_offset = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "record_family_assignments.parquet"
    with atomic_output_path(ledger_path) as temporary_path, pq.ParquetWriter(
        temporary_path, _LEDGER_SCHEMA, compression="zstd"
    ) as writer:
        for batch in parquet.iter_batches(batch_size=5_000, columns=list(required)):
            ledger_rows: list[dict[str, Any]] = []
            for local_index, row in enumerate(batch.to_pylist()):
                assigned = _text(row.get("group_id"))
                recorded_reason = _text(row.get("source_family_purity_reason"))
                assignment_reason = (
                    recorded_reason
                    or "retained_after_full_record_family_classification"
                )
                if assigned == DIRECT_GROUP:
                    decision = classifier.classify_target(row)
                    if assigned != decision.new_group:
                        violations["direct_classifier_assignment_mismatch"] += 1
                    if is_explicit_prediction_record(row):
                        violations["direct_prediction"] += 1
                    if is_pampa_record(row):
                        violations["direct_pampa"] += 1
                    if is_mdck_record(row):
                        violations["direct_mdck"] += 1
                endpoint = _text(row.get("canonical_endpoint_name")).lower()
                if assigned == INFLUX_GROUP and endpoint.startswith("efflux"):
                    violations["influx_with_efflux_endpoint"] += 1
                if assigned == PASSIVE_GROUP and endpoint.startswith("efflux"):
                    violations["passive_with_efflux_endpoint"] += 1
                searchable = _searchable(row)
                if assigned == INFLUX_GROUP:
                    if not _explicit_influx(row, searchable):
                        violations["influx_without_directional_influx_signal"] += 1
                    if _explicit_efflux(row, searchable):
                        violations["influx_with_higher_priority_efflux_signal"] += 1
                if assigned == EFFLUX_GROUP and not _explicit_efflux(row, searchable):
                    violations["efflux_without_directional_efflux_signal"] += 1
                if assigned == PASSIVE_GROUP and not (
                    is_pampa_record(row)
                    or is_mdck_record(row)
                    or _explicit_passive(row, searchable)
                    or assignment_reason.startswith(("boiled_egg_", "qikprop_"))
                ):
                    violations["passive_without_passive_assay_or_readout"] += 1

                counts[assigned] += 1
                if row.get("retrieval_eligible") is True:
                    eligible_counts[assigned] += 1
                reasons[assigned][assignment_reason] += 1
                ledger_rows.append(
                    {
                        "source_row_index": row_offset + local_index,
                        "source_index": _optional_int(row.get("source_index")),
                        "source_record_id": _text(
                            row.get("canonical_record_id")
                            or row.get("source_record_id")
                            or row.get("extraction_id")
                        ),
                        "retrieval_eligible": row.get("retrieval_eligible") is True,
                        "original_group_id": _text(
                            row.get("source_family_original_group_id")
                            or row.get("group_id")
                        ),
                        "assigned_group_id": assigned,
                        "assignment_reason": assignment_reason,
                        "canonical_endpoint_name": _text(
                            row.get("canonical_endpoint_name")
                        ),
                        "bbb_transport_label": _text(
                            row.get("bbb_transport_label")
                        ),
                        "canonical_transport_mechanism": _text(
                            row.get("canonical_transport_mechanism")
                        ),
                    }
                )
            writer.write_table(pa.Table.from_pylist(ledger_rows, schema=_LEDGER_SCHEMA))
            row_offset += len(ledger_rows)

    manifest = {
        "audit_version": AUDIT_VERSION,
        "records": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "review_ledger": str(review_path.resolve()),
        "review_ledger_sha256": sha256_file(review_path),
        "record_assignment_ledger": str(ledger_path.resolve()),
        "record_assignment_ledger_sha256": sha256_file(ledger_path),
        "n_rows_audited": row_offset,
        "n_gold_vote_source_indices_used_for_l1_provenance": len(gold_indices),
        "family_counts": dict(sorted(counts.items())),
        "retrieval_eligible_family_counts": dict(sorted(eligible_counts.items())),
        "assignment_reason_counts": {
            group: dict(sorted(group_reasons.items()))
            for group, group_reasons in sorted(reasons.items())
        },
        "violation_counts": dict(sorted(violations.items())),
        "n_violations": sum(violations.values()),
        "all_rows_pass_family_safety_checks": not violations,
        "family_contract": {
            DIRECT_GROUP: "accepted experimental CNS-access gold source only",
            NEAR_DIRECT_GROUP: "prediction, generic/missing, or functional BBB proxy",
            PASSIVE_GROUP: "PAMPA, passive/cell permeability, or passive prediction",
            EFFLUX_GROUP: "explicit efflux endpoint, label, or efflux-transporter evidence",
            INFLUX_GROUP: "explicit influx label/mechanism or directional uptake evidence",
        },
    }
    write_json_atomic(output_dir / "summary.json", manifest)
    if row_offset != parquet.metadata.num_rows:
        raise RuntimeError(
            f"audited {row_offset} rows, expected {parquet.metadata.num_rows}"
        )
    if violations:
        raise RuntimeError(f"source-family purity violations: {dict(violations)}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--review-ledger", default=str(DEFAULT_REVIEW_LEDGER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    print(audit(Path(args.records), Path(args.review_ledger), Path(args.output_dir)))


if __name__ == "__main__":
    main()
