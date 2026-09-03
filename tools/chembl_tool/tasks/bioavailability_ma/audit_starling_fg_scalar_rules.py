"""Audit strict Fg scalar proposals without rebuilding normalized evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_fg_scalar_rules import (
    FG_SCALAR_RULE_VERSION,
    propose_fg_scalar,
)


DEFAULT_INPUT = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v6/02_normalized/records.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v6/09_audits/fg_scalar_rules_v2"
)
AUDIT_VERSION = "bioavailability_fg_scalar_rule_audit.v2"
MANUAL_SAMPLE_SEED = 20260731

_AUDIT_COLUMNS = (
    "source_id",
    "source_row_number",
    "normalized_record_id",
    "cleaned_record_id",
    "molecule_id",
    "molecule_name",
    "canonical_smiles",
    "endpoint_name",
    "canonical_endpoint",
    "measurement_text",
    "unit_text",
    "canonical_measurement",
    "canonical_unit",
    "finite_scalar_value",
    "is_absolute_and_continuous",
    "normalization_validity_status",
    "measurement_unit_status",
    "unit_notation_status",
    "unit_notation_factor",
    "source_scalar_rule_id",
    "support_text",
    "assay_system",
    "qualifying_conditions",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row.get(field) or "__none__") for row in rows).items()
        )
    )


def _decision_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        decision = propose_fg_scalar(
            raw.get("measurement_text"),
            canonical_endpoint=str(raw.get("canonical_endpoint") or ""),
        )
        row = {key: _clean_json_value(raw.get(key)) for key in _AUDIT_COLUMNS}
        row.update({f"proposal_{key}": value for key, value in decision.to_dict().items()})
        rows.append(row)
    return rows


def _validate(rows: list[dict[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"decision coverage mismatch: {len(rows)} != {expected_rows}")
    ids = [row["normalized_record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("normalized_record_id is not unique in the Fg audit")
    for row in rows:
        accepted = bool(row["proposal_accepted"])
        required = (
            "proposal_rule_id",
            "proposal_semantic_label",
            "proposal_canonical_measurement",
            "proposal_canonical_unit",
            "proposal_finite_scalar_value",
        )
        if accepted and any(row.get(field) is None for field in required):
            raise ValueError(
                f"incomplete accepted proposal: {row['normalized_record_id']}"
            )
        if not accepted and row.get("proposal_finite_scalar_value") is not None:
            raise ValueError(
                f"rejected proposal contains scalar: {row['normalized_record_id']}"
            )
        if row.get("source_scalar_rule_id") != row.get("proposal_rule_id"):
            raise ValueError(f"persisted/proposed rule mismatch: {row['normalized_record_id']}")
        if accepted:
            persisted_factor = row.get("unit_notation_factor")
            proposed_factor = row.get("proposal_unit_notation_factor")
            if (
                row.get("canonical_measurement") != row.get("proposal_canonical_measurement")
                or row.get("canonical_unit") != row.get("proposal_canonical_unit")
                or row.get("unit_notation_status")
                != row.get("proposal_unit_notation_status")
                or persisted_factor != proposed_factor
            ):
                raise ValueError(
                    f"persisted/proposed canonical pair drift: {row['normalized_record_id']}"
                )


def _manual_sample(rows: list[dict[str, Any]], count: int = 100) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    frame = pd.DataFrame(rows)
    return (
        frame.sample(n=count, random_state=MANUAL_SAMPLE_SEED)
        .sort_values("source_row_number")
        .to_dict(orient="records")
    )


def _boundary_sample(
    rows: list[dict[str, Any]], *, per_bucket: int = 20
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    selected: list[pd.DataFrame] = []
    accepted = frame[frame["proposal_accepted"]]
    for _, group in accepted.groupby("proposal_rule_id", sort=True):
        selected.append(group.sort_values("source_row_number").head(per_bucket))
    rejected = frame[~frame["proposal_accepted"]]
    for _, group in rejected.groupby("proposal_reason", sort=True):
        selected.append(group.sort_values("source_row_number").head(per_bucket))
    if not selected:
        return []
    return (
        pd.concat(selected, ignore_index=True)
        .drop_duplicates("normalized_record_id")
        .sort_values(["proposal_accepted", "proposal_reason", "source_row_number"])
        .to_dict(orient="records")
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            )


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# FG single-outcome scalar-rule audit",
        "",
        f"- Audit version: `{summary['audit_version']}`",
        f"- Rule version: `{summary['rule_version']}`",
        f"- Input rows: {summary['n_fg_rows']:,}",
        f"- Accepted proposals: {summary['n_accepted']:,}",
        (
            "- Accepted proposals with a resolved molecule: "
            f"{summary['n_accepted_resolved_structure']:,}"
        ),
        (
            "- Accepted proposals with an unresolved structure: "
            f"{summary['n_accepted_unresolved_structure']:,}"
        ),
        f"- Rejected proposals: {summary['n_rejected']:,}",
        f"- Existing FG finite scalars: {summary['n_existing_fg_finite_scalars']:,}",
        "",
        (
            "This command is a read-only audit of the current normalized records; "
            "it does not modify normalized or downstream artifacts."
        ),
        "",
        "## Accepted by rule",
        "",
        "| Rule | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{key}` | {value:,} |"
        for key, value in summary["accepted_rule_counts"].items()
        if key != "__none__"
    )
    lines.extend(
        [
            "",
            "## Proposed canonical units",
            "",
            "| Unit | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{key}` | {value:,} |"
        for key, value in summary["accepted_unit_counts"].items()
        if key != "__none__"
    )
    lines.extend(
        [
            "",
            "## Rejection reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    lines.extend(
        f"| `{key}` | {value:,} |"
        for key, value in summary["rejection_reason_counts"].items()
        if key != "__none__"
    )
    lines.extend(
        [
            "",
            "## Audit files",
            "",
            "- `decisions.parquet`: one proposal or rejection for every FG row.",
            "- `manual_review_sample_100.jsonl`: deterministic uniform 100-row sample.",
            "- `accepted_review_sample_100.jsonl`: deterministic 100-row accepted sample.",
            "- `boundary_examples.jsonl`: examples stratified by rule and rejection reason.",
            "- `summary.json`: machine-readable counts and provenance.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    input_path: Path,
    output_dir: Path,
    *,
    max_rows: int = 0,
) -> dict[str, Any]:
    frame = pd.read_parquet(input_path, columns=list(_AUDIT_COLUMNS))
    frame = frame[frame["source_id"] == "fg"].copy()
    if max_rows:
        frame = frame.head(max_rows).copy()
    rows = _decision_rows(frame)
    _validate(rows, len(frame))

    accepted = [row for row in rows if row["proposal_accepted"]]
    rejected = [row for row in rows if not row["proposal_accepted"]]
    summary = {
        "audit_version": AUDIT_VERSION,
        "rule_version": FG_SCALAR_RULE_VERSION,
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "n_fg_rows": len(rows),
        "n_accepted": len(accepted),
        "n_accepted_resolved_structure": sum(
            bool(row.get("canonical_smiles")) for row in accepted
        ),
        "n_accepted_unresolved_structure": sum(
            not bool(row.get("canonical_smiles")) for row in accepted
        ),
        "n_rejected": len(rejected),
        "acceptance_fraction": len(accepted) / len(rows) if rows else 0.0,
        "n_existing_fg_finite_scalars": sum(
            row.get("finite_scalar_value") is not None for row in rows
        ),
        "accepted_rule_counts": _counts(accepted, "proposal_rule_id"),
        "accepted_unit_counts": _counts(accepted, "proposal_canonical_unit"),
        "accepted_semantic_label_counts": _counts(
            accepted, "proposal_semantic_label"
        ),
        "accepted_endpoint_counts": _counts(accepted, "canonical_endpoint"),
        "rejection_reason_counts": _counts(rejected, "proposal_reason"),
        "manual_sample_seed": MANUAL_SAMPLE_SEED,
        "invariants": {
            "one_decision_per_fg_row": True,
            "unique_normalized_record_ids": True,
            "accepted_rows_have_complete_canonical_pair": True,
            "rejected_rows_have_no_scalar": True,
            "persisted_rule_id_matches_proposal": True,
            "scientific_notation_provenance_matches_proposal": True,
            "source_records_modified": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_dir / "decisions.parquet", index=False)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(
        output_dir / "manual_review_sample_100.jsonl", _manual_sample(rows)
    )
    _write_jsonl(
        output_dir / "accepted_review_sample_100.jsonl", _manual_sample(accepted)
    )
    _write_jsonl(
        output_dir / "boundary_examples.jsonl", _boundary_sample(rows)
    )
    (output_dir / "AUDIT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    summary = run_audit(args.input, args.output_dir, max_rows=args.max_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
