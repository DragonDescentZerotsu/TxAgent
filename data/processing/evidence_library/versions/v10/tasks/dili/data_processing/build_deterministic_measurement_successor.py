"""Build a reviewed DILI measurement-resolution successor from finite source points."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.numeric_syntax import finite_point_text
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution as resolution,
)

ROOT = Path(__file__).resolve().parents[8]
DEFAULT_INPUT = (
    ROOT
    / "data/evidence_libraries/dili/v10/measurement_resolution_v4/measurement_resolution.parquet"
)
DEFAULT_RECORDS = (
    ROOT / "data/evidence_libraries/dili/v10/01_cleaned/records.parquet"
)
DEFAULT_UNITS = (
    Path(__file__).parent
    / "canonicalization_v10/unit_reconciliation_v3/mapping.json"
)
DEFAULT_GOLD = (
    ROOT
    / "tests/chembl_tool/common/measurement_resolution_quality/gold/dili.v10.2.jsonl"
)

ALLOWED_UNIT_GUARDS = {
    None,
    "unqualified_percent_unit",
    "unsupported_or_uncertain_unit",
    "incomplete_named_metric",
    "unit_contains_context_clause",
    "unit_contains_time_context",
    "unit_contains_uncertainty",
    "unit_contains_comparator_context",
    "unit_contains_additional_result",
    "unit_not_tied_to_selected_value",
    "unit_not_exact_declared_unit",
    "unsupported_literal_unit",
}
DECISIONS = {"accept_absolute", "keep_original"}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _raw_proposal(value: Any, central: str) -> str:
    try:
        payload = json.loads(_text(value) or "{}")
        entries = payload.get("measurements", [])
        if payload.get("status") != "ok" or len(entries) != 1:
            return ""
        entry = entries[0]
        point = finite_point_text(entry.get("measurement"))
        if point is None or point[1] is not None or Decimal(point[0]) != Decimal(central):
            return ""
        return _text(entry.get("unit"))
    except (json.JSONDecodeError, ArithmeticError, ValueError, TypeError):
        return ""


def _rule(
    mapping: dict[tuple[str, str, str], dict[str, Any]], endpoint: str, unit: str
) -> dict[str, Any] | None:
    rule = mapping.get(("dili", endpoint, unit)) or mapping.get(("dili", "*", unit))
    return rule if rule and rule.get("action") == "map" else None


def _mapped(rule: dict[str, Any] | None) -> tuple[str, Decimal] | None:
    if rule is None:
        return None
    return _text(rule["canonical_unit"]), Decimal(str(rule["scale"]))


def _proposal(
    record: dict[str, Any],
    resolved: dict[str, Any],
    units: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    if resolved["status"] != "unsure":
        return None
    point = finite_point_text(record.get("measurement_text"))
    if point is None:
        return None
    central, uncertainty = point
    endpoint = _text(record.get("canonical_endpoint_name") or record.get("endpoint_name"))
    source_unit = _text(record.get("unit_text"))
    llm_unit = _raw_proposal(resolved.get("raw_response_json"), central)
    source_rule = _rule(units, endpoint, source_unit) if source_unit else None
    llm_rule = _rule(units, endpoint, llm_unit) if llm_unit else None
    source_mapped, llm_mapped = _mapped(source_rule), _mapped(llm_rule)
    if source_mapped and llm_mapped and source_mapped != llm_mapped:
        return None
    if source_mapped:
        chosen_unit, chosen_rule, origin = source_unit, source_rule, "source"
    elif llm_mapped:
        chosen_unit, chosen_rule, origin = llm_unit, llm_rule, "llm"
    else:
        return None
    guard = resolution._guard_reason(record, central, chosen_unit)
    if guard not in ALLOWED_UNIT_GUARDS:
        return None
    return {
        "cleaned_record_id": _text(record["cleaned_record_id"]),
        "source_row_uid": _text(record["source_row_uid"]),
        "source_id": _text(record["source_id"]),
        "endpoint_name": _text(record.get("endpoint_name")),
        "canonical_endpoint_name": endpoint,
        "measurement_text": _text(record.get("measurement_text")),
        "parsed_measurement": central,
        "parsed_uncertainty": uncertainty,
        "source_unit": source_unit,
        "llm_unit": llm_unit,
        "selected_unit": chosen_unit,
        "selected_unit_origin": origin,
        "canonical_unit": _text(chosen_rule["canonical_unit"]),
        "unit_scale": _text(chosen_rule["scale"]),
        "prior_guard_reason": _text(resolved.get("assignment_guard_reason")) or None,
        "candidate_guard_reason": guard,
        "support_text": _text(record.get("support_text")),
        **{
            field: record.get(field)
            for field in resolution.prompt_row_fields(_text(record["source_id"]))
        },
    }


def prepare(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    unit_map = load_exact_unit_mapping(args.units)
    resolved = {
        row["cleaned_record_id"]: row
        for row in pq.read_table(args.input).to_pylist()
    }
    fields = sorted(
        {
            "cleaned_record_id",
            "source_row_uid",
            "source_id",
            "endpoint_name",
            "canonical_endpoint_name",
            "measurement_text",
            "unit_text",
            "support_text",
            *(
                field
                for source in resolution.SOURCE_IDS
                for field in resolution.prompt_row_fields(source)
            ),
        }
    )
    candidates = []
    for batch in pq.ParquetFile(args.records).iter_batches(columns=fields):
        for record in batch.to_pylist():
            prior = resolved.get(record["cleaned_record_id"])
            if prior is not None and (
                candidate := _proposal(record, prior, unit_map)
            ) is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda row: row["cleaned_record_id"])

    gold = {
        row["audit_case_id"].split(":", 1)[1]: row
        for row in _read_jsonl(args.gold)[1:]
    }
    review = []
    gold_decisions = []
    for candidate in candidates:
        case = gold.get(candidate["cleaned_record_id"])
        if case is None:
            review.append(candidate)
            continue
        expected = case["expected"]
        if expected["status"] != "ok":
            action, reason = "keep_original", f"frozen_gold_{expected['status']}"
        else:
            expected_entry = expected["measurements"][0]
            expected_rule = _rule(
                unit_map, candidate["canonical_endpoint_name"], expected_entry["unit"]
            )
            expected_mapped = _mapped(expected_rule)
            proposed_mapped = (
                candidate["canonical_unit"],
                Decimal(candidate["unit_scale"]),
            )
            values_match = bool(
                expected_mapped
                and Decimal(expected_entry["measurement"]) * expected_mapped[1]
                == Decimal(candidate["parsed_measurement"]) * proposed_mapped[1]
            )
            if expected_mapped != proposed_mapped or not values_match:
                action, reason = "keep_original", "frozen_gold_ok_proposal_mismatch"
            else:
                action, reason = "accept_absolute", "frozen_gold_ok_exact_match"
        gold_decisions.append(
            {
                "cleaned_record_id": candidate["cleaned_record_id"],
                "decision": action,
                "reason_code": reason,
                "reviewer_id": "frozen_gold_v10_2",
            }
        )

    _write_jsonl(output / "candidates.jsonl", candidates)
    _write_jsonl(output / "gold_decisions.jsonl", gold_decisions)
    packet_dir = output / "primary_packets"
    packet_dir.mkdir()
    for start in range(0, len(review), 200):
        _write_jsonl(
            packet_dir / f"packet_{start // 200:04d}.jsonl",
            review[start : start + 200],
        )
    _write_json(
        output / "manifest.json",
        {
            "version": "dili_finite_point_rescue_candidates.v1",
            "inputs": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in (args.input, args.records, args.units, args.gold)
            },
            "candidate_count": len(candidates),
            "gold_decision_counts": dict(
                sorted(Counter(row["decision"] for row in gold_decisions).items())
            ),
            "review_count": len(review),
            "packet_count": (len(review) + 199) // 200,
            "candidate_sha256": file_sha256(output / "candidates.jsonl"),
            "candidate_policy": {
                "current_status": "unsure",
                "numeric_grammar": "finite source point with optional uncertainty",
                "unit_sources": ["source", "matching_single_llm_proposal"],
                "unit_mapping": (
                    "reviewed exact map only; conflicting mapped units excluded"
                ),
                "allowed_guard_reasons": sorted(
                    reason or "none" for reason in ALLOWED_UNIT_GUARDS
                ),
            },
            "review_policy": {
                "packet_size": 200,
                "decisions": sorted(DECISIONS),
                "promotion": "frozen gold exact match or independent double accept",
            },
        },
    )


def _load_decisions(
    paths: list[Path], expected: set[str], label: str
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            record_id = _text(row.get("cleaned_record_id"))
            if (
                record_id in decisions
                or row.get("decision") not in DECISIONS
                or not _text(row.get("reason_code"))
                or not _text(row.get("reviewer_id"))
            ):
                raise ValueError(f"invalid {label} decision in {path}: {record_id!r}")
            decisions[record_id] = row
    if set(decisions) != expected:
        raise ValueError(
            f"{label} coverage mismatch: missing={len(expected - set(decisions))}, "
            f"extra={len(set(decisions) - expected)}"
        )
    return decisions


def consolidate(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = _read_jsonl(args.prepared / "candidates.jsonl")
    by_id = {row["cleaned_record_id"]: row for row in candidates}
    gold_rows = _read_jsonl(args.prepared / "gold_decisions.jsonl")
    gold = {row["cleaned_record_id"]: row for row in gold_rows}
    review_ids = set(by_id) - set(gold)
    primary = _load_decisions(
        sorted(args.primary.glob("*.jsonl")), review_ids, "primary"
    )
    checker_expected = {
        record_id
        for record_id, row in primary.items()
        if row["decision"] == "accept_absolute"
    }
    checker = _load_decisions(
        sorted(args.checker.glob("*.jsonl")), checker_expected, "checker"
    )
    durable = []
    accepted = set()
    for record_id in sorted(by_id):
        if record_id in gold:
            first, second = gold[record_id], None
            action = first["decision"]
        else:
            first = primary[record_id]
            second = checker.get(record_id)
            if second and second["reviewer_id"] == first["reviewer_id"]:
                raise ValueError(f"checker is not independent: {record_id}")
            action = (
                "accept_absolute"
                if first["decision"] == "accept_absolute"
                and second
                and second["decision"] == "accept_absolute"
                else "keep_original"
            )
        if action == "accept_absolute":
            accepted.add(record_id)
        proposal = by_id[record_id]
        durable.append(
            {
                "cleaned_record_id": record_id,
                "proposal": {
                    key: proposal[key]
                    for key in (
                        "source_row_uid",
                        "source_id",
                        "canonical_endpoint_name",
                        "parsed_measurement",
                        "parsed_uncertainty",
                        "selected_unit",
                        "selected_unit_origin",
                        "canonical_unit",
                        "unit_scale",
                    )
                },
                "primary": first,
                "checker": second,
                "final_action": action,
            }
        )
    _write_jsonl(args.output / "reviewed_rescue_decisions.jsonl", durable)

    table = pq.read_table(args.input)
    rows = table.to_pylist()
    original_status = Counter(row["status"] for row in rows)
    for row in rows:
        if row["cleaned_record_id"] not in accepted:
            continue
        proposal = by_id[row["cleaned_record_id"]]
        row.update(
            status="ok",
            measurements_json=json.dumps(
                [
                    {
                        "measurement": proposal["parsed_measurement"],
                        "unit": proposal["selected_unit"],
                    }
                ],
                separators=(",", ":"),
            ),
            quantity_count=1,
            assignment_method="reviewed_deterministic_finite_point_rescue",
            assignment_guard_reason=None,
        )
    output_mapping = args.output / "measurement_resolution.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        output_mapping,
        compression="zstd",
    )
    final_status = Counter(row["status"] for row in rows)
    candidate_manifest = json.loads(
        (args.prepared / "manifest.json").read_text(encoding="utf-8")
    )
    _write_json(
        args.output / "manifest.json",
        {
            "version": "dili_measurement_resolution.v5",
            "predecessor": {
                "path": str(args.input.relative_to(ROOT)),
                "sha256": file_sha256(args.input),
                "rows": len(rows),
            },
            "candidate_manifest_sha256": file_sha256(
                args.prepared / "manifest.json"
            ),
            "candidate_sha256": candidate_manifest["candidate_sha256"],
            "candidate_input_hashes": candidate_manifest["inputs"],
            "reviewed_decisions_sha256": file_sha256(
                args.output / "reviewed_rescue_decisions.jsonl"
            ),
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "kept_count": len(candidates) - len(accepted),
            "status_counts_before": dict(sorted(original_status.items())),
            "status_counts_after": dict(sorted(final_status.items())),
            "measurement_resolution_sha256": file_sha256(output_mapping),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    prep.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    prep.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    prep.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=prepare)
    join = subparsers.add_parser("consolidate")
    join.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    join.add_argument("--prepared", type=Path, required=True)
    join.add_argument("--primary", type=Path, required=True)
    join.add_argument("--checker", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    join.set_defaults(func=consolidate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
