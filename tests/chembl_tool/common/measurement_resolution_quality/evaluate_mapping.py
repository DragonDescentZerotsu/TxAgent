"""Score a frozen measurement-resolution mapping against reviewed gold rows.

This is an offline evaluator. It never calls a model and is intentionally not a
pytest test: a live replay is an experiment whose output is then scored here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v1.normalization.measurements import (
    normalize_measurement_and_unit,
    parse_point_measurement,
)


HERE = Path(__file__).resolve().parent
DEFAULT_GOLD = HERE / "gold/bbb_martins.v8.jsonl"


def _load_gold(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    manifest, cases = rows[0], rows[1:]
    if manifest["cases"] != len(cases):
        raise ValueError(f"gold manifest count does not match {path}")
    by_id = {
        case["audit_case_id"].split(":", 1)[-1]: case for case in cases
    }
    if len(by_id) != len(cases):
        raise ValueError(f"duplicate audit_case_id in {path}")
    return manifest, by_id


def _pairs(row: dict[str, Any]) -> list[dict[str, str]]:
    value = row.get("measurements_json") or "[]"
    pairs = json.loads(value) if isinstance(value, str) else value
    if not isinstance(pairs, list):
        raise ValueError("measurements_json must contain a list")
    return pairs


def _same_number(left: object, right: object) -> bool:
    try:
        return Decimal(str(left).strip()) == Decimal(str(right).strip())
    except (InvalidOperation, ValueError):
        return False


def _clean_unit(value: object) -> str:
    return " ".join(str(value or "").split())


def _answers(expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the reviewed answer plus any explicitly accepted alternatives."""
    return [expected, *expected.get("alternatives", [])]


def _canonical_pair(pair: dict[str, Any], *, task: str) -> tuple[float, str] | None:
    """Fold harmless spelling/scale variants through the production normalizer."""
    try:
        normalized = normalize_measurement_and_unit(
            pair["measurement"], pair["unit"], task=task
        )
        value = parse_point_measurement(normalized.canonical_measurement).value
    except (KeyError, TypeError, ValueError):
        return None
    if value is None or not math.isfinite(value):
        return None
    return float(value), normalized.canonical_unit


def _pair_matches(
    predicted: dict[str, Any], gold: dict[str, Any], *, task: str
) -> tuple[bool, bool, bool]:
    raw_measurement = _same_number(predicted.get("measurement"), gold.get("measurement"))
    raw_unit = _clean_unit(predicted.get("unit")) == _clean_unit(gold.get("unit"))
    predicted_canonical = _canonical_pair(predicted, task=task)
    gold_scalar = gold.get("expected_scalar")
    gold_unit = gold.get("expected_canonical_unit")
    scalar_match = raw_measurement
    unit_match = raw_unit
    if predicted_canonical is not None and gold_scalar is not None and gold_unit:
        scalar_match = math.isclose(
            predicted_canonical[0], float(gold_scalar), rel_tol=1e-9, abs_tol=1e-15
        )
        unit_match = predicted_canonical[1] == gold_unit
    return scalar_match, unit_match, scalar_match and unit_match


def _score_case(
    case: dict[str, Any], prediction: dict[str, Any], *, task: str
) -> dict[str, Any]:
    expected = case["expected"]
    gold_pairs = expected["measurements"]
    predicted_pairs = _pairs(prediction)
    answers = _answers(expected)
    status_match = any(
        prediction.get("status") == answer["status"] for answer in answers
    )

    measurement_match = unit_match = pair_match = None
    raw_measurement_match = raw_unit_match = raw_pair_match = None
    if expected["status"] == "ok":
        same_length = len(predicted_pairs) == len(gold_pairs)
        raw_measurement_match = same_length and all(
            _same_number(predicted["measurement"], gold["measurement"])
            for predicted, gold in zip(predicted_pairs, gold_pairs)
        )
        raw_unit_match = same_length and all(
            _clean_unit(predicted["unit"]) == _clean_unit(gold["unit"])
            for predicted, gold in zip(predicted_pairs, gold_pairs)
        )
        raw_pair_match = raw_measurement_match and raw_unit_match
        answer_matches = []
        for answer in answers:
            answer_pairs = answer["measurements"]
            if answer["status"] != "ok" or len(predicted_pairs) != len(answer_pairs):
                continue
            comparisons = [
                _pair_matches(predicted, gold, task=task)
                for predicted, gold in zip(predicted_pairs, answer_pairs)
            ]
            answer_matches.append(
                (
                    all(item[0] for item in comparisons),
                    all(item[1] for item in comparisons),
                    all(item[2] for item in comparisons),
                )
            )
        measurement_match = any(item[0] for item in answer_matches)
        unit_match = any(item[1] for item in answer_matches)
        pair_match = any(item[2] for item in answer_matches)

    return {
        "cleaned_record_id": prediction["cleaned_record_id"],
        "source_id": case["source_id"],
        "endpoint_name": case["input"].get("endpoint_name"),
        "measurement_text": case["input"].get("measurement_text"),
        "unit_text": case["input"].get("unit_text"),
        "gold_status": expected["status"],
        "predicted_status": prediction.get("status"),
        "gold_measurements": json.dumps(gold_pairs, ensure_ascii=False),
        "predicted_measurements": json.dumps(predicted_pairs, ensure_ascii=False),
        "status_match": status_match,
        "measurement_match": measurement_match,
        "unit_match": unit_match,
        "pair_match": pair_match,
        "raw_measurement_match": raw_measurement_match,
        "raw_unit_match": raw_unit_match,
        "raw_pair_match": raw_pair_match,
        "record_match": status_match and (
            pair_match if expected["status"] == "ok" else not predicted_pairs
        ),
        "ok_pair_correct": bool(
            prediction.get("status") == "ok"
            and expected["status"] == "ok"
            and pair_match
        ),
        "assignment_method": prediction.get("assignment_method"),
    }


def _accuracy(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    applicable = [row[field] for row in rows if row[field] is not None]
    correct = sum(value is True for value in applicable)
    return {
        "correct": correct,
        "total": len(applicable),
        "accuracy": correct / len(applicable) if applicable else None,
    }


def evaluate(
    mapping_path: Path,
    gold_path: Path = DEFAULT_GOLD,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_manifest, gold = _load_gold(gold_path)
    predictions = pq.read_table(mapping_path).to_pylist()
    unknown = sorted(
        str(row["cleaned_record_id"])
        for row in predictions
        if str(row["cleaned_record_id"]) not in gold
    )
    if unknown:
        raise ValueError(
            f"mapping contains {len(unknown)} row(s) absent from gold; first={unknown[0]}"
        )
    scored = [
        _score_case(
            gold[str(row["cleaned_record_id"])],
            row,
            task=str(gold_manifest["task_id"]),
        )
        for row in predictions
    ]
    gold_ok = [row for row in scored if row["gold_status"] == "ok"]
    recovered_ok = [row for row in gold_ok if row["predicted_status"] == "ok"]
    model_target_cases = (gold_manifest.get("route_counts") or {}).get("extract")
    metrics = {
        "gold_path": str(gold_path),
        "gold_corpus_version": gold_manifest["corpus_version"],
        "gold_cases": len(gold),
        "mapping_path": str(mapping_path),
        "evaluated_cases": len(scored),
        "gold_coverage": len(scored) / len(gold),
        "model_target_cases": model_target_cases,
        "model_target_coverage": (
            len(scored) / model_target_cases if model_target_cases else None
        ),
        "status": _accuracy(scored, "status_match"),
        "gold_ok_recovery": {
            "correct": len(recovered_ok),
            "total": len(gold_ok),
            "accuracy": len(recovered_ok) / len(gold_ok) if gold_ok else None,
        },
        "measurement": _accuracy(scored, "measurement_match"),
        "unit": _accuracy(scored, "unit_match"),
        "measurement_and_unit": _accuracy(scored, "pair_match"),
        "raw_measurement": _accuracy(scored, "raw_measurement_match"),
        "raw_unit": _accuracy(scored, "raw_unit_match"),
        "raw_measurement_and_unit": _accuracy(scored, "raw_pair_match"),
        "measurement_given_ok_prediction": _accuracy(
            recovered_ok, "measurement_match"
        ),
        "unit_given_ok_prediction": _accuracy(recovered_ok, "unit_match"),
        "pair_given_ok_prediction": _accuracy(
            [row for row in scored if row["predicted_status"] == "ok"],
            "ok_pair_correct",
        ),
        "whole_record": _accuracy(scored, "record_match"),
        "by_source": {
            source: {
                "cases": sum(row["source_id"] == source for row in scored),
                "whole_record": _accuracy(
                    [row for row in scored if row["source_id"] == source],
                    "record_match",
                ),
            }
            for source in sorted({row["source_id"] for row in scored})
        },
    }
    return metrics, scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    metrics, rows = evaluate(args.mapping, args.gold)
    output_dir = args.output_dir or args.mapping.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "quality_metrics.json"
    rows_path = output_dir / "quality_rows.csv"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")
    print(f"wrote {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
