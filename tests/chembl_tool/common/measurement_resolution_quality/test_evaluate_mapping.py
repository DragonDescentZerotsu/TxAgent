from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests.chembl_tool.common.measurement_resolution_quality.evaluate_mapping import (
    evaluate,
)


def _case(
    record_id: str,
    source_id: str,
    status: str,
    measurements: list[dict[str, str]],
    *,
    alternatives: list[dict] | None = None,
) -> dict:
    expected = {"status": status, "measurements": measurements}
    if alternatives:
        expected["alternatives"] = alternatives
    return {
        "audit_case_id": f"fixture:{record_id}",
        "source_id": source_id,
        "input": {
            "endpoint_name": "endpoint",
            "measurement_text": "text",
            "unit_text": "unit",
        },
        "expected": expected,
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    pair = {"measurement": "1", "unit": "mg/L"}
    alternative = {
        "status": "ok",
        "measurements": [{"measurement": "2", "unit": "mg/L"}],
    }
    cases = [
        _case("ok", "source-a", "ok", [pair], alternatives=[alternative]),
        _case("unavailable", "source-a", "unavailable", []),
        _case("wrong", "source-b", "unsure", []),
    ]
    gold = tmp_path / "gold.jsonl"
    manifest = {"cases": 3, "corpus_version": "fixture.v1", "task_id": "bbb_martins"}
    gold.write_text("\n".join(json.dumps(row) for row in [manifest, *cases]) + "\n")
    predictions = [
        {
            "cleaned_record_id": "ok",
            "status": "ok",
            "measurements_json": json.dumps(alternative["measurements"]),
        },
        {
            "cleaned_record_id": "unavailable",
            "status": "unavailable",
            "measurements_json": "[]",
        },
        {
            "cleaned_record_id": "wrong",
            "status": "ok",
            "measurements_json": json.dumps([pair]),
        },
    ]
    mapping = tmp_path / "mapping.parquet"
    pq.write_table(pa.Table.from_pylist(predictions), mapping)
    return mapping, gold


def test_evaluate_scores_alternatives_and_aggregates_sources(tmp_path: Path) -> None:
    mapping, gold = _write_fixture(tmp_path)

    metrics, rows = evaluate(mapping, gold)

    assert metrics["status"] == {"correct": 2, "total": 3, "accuracy": 2 / 3}
    assert metrics["whole_record"] == {"correct": 2, "total": 3, "accuracy": 2 / 3}
    assert metrics["pair_given_ok_prediction"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert metrics["by_source"]["source-a"]["whole_record"]["accuracy"] == 1.0
    assert metrics["by_source"]["source-b"]["whole_record"]["accuracy"] == 0.0
    assert rows[0]["pair_match"] is True
    assert rows[0]["raw_pair_match"] is False
