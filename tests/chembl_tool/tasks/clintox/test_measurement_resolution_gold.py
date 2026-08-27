import json
from collections import Counter
from pathlib import Path

import pytest

from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
    gold_extract_ids,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
    parse_point_measurement,
)
from tools.chembl_tool.tasks.clintox.starling_categorical_response import POLICY


GOLD = Path(
    "tools/chembl_tool/tasks/clintox/data_processing/stage2_review_v1/"
    "measurement_resolution_gold.jsonl"
)


def _load():
    rows = [json.loads(line) for line in GOLD.read_text().splitlines() if line]
    return rows[0], rows[1:]


def test_clintox_gold_contract_and_current_replay_subset():
    manifest, cases = _load()
    assert manifest["corpus_version"] == "clintox_measurement_resolution_gold.v1"
    assert manifest["model_outputs_used_as_label_evidence"] is False
    assert len(cases) == len({row["audit_case_id"] for row in cases}) == 350
    assert Counter(row["source_id"] for row in cases) == {
        source_id: 50 for source_id in manifest["source_counts"]
    }
    assert Counter(row["route_bucket"] for row in cases) == {
        "accept": 28,
        "categorical": 72,
        "extract": 200,
        "reject": 50,
    }
    assert len(gold_extract_ids(GOLD)) == 200
    assert all(
        row["review"]["route_conflict_reviewed"]
        for row in cases
        if row["routing_audit_status"] == "semantic_conflict"
    )


def test_clintox_gold_measurements_and_categorical_encodings_are_reproducible():
    _, cases = _load()
    specs = {spec.scale_id: spec for spec in POLICY.controlled_measurements}
    for case in cases:
        expected = case["expected"]
        if case["expected_kind"] == "categorical":
            record = {
                "source_id": case["source_id"],
                **case["input"],
                **case["controlled_inputs"],
            }
            encoded = POLICY.encode(record)
            assert encoded is not None
            category = specs[encoded.encoder_id].category_for_value(encoded.value)
            assert category is not None
            assert expected == {
                "encoder_id": encoded.encoder_id,
                "measurement": encoded.measurement_text,
                "value": encoded.value,
                "unit": encoded.unit,
                "category_id": category.category_id,
                "category_rank": category.rank,
            }
            continue

        assert expected["status"] in {"ok", "relative", "unsure", "unavailable"}
        assert (expected["status"] == "ok") == bool(expected["measurements"])
        for item in expected["measurements"]:
            pair = normalize_measurement_and_unit(
                item["measurement"], item["unit"], task="clintox"
            )
            scalar = parse_point_measurement(pair.canonical_measurement).value
            assert scalar == pytest.approx(item["expected_scalar"], rel=1e-12)
            assert pair.canonical_unit == item["expected_canonical_unit"]
