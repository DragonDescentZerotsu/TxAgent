from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v10.tasks.dili import (
    starling_measurement_resolution as measurement_config,
)


ROOT = Path(__file__).resolve().parents[6]
GOLD_PATH = (
    ROOT
    / "tests/chembl_tool/common/measurement_resolution_quality/gold/dili.v10.3.jsonl"
)
STAGE1_PATH = ROOT / "data/evidence_libraries/dili/v10/01_cleaned/records.parquet"
PLAIN_DECIMAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def test_dili_gold_is_complete_source_review_over_active_stage1_rows() -> None:
    lines = [json.loads(line) for line in GOLD_PATH.read_text().splitlines()]
    manifest, cases = lines[0], lines[1:]

    assert manifest["task_id"] == "dili"
    assert manifest["successor_of"] == "dili.v10.2.jsonl"
    assert manifest["cases"] == manifest["reviewed_cases"] == len(cases) == 1192
    assert manifest["labels_frozen_before_low_reasoning_gold_pilot"] is True
    assert manifest["model_outputs_used_as_label_evidence"] is False
    frozen_prompt = manifest["prompt_at_label_freeze"]
    assert frozen_prompt["prompt_version"] == "dili_measurement_resolution_prompt.v8"
    assert frozen_prompt["prompt_version"] != measurement_config.PROMPT_VERSION
    assert set(frozen_prompt["rendered_sha256"]) == set(measurement_config.SOURCE_IDS)
    assert manifest["deterministic_acceptance_policy"]["version"] == (
        measurement_config.DILI_MEASUREMENT_ROUTING_VERSION
    )
    assert manifest["stage1"]["cleaned_records_sha256"] == file_sha256(STAGE1_PATH)
    stage1_artifacts = {
        "cleaned_records": STAGE1_PATH,
        "clean_manifest": STAGE1_PATH.with_name("manifest.json"),
        "endpoint_profile": measurement_config.DEFAULT_PROFILE_PATH,
        "release_manifest": STAGE1_PATH.parents[1] / "manifest.json",
    }
    for label, expected_path in stage1_artifacts.items():
        assert (ROOT / manifest["stage1"][f"{label}_path"]).resolve() == (
            expected_path.resolve()
        )
        assert manifest["stage1"][f"{label}_sha256"] == file_sha256(expected_path)
    source_snapshot = ROOT / manifest["source_snapshot"]
    assert source_snapshot.resolve() == (
        measurement_config.TASK_ROOT / "source_manifest.json"
    ).resolve()
    assert manifest["source_snapshot_sha256"] == file_sha256(source_snapshot)
    assert len({case["audit_case_id"] for case in cases}) == len(cases)
    assert [case["review_index"] for case in cases] == list(
        range(1, len(cases) + 1)
    )

    case_ids = {case["audit_case_id"].split(":", 1)[1] for case in cases}
    active = {}
    stage1_accept_ids = set()
    prompt_fields = {
        field
        for source_id in measurement_config.SOURCE_IDS
        for field in measurement_config.prompt_row_fields(source_id)
    }
    parquet = pq.ParquetFile(STAGE1_PATH)
    for batch in parquet.iter_batches(
        columns=sorted(
            {
            "cleaned_record_id",
            "canonical_endpoint_name",
            "endpoint_name",
            "measurement_resolution_route",
            "measurement_resolution_rule_id",
            "measurement_resolution_exact_measurement",
            "measurement_resolution_exact_unit",
            "source_id",
            "source_row_uid",
                *prompt_fields,
            }
        )
    ):
        for row in batch.to_pylist():
            if row["measurement_resolution_route"] == "accept":
                stage1_accept_ids.add(str(row["cleaned_record_id"]))
            if row["cleaned_record_id"] in case_ids:
                active[str(row["cleaned_record_id"])] = row
    assert len(active) == len(cases)
    for case in cases:
        record = active[case["audit_case_id"].split(":", 1)[1]]
        route = measurement_config.route_measurement(record)
        assert route.bucket == record["measurement_resolution_route"] == case["route_bucket"]
        assert route.rule_id == record["measurement_resolution_rule_id"] == case["route_rule_id"]
        assert record["source_row_uid"] == case["source_row_uid"]
        assert record["source_id"] == case["source_id"]
        assert record["canonical_endpoint_name"] == case["canonical_endpoint_name"]
        assert record["endpoint_name"] == case["input"]["endpoint_name"]
        assert case["input"] == {
            field: record.get(field)
            for field in measurement_config.prompt_row_fields(record["source_id"])
        }

        expected = case["expected"]
        measurements = expected["measurements"]
        if expected["status"] == "ok":
            assert len(measurements) == 1
            assert PLAIN_DECIMAL.fullmatch(measurements[0]["measurement"])
            assert measurements[0]["unit"].strip()
        else:
            assert expected["status"] in {"relative", "unsure", "unavailable"}
            assert measurements == []
        if route.bucket == "accept":
            assert expected == {
                "status": "ok",
                "measurements": [
                    {"measurement": route.measurement_text, "unit": route.unit_text}
                ],
            }

    assert stage1_accept_ids == {
        case["audit_case_id"].split(":", 1)[1]
        for case in cases
        if case["route_bucket"] == "accept"
    }

    assert manifest["status_counts"] == dict(
        sorted(Counter(case["expected"]["status"] for case in cases).items())
    )
    assert manifest["route_counts"] == dict(
        sorted(Counter(case["route_bucket"] for case in cases).items())
    )
    assert manifest["source_counts"] == dict(
        sorted(Counter(case["source_id"] for case in cases).items())
    )
