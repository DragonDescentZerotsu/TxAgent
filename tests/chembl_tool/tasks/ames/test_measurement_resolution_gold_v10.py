from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_resolution import (
    prompt_manifest,
    prompt_row_fields,
    render_prompt,
)

REPO = Path(__file__).resolve().parents[4]
GOLD = REPO / "tests/chembl_tool/common/measurement_resolution_quality/gold/ames.v10.1.jsonl"
STATUSES = {"ok", "relative", "unavailable", "unsure"}


def _load() -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in GOLD.read_text().splitlines() if line]
    return rows[0], rows[1:]


def _clean(value):
    return None if isinstance(value, float) and math.isnan(value) else value


def _stage1_rows(cases: list[dict], path: Path) -> dict[str, dict]:
    ids = {case["audit_case_id"].split(":", 1)[1] for case in cases}
    fields = set().union(*(case["input"].keys() for case in cases))
    columns = sorted(fields | {
        "cleaned_record_id", "source_row_uid", "source_id",
        "canonical_endpoint_name", "measurement_resolution_route",
    })
    found = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=columns):
        for row in batch.to_pylist():
            record_id = str(row["cleaned_record_id"])
            if record_id in ids:
                assert record_id not in found
                found[record_id] = row
    return found


def test_gold_is_frozen_complete_and_source_reviewed_before_replay():
    manifest, cases = _load()
    assert manifest["corpus_version"] == "ames_measurement_resolution_gold.v2"
    assert manifest["task_id"] == "ames"
    assert manifest["expectation_update_policy"] == "source_evidence_only_before_model_pilot"
    assert manifest["labelled_before_any_model_run"] is True
    assert manifest["routing_version"] == MEASUREMENT_ROUTING_VERSION
    assert manifest["cases"] == len(cases) == 500
    assert manifest["route_counts"] == {"extract": 500}
    assert len({case["audit_case_id"] for case in cases}) == 500
    assert len({case["source_row_uid"] for case in cases}) == 500
    assert manifest["model_outputs_used_as_label_evidence"] is False
    assert all("source" in case["label_provenance"] for case in cases)


def test_gold_counts_and_expected_shapes_are_self_consistent():
    manifest, cases = _load()
    decimal = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
    for field, key in (("source_counts", "source_id"), ("stratum_counts", "stratum")):
        assert manifest[field] == dict(sorted(Counter(case[key] for case in cases).items()))
    assert manifest["status_counts"] == dict(sorted(Counter(case["expected"]["status"] for case in cases).items()))
    for case in cases:
        expected = case["expected"]
        assert expected["status"] in STATUSES
        assert bool(expected["measurements"]) == (expected["status"] == "ok")
        assert len(expected["measurements"]) <= 1
        assert all(decimal.fullmatch(pair["measurement"]) and pair["unit"] for pair in expected["measurements"])


def test_gold_rows_match_the_frozen_stage1_source_fields_exactly():
    manifest, cases = _load()
    stage1 = manifest["stage1"]
    records = REPO / stage1["cleaned_records_path"]
    profile = REPO / stage1["endpoint_profile_path"]
    assert file_sha256(records) == stage1["cleaned_records_sha256"]
    assert file_sha256(profile) == stage1["endpoint_profile_sha256"]
    found = _stage1_rows(cases, records)
    assert len(found) == 500
    for case in cases:
        row = found[case["audit_case_id"].split(":", 1)[1]]
        assert row["source_row_uid"] == case["source_row_uid"]
        assert row["source_id"] == case["source_id"]
        assert row["canonical_endpoint_name"] == case["canonical_endpoint_name"]
        assert row["measurement_resolution_route"] == case["route_bucket"]
        assert all(_clean(row.get(key)) == _clean(value) for key, value in case["input"].items())


def test_prompt_contract_keeps_scales_source_printed_and_hides_raw_outcome():
    fields = prompt_row_fields("mutagenicity_outcomes")
    rendered = render_prompt("fixed_mutation")
    assert "mutagenicity_result" not in fields
    assert "copy the printed coefficient and keep the power in the unit" in rendered
    assert "retain only the component attached to the selected value" in rendered
    assert prompt_manifest()["template_sha256"] == file_sha256(
        REPO / "data/processing/evidence_library/versions/v10/tasks/ames/prompts/measurement_resolution_v2.jinja"
    )
