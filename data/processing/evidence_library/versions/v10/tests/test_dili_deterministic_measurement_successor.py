from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v10.tasks.dili.data_processing.build_deterministic_measurement_successor import (
    _proposal,
    _raw_proposal,
)


def test_raw_proposal_requires_one_matching_measurement() -> None:
    payload = json.dumps(
        {"status": "ok", "measurements": [{"measurement": "1,200", "unit": "%"}]}
    )
    assert _raw_proposal(payload, "1200") == "%"
    assert _raw_proposal(payload, "1201") == ""


def test_candidate_rejects_disagreeing_source_and_model_units() -> None:
    record = {
        "cleaned_record_id": "r1",
        "source_row_uid": "s1",
        "source_id": "dili_v1",
        "endpoint_name": "viability",
        "canonical_endpoint_name": "viability",
        "measurement_text": "25 +/- 2",
        "unit_text": "%",
        "support_text": "Viability was 25 +/- 2 percent.",
    }
    resolved = {
        "status": "unsure",
        "raw_response_json": json.dumps(
            {"status": "ok", "measurements": [{"measurement": "25", "unit": "ratio"}]}
        ),
        "assignment_guard_reason": "unit_not_exact_declared_unit",
    }
    units = {
        ("dili", "*", "%"): {"action": "map", "canonical_unit": "percent", "scale": "1"},
        ("dili", "*", "ratio"): {"action": "map", "canonical_unit": "ratio", "scale": "1"},
    }
    with patch(
        "data.processing.evidence_library.versions.v10.tasks.dili.data_processing."
        "build_deterministic_measurement_successor.resolution._guard_reason",
        return_value=None,
    ):
        assert _proposal(record, resolved, units) is None


def test_published_v5_changes_only_double_reviewed_candidates() -> None:
    root = Path(__file__).resolve().parents[1]
    library = root.parents[4] / "data/evidence_libraries/dili/v10"
    v4 = pq.read_table(library / "measurement_resolution_v4/measurement_resolution.parquet")
    v5 = pq.read_table(library / "measurement_resolution_v5/measurement_resolution.parquet")
    decisions = {
        row["cleaned_record_id"]
        for row in map(
            json.loads,
            (library / "measurement_resolution_v5/reviewed_rescue_decisions.jsonl")
            .read_text()
            .splitlines(),
        )
        if row["final_action"] == "accept_absolute"
    }
    assert len(decisions) == 7_618
    changed = set()
    for prior, successor in zip(v4.to_pylist(), v5.to_pylist(), strict=True):
        assert prior["cleaned_record_id"] == successor["cleaned_record_id"]
        if prior != successor:
            changed.add(prior["cleaned_record_id"])
            assert prior["status"] == "unsure"
            assert successor["status"] == "ok"
            assert successor["assignment_method"] == (
                "reviewed_deterministic_finite_point_rescue"
            )
    assert changed == decisions
