from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing import (
    build_measurement_candidates as writer,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    candidate_set_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_candidates import (
    CANDIDATE_GENERATOR_VERSION,
    MAX_CANDIDATE_FIELD_BYTES,
)


def _row(*, record_id: str = "record", unit: str = "mutation frequency") -> dict:
    candidates = [
        {
            "candidate_id": "1",
            "measurement": "1",
            "unit": unit,
            "rule_id": "source_phrase_metric.v1",
            "evidence": ["measurement_text: 1", f"support_text: {unit}"],
            "hints": ["point"],
        }
    ]
    payload = json.dumps(candidates, separators=(",", ":"), sort_keys=True)
    return {
        "cleaned_record_id": record_id,
        "source_row_uid": "source-row",
        "source_id": "fixed_mutation",
        "canonical_endpoint_name": "microbial_forward_mutation",
        "measurement_candidates_json": payload,
        "candidate_set_sha256": candidate_set_sha256(payload),
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_generator_version": CANDIDATE_GENERATOR_VERSION,
        "candidate_count": 1,
    }


def test_frozen_development_fixture_has_complete_unique_cohort(tmp_path: Path):
    cases = writer._gold_cases(writer.DEFAULT_GOLD)
    assert len(cases) == writer.GOLD_CASES
    assert len({case["audit_case_id"] for case in cases}) == writer.GOLD_CASES
    copied = tmp_path / "ames.v10.jsonl"
    copied.write_bytes(writer.DEFAULT_GOLD.read_bytes())
    with pytest.raises(ValueError, match="not the frozen gold"):
        writer._gold_cases(copied)


def test_writer_projects_every_generator_evidence_field():
    assert set(writer.EVIDENCE_FIELDS) <= set(writer._source_columns())


def _gold_projection_case(measurement_text: str) -> dict:
    return {
        "input": {"measurement_text": measurement_text},
        "expected": {
            "measurements": [{"measurement": "2.5", "unit": "10^-6 mutation rate"}]
        },
    }


def test_gold_oracle_projects_legacy_scaled_pair_to_stage1_decimal():
    case = _gold_projection_case("0.0000025")
    assert writer._gold_expected_pair(case) == ("0.0000025", "mutation rate", True)


def test_gold_oracle_keeps_scale_when_coefficient_is_stage1_value():
    case = _gold_projection_case("2.5")
    assert writer._gold_expected_pair(case) == ("2.5", "10^-6 mutation rate", False)


def test_frozen_gold_has_34_legacy_scaled_pairs_to_project():
    cases = writer._gold_cases(writer.DEFAULT_GOLD)
    projected = Counter(
        case["source_id"]
        for case in cases
        if case["expected"]["status"] == "ok" and writer._gold_expected_pair(case)[2]
    )
    assert projected == {"fixed_mutation": 33, "mutagenicity_mechanism": 1}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cleaned_record_id", "", "empty identity"),
        ("source_row_uid", None, "empty identity"),
        ("source_id", "other", "unknown source"),
        ("canonical_endpoint_name", "", "empty canonical endpoint"),
    ],
)
def test_candidate_validator_rejects_incomplete_provenance(field, value, message):
    row = _row()
    row[field] = value
    with pytest.raises(ValueError, match=message):
        writer._validate_candidate_rows([row], {})


def test_candidate_validator_rejects_oversized_field():
    row = _row(unit="x" * (MAX_CANDIDATE_FIELD_BYTES + 1))
    with pytest.raises(ValueError, match="field byte bound"):
        writer._validate_candidate_rows([row], {})


def test_manifest_claims_are_recomputed(tmp_path: Path):
    artifact = tmp_path / "candidates.parquet"
    pq.write_table(pa.table({"value": [1]}), artifact)
    manifest = {
        "candidate_path": str(artifact.resolve()),
        "candidate_schema": pq.read_schema(artifact).to_string(),
        "maximum_candidates": writer.MAX_CANDIDATES,
        "maximum_candidate_field_bytes": writer.MAX_CANDIDATE_FIELD_BYTES,
        "maximum_candidate_json_bytes": writer.MAX_CANDIDATE_JSON_BYTES,
        "development_gold_validation": {
            "construction_input": True,
            "independent_generalization_evidence": False,
        },
    }
    writer._validate_manifest_claims(artifact, manifest)
    manifest["candidate_schema"] = "forged"
    with pytest.raises(ValueError, match="candidate_schema"):
        writer._validate_manifest_claims(artifact, manifest)


def test_implementation_roles_are_bound_to_canonical_files():
    generator = writer.TASK_ROOT / "starling_measurement_candidates.py"
    reference = {"path": str(generator), "sha256": file_sha256(generator)}
    manifest = {
        "implementation": {
            role: dict(reference)
            for role in ("writer", "generator", "compiler_contract", "source_fields")
        }
    }
    with pytest.raises(ValueError, match="implementation role mismatch"):
        writer._validate_implementation(manifest)
