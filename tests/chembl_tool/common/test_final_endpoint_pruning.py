import json
from pathlib import Path

import pandas as pd
import pytest

from tools.chembl_tool.common.starling.final_endpoint_pruning import (
    DECISIONS_FILENAME,
    REVIEWS_FILENAME,
    SCHEMA_VERSION,
    VERSION,
    _compact_semantic_row,
    _validate_candidate_review,
    consensus_decisions,
    render_review_prompt,
    semantic_review_columns,
    supported_gap,
    validate_review_response,
)
from tools.chembl_tool.common.starling.normalization.audit import write_parquet
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.record_collapse import (
    build_collapsed_record_stage,
)


def test_supported_gap_is_inclusive_100x_not_threefold():
    assert supported_gap([1, 1.1, 1.2, 3.6, 3.7, 3.8]) is None
    assert supported_gap([1, 1.1, 1.2, 118.8, 130, 140]) is None
    gap = supported_gap([1, 1.1, 1.2, 120, 130, 140])
    assert gap is not None
    assert gap["gap_lower_value"] == 1.2
    assert gap["gap_upper_value"] == 120
    assert gap["gap_decades"] == pytest.approx(2.0)


def test_supported_gap_uses_logged_units_without_double_logging():
    values = [-8.2, -8.1, -8.0, -6.0, -5.9, -5.8]
    gap = supported_gap(values, unit_text="log10(cm/s)")
    assert gap is not None
    assert gap["gap_lower_value"] == -8.0
    assert gap["gap_upper_value"] == -6.0
    assert gap["gap_decades"] == pytest.approx(2.0)
    assert gap["gap_geometry"] == "canonical_log10"


def test_supported_gap_handles_negative_log_and_natural_log_units():
    negative_log = [2.0, 2.1, 2.2, 4.2, 4.3, 4.4]
    assert supported_gap(negative_log, unit_text="-log10(cm/s)") is not None
    natural_log = [
        value * 2.302585092994046
        for value in (-4.2, -4.1, -4.0, -2.0, -1.9, -1.8)
    ]
    gap = supported_gap(natural_log, unit_text="ln(ng/mL)")
    assert gap is not None
    assert gap["gap_decades"] == pytest.approx(2.0)


@pytest.mark.parametrize("unit", ["log(cm/s)", "log2", "logit_response"])
def test_supported_gap_skips_ambiguous_log_geometry(unit):
    assert supported_gap([-4.2, -4.1, -4.0, -2.0, -1.9, -1.8], unit_text=unit) is None


def test_semantic_projection_excludes_identity_provenance_labels_and_policy():
    columns = semantic_review_columns(
        [
            "canonical_record_id",
            "canonical_smiles",
            "source_id",
            "source_record_id",
            "pmid",
            "outcome_label",
            "retrieval_eligible",
            "measurement_resolution_rule_id",
            "measurement_resolution_origin",
            "measurement_resolution_input_measurement",
            "measurement_resolution_input_unit",
            "canonical_endpoint_name",
            "finite_scalar_value",
            "assay_or_test",
            "species_or_population",
            "qualifying_conditions",
            "support_text",
        ]
    )
    assert set(columns) == {
        "canonical_record_id",
        "measurement_resolution_origin",
        "measurement_resolution_input_measurement",
        "measurement_resolution_input_unit",
        "canonical_endpoint_name",
        "finite_scalar_value",
        "assay_or_test",
        "species_or_population",
        "qualifying_conditions",
        "support_text",
    }


def test_semantic_row_names_the_llm_refined_extraction():
    row = _compact_semantic_row(
        {
            "canonical_record_id": "r1",
            "measurement_text": "5.6 x 10^-3",
            "unit_text": "cm/h",
            "measurement_resolution_origin": "llm",
            "measurement_resolution_input_measurement": "5.6",
            "measurement_resolution_input_unit": "10^-3 cm/h",
            "canonical_measurement_text": "0.0000015555555556",
            "canonical_unit_text": "cm/s",
            "finite_scalar_value": 1.5555555556e-6,
        }
    )
    assert row["llm_refined_extraction"] == {
        "measurement": "5.6",
        "unit": "10^-3 cm/h",
    }
    assert not any(key.startswith("measurement_resolution_") for key in row)


def test_review_contract_and_two_reviewer_consensus():
    response = {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "canonical_record_id": "r1",
                "decision": "drop",
                "reason_code": "wrong_unit_or_scale",
                "reason": "The raw unit denotes a different scale.",
            },
            {
                "canonical_record_id": "r2",
                "decision": "keep",
                "reason_code": "valid_extreme",
                "reason": "The source describes the same quantity.",
            },
        ],
    }
    validated = validate_review_response(response, record_ids={"r1", "r2"})
    candidate = {
        "task_id": "test",
        "bucket_id": "bucket",
        "pair_bucket_key": "hidden-key",
        "rows": [
            {"canonical_record_id": "r1", "finite_scalar_value": 1.0},
            {"canonical_record_id": "r2", "finite_scalar_value": 100.0},
        ],
    }
    review_a = {"response": validated}
    review_b = {"response": validated}
    decisions = consensus_decisions(candidate, review_a, review_b)
    assert [row["consensus_decision"] for row in decisions] == ["drop", "keep"]

    disagreement = json.loads(json.dumps(validated))
    disagreement["rows"][0].update(
        decision="keep",
        reason_code="insufficient_evidence",
        reason="The displayed semantics are insufficient to drop it.",
    )
    decisions = consensus_decisions(candidate, review_a, {"response": disagreement})
    assert decisions[0]["consensus_decision"] == "keep"
    assert decisions[0]["consensus_reason_code"] == "review_disagreement_keep"


def test_prompt_contains_compact_semantics_not_bucket_key():
    prompt = render_review_prompt(
        {
            "task_id": "test",
            "pair_bucket_key": "secret-key",
            "canonical_bucket": {
                "canonical_endpoint_name": "permeability",
                "canonical_unit_text": "cm/s",
                "canonical_pair_fields": {"canonical_species_context": "rat"},
            },
            "distribution": {"record_count": 1, "gap_decades": 2.0},
            "rows": [
                {
                    "canonical_record_id": "r1",
                    "finite_scalar_value": 1.0,
                    "support_text": "measured permeability",
                }
            ],
        }
    )
    assert "secret-key" not in prompt
    assert "measured permeability" in prompt


def test_bucket_uses_short_ids_and_maps_them_back():
    rows = [
        {"canonical_record_id": f"canonical-{index}", "finite_scalar_value": index}
        for index in range(2)
    ]
    candidate = {
        "task_id": "test",
        "canonical_bucket": {},
        "distribution": {},
        "rows": rows,
    }
    prompt = render_review_prompt(candidate)
    assert "canonical-0" not in prompt
    assert '"row_id":"r0001"' in prompt
    response = {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "row_id": f"r{index:04d}",
                "decision": "keep",
                "reason_code": "fits_bucket",
                "reason": "Same scientific quantity.",
            }
            for index in range(1, 3)
        ],
    }
    normalized = _validate_candidate_review(response, candidate)
    assert normalized["rows"][0]["canonical_record_id"] == "canonical-0"
    assert normalized["rows"][-1]["canonical_record_id"] == "canonical-1"


def test_collapse_excludes_only_consensus_drops(tmp_path: Path):
    records = pd.DataFrame(
        [
            _record("r1", 1.0),
            _record("r2", 100.0),
        ]
    )
    buckets = pd.DataFrame(
        [
            _bucket("r1"),
            _bucket("r2"),
        ]
    )
    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "buckets.parquet"
    records.to_parquet(records_path, index=False)
    buckets.to_parquet(buckets_path, index=False)
    metadata_path = tmp_path / "pair_bucket_metadata.json"
    metadata_path.write_text("{}\n", encoding="utf-8")

    pruning = tmp_path / "final_endpoint_pruning_v2"
    pruning.mkdir()
    reviews_path = pruning / REVIEWS_FILENAME
    reviews_path.write_text("{}\n", encoding="utf-8")
    decisions_path = pruning / DECISIONS_FILENAME
    write_parquet(
        decisions_path,
        [
            {
                "canonical_record_id": "r1",
                "consensus_decision": "drop",
                "consensus_reason_code": "wrong_unit_or_scale",
            },
            {
                "canonical_record_id": "r2",
                "consensus_decision": "keep",
                "consensus_reason_code": "fits_bucket",
            },
        ],
    )
    pruning_manifest = {
        "version": VERSION,
        "task_id": "test",
        "inputs": {
            "records": {"sha256": file_sha256(records_path)},
            "pair_bucket_records": {"sha256": file_sha256(buckets_path)},
        },
        "summary": {"candidate_rows": 2, "dropped_rows": 1},
        "files": {
            REVIEWS_FILENAME: file_sha256(reviews_path),
            DECISIONS_FILENAME: file_sha256(decisions_path),
        },
    }
    manifest_path = pruning / "manifest.json"
    manifest_path.write_text(json.dumps(pruning_manifest) + "\n", encoding="utf-8")

    before = (file_sha256(records_path), file_sha256(buckets_path))
    manifest = build_collapsed_record_stage(
        task_id="test",
        records_path=records_path,
        pair_bucket_records_path=buckets_path,
        pair_bucket_metadata_path=metadata_path,
        out_dir=tmp_path / "collapsed",
        final_endpoint_pruning_manifest_path=manifest_path,
    )
    collapsed = pd.read_parquet(tmp_path / "collapsed/records.parquet")
    assert collapsed["finite_scalar_value"].tolist() == [100.0]
    assert manifest["summary"]["final_endpoint_pruning_excluded_records"] == 1
    assert manifest["summary"]["excluded_reason_counts"]["final_endpoint_pruning"] == 1
    assert before == (file_sha256(records_path), file_sha256(buckets_path))


def _record(record_id: str, value: float) -> dict:
    return {
        "canonical_record_id": record_id,
        "source_id": "source",
        "source_record_id": record_id,
        "canonical_smiles": "CCO",
        "retrieval_eligible": True,
        "measurement_kind": "continuous",
        "finite_scalar_value": value,
        "variation_value": None,
        "canonical_endpoint_name": "test_endpoint",
        "canonical_measurement_text": str(value),
        "canonical_unit_text": "mg/L",
        "support_text": "same quantity",
        "group_id": "Observed.indirect",
    }


def _bucket(record_id: str) -> dict:
    return {
        "canonical_record_id": record_id,
        "pair_bucket_key": "bucket",
        "canonical_unit_text": "mg/L",
        "canonical_pair_fields_json": "{}",
        "assay_transfer_eligible": True,
        "assay_transfer_ineligibility_reason": None,
    }
