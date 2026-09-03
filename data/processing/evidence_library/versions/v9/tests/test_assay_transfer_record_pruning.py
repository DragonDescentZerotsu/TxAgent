import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.versions.v9.assay_transfer_record_pruning import (
    SCHEMA_VERSION,
    _load_manual_ineligibility,
    _candidate_buckets,
    held_out_quartile_tail_outliers,
    tail_outliers,
    validate_review_response,
)


def test_manual_ineligibility_is_identity_bound(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    mapping_path = tmp_path / "manual.json"
    pq.write_table(
        pa.Table.from_pylist(
            [{"canonical_record_id": "a" * 64, "source_row_uid": "source-row-1"}]
        ),
        records_path,
    )
    payload = {
        "version": "bbb_assay_transfer_manual_ineligibility.v1",
        "task_id": "bbb_martins",
        "records": [
            {
                "canonical_record_id": "a" * 64,
                "source_row_uid": "source-row-1",
                "reason_code": "valid_source_out_of_assay_transfer_domain",
                "rationale": "A source-valid result can still be outside the comparable transfer domain.",
            }
        ],
    }
    mapping_path.write_text(json.dumps(payload), encoding="utf-8")

    decisions, manifest = _load_manual_ineligibility(
        mapping_path, task_id="bbb_martins", records_path=records_path
    )

    assert decisions == {"a" * 64: "valid_source_out_of_assay_transfer_domain"}
    assert manifest["records"] == 1
    payload["records"][0]["source_row_uid"] = "wrong-source-row"
    mapping_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity drift"):
        _load_manual_ineligibility(
            mapping_path, task_id="bbb_martins", records_path=records_path
        )


def test_held_out_quartile_finds_a_masked_tied_tail() -> None:
    values = [1.0] * 15 + [1000.0] * 5

    assert tail_outliers(values, unit_text="log(response)") == []
    tails = held_out_quartile_tail_outliers(values, unit_text="log(response)")

    assert [tail["record_index"] for tail in tails] == list(range(15, 20))
    assert {tail["held_out_tail_side"] for tail in tails} == {"upper"}
    assert all(tail["held_out_tail_zero_reference_sd"] for tail in tails)


def test_bbb_candidates_union_existing_and_held_out_triggers(tmp_path: Path) -> None:
    values = [1.0] * 15 + [1000.0] * 5
    records = [
        {
            "canonical_record_id": f"r{index:02d}",
            "canonical_smiles": f"parent-{index:02d}",
            "measurement_kind": "continuous",
            "finite_scalar_value": value,
            "assay_transfer_pretransform_scalar_value": value,
            "assay_transfer_transform_id": "raw.v1",
        }
        for index, value in enumerate(values)
    ]
    buckets = [
        {
            "canonical_record_id": row["canonical_record_id"],
            "pair_bucket_key": '["source","endpoint","log(response)"]',
            "canonical_pair_fields_json": "{}",
            "canonical_endpoint_name": "endpoint",
            "canonical_unit_text": "log(response)",
            "assay_transfer_eligible": True,
        }
        for row in records
    ]
    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "buckets.parquet"
    pq.write_table(pa.Table.from_pylist(records), records_path)
    pq.write_table(pa.Table.from_pylist(buckets), buckets_path)

    candidates, summary = _candidate_buckets("bbb_martins", records_path, buckets_path)

    assert candidates[0]["target_record_ids"] == [
        f"r{index:02d}" for index in range(15, 20)
    ]
    assert summary["leave_one_out_candidate_rows"] == 0
    assert summary["held_out_quartile_candidate_rows"] == 5
    assert summary["tail_candidate_rows"] == 5
    assert candidates[0]["distribution"]["observed_median"] == 1.0
    assert candidates[0]["distribution"]["observed_maximum"] == 1000.0


def test_review_accepts_scientifically_implausible_pruning() -> None:
    response = {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "row_id": "r0001",
                "decision": "prune_record",
                "reason_code": "scientifically_implausible_or_out_of_domain",
                "reason": "Magnitude is incompatible with the assay's useful range.",
            }
        ],
    }

    validated = validate_review_response(
        response, record_ids={"r0001"}, id_field="row_id"
    )

    assert validated == response
