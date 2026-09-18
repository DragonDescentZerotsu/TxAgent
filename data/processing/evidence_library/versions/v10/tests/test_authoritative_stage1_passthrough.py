import pyarrow as pa
import pyarrow.parquet as pq
from data.processing.evidence_library.shared.v2.record_deduplication import (
    _decorate,
    build_deduplicated_record_stage,
)


def test_authoritative_stage1_survivors_are_not_deduplicated_again(tmp_path):
    records = [
        {
            "canonical_record_id": record_id,
            "source_row_uid": uid,
            "source_id": "oral_exposure",
            "canonical_smiles": "CCO",
            "support_text": support,
            "pmid": "1",
            "canonical_measurement_text": "50",
            "canonical_unit_text": "%",
        }
        for record_id, uid, support in (
            ("record-a", "sr_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "same claim a"),
            ("record-b", "sr_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "same claim b"),
        )
    ]
    sidecars = [
        {
            "canonical_record_id": record["canonical_record_id"],
            "pair_bucket_key": '["oral"]',
            "canonical_pair_fields_json": '{"endpoint":"oral"}',
            "assay_transfer_eligible": True,
            "assay_transfer_ineligibility_reason": None,
        }
        for record in records
    ]
    records_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "sidecars.parquet"
    output = tmp_path / "output"
    pq.write_table(pa.Table.from_pylist(records), records_path)
    pq.write_table(pa.Table.from_pylist(sidecars), sidecar_path)

    manifest = build_deduplicated_record_stage(
        task_id="bioavailability_ma",
        records_path=records_path,
        pair_bucket_records_path=sidecar_path,
        out_dir=output,
        collapse_duplicates=False,
    )

    assert manifest["contract"]["mode"] == "authoritative_stage1_passthrough"
    assert manifest["contract"]["row_deletion_performed"] is False
    assert manifest["summary"]["retained_records"] == 2
    assert manifest["summary"]["duplicates_removed"] == 0
    assert pq.read_table(output / "records.parquet").num_rows == 2
    assert pq.read_table(output / "duplicates.parquet").num_rows == 0


def test_gold_condition_does_not_change_pair_bucket_identity():
    sidecar = {
        "pair_bucket_key": '["oral_exposure","bioavailability","%","human"]',
        "canonical_pair_fields_json": '{"canonical_species_context":"human"}',
        "assay_transfer_eligible": True,
        "assay_transfer_ineligibility_reason": None,
    }
    rows = [
        _decorate(
            {"canonical_record_id": f"record-{index}", "source_row_number": index},
            sidecar,
            {
                "retrieval_source_id": "direct_vote",
                "condition_group": condition,
                "condition_key_status": status,
                "condition_atoms": [],
                "condition_scope": "reported",
            },
        )
        for index, condition, status in (
            (1, "prandial_state=fasted", "reviewed"),
            (2, "unresolved:claim", "unresolved"),
        )
    ]

    assert {row["pair_bucket_key"] for row in rows} == {sidecar["pair_bucket_key"]}
    assert {row["canonical_pair_fields_json"] for row in rows} == {
        sidecar["canonical_pair_fields_json"]
    }
    assert "canonical_direct_condition_group" not in rows[0]
