from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing import starling_source_row_identity as identity
from data.processing.evidence_library.shared.v2.record_deduplication import (
    build_deduplicated_record_stage,
)


def test_bootstrap_is_global_and_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "starling"
    for task in ("carcinogens", "dili"):
        path = raw / task / f"{task}_base" / "extractions.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.table({"extraction_id": ["same", "same"]}), path)
    ledger = raw / "source_row_uid_ledger"
    monkeypatch.setattr(identity, "RAW_ROOT", raw)
    monkeypatch.setattr(identity, "LOCK_PATH", raw / ".source-row-uid.lock")
    monkeypatch.setattr(identity, "INCOMPLETE_PATH", raw / ".source-row-uid-incomplete.json")

    result = identity.sync(raw_root=raw, ledger_root=ledger, bootstrap=True)
    assert result["active_rows"] == 4
    assert identity.verify(raw_root=raw, ledger_root=ledger)["raw_rows"] == 4
    uids = []
    for path in identity.authoritative_sources(raw):
        uids.extend(pq.read_table(path.path, columns=[identity.UID_FIELD])[identity.UID_FIELD].to_pylist())
    assert len(uids) == len(set(uids)) == 4
    for spec in identity.authoritative_sources(raw):
        table = pq.read_table(spec.path)
        assert table.drop([identity.UID_FIELD]).equals(
            pa.table({"extraction_id": ["same", "same"]})
        )
    identity.publish_compressed(raw_root=raw)
    assert identity.verify_compressed(raw_root=raw) == {"sources": 2, "parts": 2}

    table = pq.read_table(identity.authoritative_sources(raw)[1].path)
    table = table.set_column(table.schema.get_field_index(identity.UID_FIELD), identity.UID_FIELD, pa.array([uids[0], uids[-1]]))
    pq.write_table(table, identity.authoritative_sources(raw)[1].path)
    with pytest.raises(ValueError, match="duplicate UID"):
        identity.verify(raw_root=raw, ledger_root=ledger)


def test_pair_bucket_deduplication_preserves_source_uids(tmp_path: Path) -> None:
    records = [
        {
            "canonical_record_id": f"record-{index}",
            "source_row_uid": f"sr_{index:032x}",
            "source_id": "source",
            "canonical_smiles": "CC",
            "retrieval_source_id": "indirect",
            "support_text": "same supported statement",
            "measurement_text": "1",
            "canonical_measurement_text": "1",
            "canonical_unit_text": "unit",
            "pair_bucket_key": "bucket",
            "canonical_pair_fields_json": "{}",
        }
        for index in range(2)
    ]
    sidecars = [
        {
            "canonical_record_id": f"record-{index}",
            "source_id": "source",
            "pair_bucket_key": "bucket",
            "canonical_pair_fields_json": "{}",
            "assay_transfer_eligible": True,
            "bucket_eligible": True,
        }
        for index in range(2)
    ]
    records_path = tmp_path / "records.parquet"
    sidecars_path = tmp_path / "sidecars.parquet"
    pq.write_table(pa.Table.from_pylist(records), records_path)
    pq.write_table(pa.Table.from_pylist(sidecars), sidecars_path)

    build_deduplicated_record_stage(
        task_id="test",
        records_path=records_path,
        pair_bucket_records_path=sidecars_path,
        out_dir=tmp_path / "output",
    )

    retained = pq.read_table(tmp_path / "output/pair_bucket_records.parquet")
    duplicates = pq.read_table(tmp_path / "output/duplicates.parquet")
    assert retained["source_row_uid"].to_pylist() == ["sr_" + "0" * 32]
    assert duplicates["source_row_uid"].to_pylist() == ["sr_" + "0" * 31 + "1"]
    assert duplicates["retained_source_row_uid"].to_pylist() == [
        "sr_" + "0" * 32
    ]
