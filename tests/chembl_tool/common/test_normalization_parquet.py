import pyarrow.parquet as pq

from tools.chembl_tool.common.starling.normalization import audit


def test_parquet_writer_streams_union_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PARQUET_BATCH_ROWS", 2)
    path = tmp_path / "records.parquet"
    rows = [
        {"id": "a", "value": 1},
        {"id": "b", "value": 2},
        {"id": "c", "value": 3.5, "late_field": "kept"},
    ]

    audit.write_parquet(path, rows)

    assert pq.read_table(path).to_pylist() == [
        {"id": "a", "value": 1.0, "late_field": None},
        {"id": "b", "value": 2.0, "late_field": None},
        {"id": "c", "value": 3.5, "late_field": "kept"},
    ]


def test_parquet_writer_accepts_empty_sidecar(tmp_path):
    path = tmp_path / "empty.parquet"
    audit.write_parquet(path, [])
    assert pq.read_table(path).num_rows == 0
