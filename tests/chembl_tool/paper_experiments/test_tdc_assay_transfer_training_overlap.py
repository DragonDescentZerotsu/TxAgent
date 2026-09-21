from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.paper_experiments.audit_tdc_assay_transfer_training_overlap import (
    _member_rows,
    _scan_training,
    _summary_rows,
)


def test_training_scan_tracks_exposure_and_tdc_gold_labels(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table({
            "query_parent_id": ["p1", "p1", "other"],
            "retrieval_parent_id": ["other", "p1", "p2"],
            "source_id": ["s1", "s2", "s3"],
            "query_scaffold": ["scaf1", "scaf1", ""],
            "retrieval_scaffold": ["scaf2", "scaf1", "scaf3"],
        }),
        train_path,
    )
    profile = _scan_training(train_path, {"p1", "p2"}, include_scaffolds=True)
    queries = [
        {
            "benchmark_row_id": "q1",
            "molecule_identity_key": "p1",
            "bemis_murcko_scaffold": "scaf1",
            "Y": 1,
        },
        {
            "benchmark_row_id": "q2",
            "molecule_identity_key": "p2",
            "bemis_murcko_scaffold": "scaf3",
            "Y": 0,
        },
    ]

    members = _member_rows("task", "test", "direct", "L1", queries, profile)
    overlap = _summary_rows("task", "test", "direct", "L1", queries, profile, members)

    assert profile.details["p1"].query_occurrences == 2
    assert profile.details["p1"].retrieval_occurrences == 1
    assert [row["overlap_query_count"] for row in overlap] == [2, 2]
    assert overlap[0]["overlap_gold_0_count"] == 1
    assert overlap[0]["overlap_gold_1_count"] == 1
