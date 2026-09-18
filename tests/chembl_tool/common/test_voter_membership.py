import pyarrow.parquet as pq

from data.processing.gold_labels.voter_membership import (
    materialize_gold_label_record_index,
    materialize_voter_membership,
    write_gold_label_record_index,
    write_voter_membership,
)


def _vote(index: int, label: int = 1) -> dict:
    return {
        "source_row_uid": f"sr_{index:032x}",
        "source_id": "stage1:test",
        "source_record_id": f"row:{index}",
        "Y": label,
        "label_method": "direct",
    }


def _aggregate(parent: str, group: str) -> dict:
    return {
        "drug": "CCO",
        "molecule_identity_key": parent,
        "condition_group": group,
        "condition_atoms": [group],
    }


def test_membership_parquet_does_not_cap_large_vote_groups(tmp_path) -> None:
    key = ("PARENT", "disease=asthma")
    published = {
        **_aggregate(*key),
        "Y": 1,
        "label_decision": "unanimous",
        "benchmark_row_id": "ROW_1",
        "split": "train",
    }
    rows = materialize_voter_membership(
        task_name="Synthetic",
        vote_groups={key: [_vote(index) for index in range(61)]},
        published_aggregates=[published],
    )
    path = tmp_path / "voter_membership.parquet"
    manifest = write_voter_membership(
        path,
        rows,
        stage1_sha256="a" * 64,
        provenance={"source": "stage1/records.parquet"},
    )

    table = pq.read_table(path)
    assert table.num_rows == 61
    assert len(set(table.column("source_row_uid").to_pylist())) == 61
    assert set(table.column("aggregate_status").to_pylist()) == {"published"}
    assert manifest["n_rows"] == 61
    assert manifest["stage1_sha256"] == "a" * 64
    assert (tmp_path / "voter_membership.manifest.json").is_file()


def test_rejected_parent_votes_remain_in_membership(tmp_path) -> None:
    key = ("REJECTED_PARENT", "disease=asthma")
    rejected = {
        **_aggregate(*key),
        "drop_reason": "parent_condition_label_tie",
    }
    rows = materialize_voter_membership(
        task_name="Synthetic",
        vote_groups={key: [_vote(1, 0), _vote(2, 1)]},
        published_aggregates=[],
        rejected_parent_aggregates=[rejected],
    )
    path = tmp_path / "voter_membership.parquet"
    write_voter_membership(path, rows)
    stored = pq.read_table(path).to_pylist()

    assert len(stored) == 2
    assert {row["aggregate_status"] for row in stored} == {"parent_label_tie"}
    assert {row["vote_label"] for row in stored} == {0, 1}
    assert all(
        row["benchmark_row_id"] is None and row["split"] is None for row in stored
    )


def test_gold_label_record_index_reproduces_physical_votes(tmp_path) -> None:
    rows = [
        {
            "aggregate_status": "published",
            "benchmark_row_id": "gold-1",
            "source_row_uid": "sr_" + suffix * 32,
            "source_id": "source",
            "source_record_id": suffix,
            "molecule_identity_key": "parent",
            "condition_group": "condition",
            "split": "train",
            "vote_label": label,
        }
        for suffix, label in (("a", 1), ("b", 1), ("c", 0))
    ]
    cards = [{
        "benchmark_row_id": "gold-1",
        "molecule_identity_key": "parent",
        "condition_group": "condition",
        "split": "train",
        "source_record_count": 3,
        "label_counts": {"0": 1, "1": 2},
    }]
    index = materialize_gold_label_record_index(rows, cards)
    assert index[0]["source_row_uids"] == ["sr_" + suffix * 32 for suffix in "abc"]
    assert index[0]["voter_mean"] == 2 / 3
    manifest = write_gold_label_record_index(
        tmp_path / "gold_label_record_index.parquet",
        index,
        membership_sha256="d" * 64,
    )
    assert manifest["n_gold_labels"] == 1
    assert manifest["n_physical_voters"] == 3
