import json

from semantic_buckets import source_local_overmerge_pass4_v1 as run
from semantic_buckets import source_local_overmerge_reconcile_v1 as ledger


def _group(index):
    return {"task": "dili", "parent_id": "parent", "level": "L2", "source_id": "source",
            "group_id": f"g{index}", "label": f"assay {index}",
            "values": [f"assay value {index}"], "rationale": "previous review"}


def test_pass4_uses_luna_flex_schedule_and_preserves_scope():
    schedule = run._schedule([_group(index) for index in range(200)])
    assert len(schedule) == 2
    assert sorted(len(row["items"]) for row in schedule) == [100, 100]
    assert {row["scope"]["provider_family"] for row in schedule} == {"luna"}
    assert {row["scope"]["parent_id"] for row in schedule} == {"parent"}


def test_pass4_decisions_map_group_members_once(tmp_path):
    groups = [_group(0), _group(1)]
    schedule = run._schedule(groups)
    connection = ledger._database(tmp_path / "requests.sqlite3")
    ledger._insert(connection, schedule)
    connection.execute("UPDATE requests SET status='complete', response_json=?", (
        json.dumps({"merge_sets": [{"member_ids": [0, 1], "label": "same assay",
                                   "rationale": "same measured endpoint"}]}),))
    connection.commit()
    merges, replacements = run._decisions(groups, connection, schedule)
    assert len(merges) == 1
    assert set(replacements) == {"g0", "g1"}
    assert set(merges[0]["values"]) == {"assay value 0", "assay value 1"}
