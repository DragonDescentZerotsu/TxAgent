from optimization.semantic_buckets import storage as semantic_storage
import json
import pandas as pd
from optimization.semantic_buckets import bucket_refinement as review

"Behavioral checks for disjoint merge passes."


def test_scope_rows_assign_mixed_endpoint_family_once():
    groups = {
        "a": {
            "level": "L2",
            "source_id": "s",
            "endpoints": ["x", "y"],
            "values": ["measured activity"],
            "record_count": 3,
        },
        "b": {
            "level": "L2",
            "source_id": "s",
            "endpoints": ["x"],
            "values": ["other activity"],
            "record_count": 99,
        },
        "c": {
            "level": "L2",
            "source_id": "s",
            "endpoints": ["x"],
            "values": ["large activity"],
            "record_count": 100,
        },
    }
    scopes = review._scope_rows(groups)
    assert set(scopes.semantic_bucket_id) == {"a", "b"}
    assert (
        scopes.set_index("semantic_bucket_id").loc["a", "endpoint"]
        == "__mixed_endpoint__"
    )


def test_provisional_groups_do_not_take_transitive_closure(tmp_path):
    pd.DataFrame(
        [
            {
                "semantic_bucket_id": bucket,
                "level": "L2",
                "source_id": "s",
                "endpoint": "x",
                "label": bucket,
                "record_count": 2,
            }
            for bucket in ("a", "b", "c")
        ]
    ).to_csv(tmp_path / "small_family_scopes.tsv", sep="\t", index=False)
    db = semantic_storage.review_database(tmp_path / "requests.sqlite3")
    semantic_storage.insert_reviews(
        db,
        [
            {
                "request_id": "r1",
                "task": "dili",
                "pass_number": 1,
                "scope": {},
                "items": ["a", "b", "c"],
                "prompt": "p",
            }
        ],
    )
    db.execute(
        "UPDATE requests SET status='complete', response_json=? WHERE request_id='r1'",
        (
            json.dumps(
                {
                    "merge_sets": [
                        {
                            "member_ids": [0, 1],
                            "label": "same activity",
                            "rationale": "measured same activity",
                        }
                    ]
                }
            ),
        ),
    )
    db.commit()
    groups = review._provisional(tmp_path, db)
    db.close()
    assert {tuple(row["member_bucket_ids"]) for row in groups} == {("a", "b"), ("c",)}
    assert (
        next((row for row in groups if row["member_bucket_ids"] == ["c"]))["label"]
        == "c"
    )
