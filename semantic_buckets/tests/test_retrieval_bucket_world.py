import pytest

from semantic_buckets import build_retrieval_bucket_world as world


def test_bucket_rows_unions_splits_and_preserves_level() -> None:
    mapping = {
        "u1": ("L2", "b1"),
        "u2": ("L2", "b1"),
        "u3": ("L2", "b2"),
    }

    rows = world._bucket_rows("bbb_martins", "L2", {"u1", "u2"}, {"u2", "u3"}, mapping)

    assert rows == [
        {"task": "bbb_martins", "level": "L2", "semantic_bucket_id": "b1",
         "record_count": 2, "valid_record_count": 2, "test_record_count": 1},
        {"task": "bbb_martins", "level": "L2", "semantic_bucket_id": "b2",
         "record_count": 1, "valid_record_count": 0, "test_record_count": 1},
    ]


def test_bucket_rows_rejects_unmapped_or_wrong_level_uids() -> None:
    with pytest.raises(ValueError, match="unmapped retrieval UIDs"):
        world._bucket_rows("bbb_martins", "L2", {"missing"}, set(), {})
    with pytest.raises(ValueError, match="level mismatch"):
        world._bucket_rows("bbb_martins", "L2", {"u1"}, set(), {"u1": ("L3", "b1")})
