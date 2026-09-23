from semantic_buckets.overmerge_pass2_batches import batch_groups


def test_pass2_rebatches_across_pass1_boundaries() -> None:
    groups = [
        {"parent_id": "p", "group_id": "a", "label": "AK release", "values": ["adenylate kinase release"], "pass1_batch": 0},
        {"parent_id": "p", "group_id": "z", "label": "ALT release", "values": ["alanine transaminase release"], "pass1_batch": 0},
        {"parent_id": "p", "group_id": "b", "label": "AK release cytotoxicity", "values": ["AK release cytotoxicity"], "pass1_batch": 1},
        {"parent_id": "q", "group_id": "c", "label": "AK release", "values": ["AK release"], "pass1_batch": 1},
    ]
    batches = batch_groups(groups, batch_size=2)
    assert [[group["group_id"] for group in batch] for batch in batches] == [["a", "b"], ["z"], ["c"]]
    assert batch_groups(list(reversed(groups)), batch_size=2) == batches


def test_full_name_and_acronym_share_a_review_batch() -> None:
    groups = [
        {"parent_id": "p", "group_id": "long", "label": "adenylate kinase release", "values": ["adenylate kinase release"]},
        {"parent_id": "p", "group_id": "unrelated", "label": "cholesterol secretion", "values": ["cholesterol secretion"]},
        {"parent_id": "p", "group_id": "short", "label": "AK release", "values": ["AK release"]},
    ]
    batches = batch_groups(groups, batch_size=2)
    assert [set(group["group_id"] for group in batch) for batch in batches] == [
        {"long", "short"}, {"unrelated"}
    ]
