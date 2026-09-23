import json

import pytest

from semantic_buckets.source_local_overmerge_reconcile_v1 import _resolved, _validate_response


def test_merge_decisions_preserve_omitted_singletons():
    response = _validate_response(json.dumps({"merge_sets": [
        {"member_ids": [0, 2], "label": "same readout", "rationale": "synonymous values"}
    ]}), 3)
    assert _resolved(["a", "b", "c"], response) == [
        (["a", "c"], "same readout", "synonymous values"),
        (["b"], "", "Unmerged singleton"),
    ]


@pytest.mark.parametrize("groups", [
    [{"member_ids": [0, 0], "label": "x", "rationale": "y"}],
    [{"member_ids": [0, 2], "label": "x", "rationale": "y"}],
    [{"member_ids": [0, 1], "label": "x", "rationale": ""}],
])
def test_merge_decisions_reject_invalid_groups(groups):
    with pytest.raises(ValueError):
        _validate_response(json.dumps({"merge_sets": groups}), 2)
