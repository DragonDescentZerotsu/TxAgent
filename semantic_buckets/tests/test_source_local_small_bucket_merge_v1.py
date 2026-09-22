import json

import pytest

from semantic_buckets.source_local_small_bucket_merge_v1 import _validate


def test_merge_response_accepts_disjoint_id_lists():
    assert _validate(json.dumps([[0, 3], [1, 2, 4]]), 5) == [[0, 3], [1, 2, 4]]


@pytest.mark.parametrize("value", ["{}", "[[0]]", "[[0,2],[2,3]]", "[[0,5]]", "[[true,1]]"])
def test_merge_response_rejects_invalid_shapes(value):
    with pytest.raises(ValueError):
        _validate(value, 5)
