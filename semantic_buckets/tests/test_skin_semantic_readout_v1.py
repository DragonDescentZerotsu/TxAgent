import json

from semantic_buckets.skin_semantic_readout_v1 import _identities


def test_categorical_scale_splits_readout_but_not_semantic_bucket():
    base = ["direct_skin_reaction", "sensitization_outcome", "unitless", "human", "human"]
    first = _identities(2, json.dumps([*base, "binary_outcome"]))
    second = _identities(2, json.dumps([*base, "severity_grade"]))

    assert first[1] == second[1]
    assert first[2] != second[2]
    assert first[3] != second[3]


def test_level_and_assay_context_are_semantic_identity():
    key = json.dumps([
        "sensitization_aop", "keratinocyte_activation", "ratio",
        "key_event_2", "in_vitro", "human",
    ])
    changed = json.dumps([
        "sensitization_aop", "keratinocyte_activation", "ratio",
        "key_event_2", "in_vitro", "mouse",
    ])

    assert _identities(2, key)[1] != _identities(3, key)[1]
    assert _identities(3, key)[1] != _identities(3, changed)[1]
