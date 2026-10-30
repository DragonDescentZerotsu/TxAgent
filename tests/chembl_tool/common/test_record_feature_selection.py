"""Coverage selector invariants and parity with apricot's public API."""
import copy

import numpy as np
import pytest

from tools.chembl_tool.common.record_feature_selection import (
    feature_matrix, record_features, select_feature_cards, selection_contract,
)
from tools.chembl_tool.common.progressive_assay_reasoning import (
    append_evidence, select_initial_evidence, select_progressive_delta,
)


def card(cid, endpoint="liver tumor", result="negative", **extra):
    return {"card_id": cid, "endpoint": endpoint, "reported_value": result,
            "species": "mouse", "support_text": "Complete original record.", **extra}


def test_nonnegative_field_scoped_features_exclude_labels_and_bookkeeping():
    c = card("a", qualifying_conditions="needs_more_context=False | dose_unit=mg/kg")
    f = record_features(c)
    assert f and all(np.isfinite(v) and v > 0 for v in f.values())
    changed = {**c, "Y": 1, "evidence_direction": "positive", "card_id": "other",
               "_assay_key": "other-study", "qualifying_conditions": "needs_more_context=True | dose_unit=mg/kg"}
    assert f == record_features(changed)
    assert record_features({"endpoint": "unknown", "reported_value": "unspecified (2)"}) == {}
    assert record_features(card("a", result="increased")) != record_features(card("a", result="not increased"))
    assert "reported_value/text:none" in record_features(card("a", result="none of the animals had tumors"))


def test_each_populated_field_has_equal_initial_sqrt_mass():
    f = record_features(card("a", assay_context="a much longer experimental description with many observations"))
    for field in {k.split("/")[0] for k in f}:
        assert sum(np.sqrt(v) for k, v in f.items() if k.startswith(field + "/")) == pytest.approx(1)


def test_history_is_coverage_only_and_vocabulary_uses_candidates():
    candidates = [card("a"), card("b", "thyroid tumor", "positive")]
    history = [card("past"), {"endpoint": "futureexclusive"}]
    X, initial, names = feature_matrix(candidates, history)
    assert X.shape[0] == 2 and initial.shape == (1, X.shape[1])
    assert not any("futureexclusive" in n for n in names)
    assert initial.sum() > 0


@pytest.fixture
def apricot():
    module = pytest.importorskip("apricot")
    selection_contract()
    return module


def test_cached_kernel_matches_public_apricot_ranking_and_gains(apricot):
    candidates = [card("a"), card("b"), card("c", "thyroid tumor", "positive"),
                  card("d", "bile flow", "no change")]
    history = [card("past")]
    X, initial, _ = feature_matrix(candidates, history)
    reference = apricot.FeatureBasedSelection(3, concave_func="sqrt", optimizer="naive",
                                              initial_subset=initial, random_state=0).fit(X)
    chosen, audit = select_feature_cards(candidates, 3, covered_cards=history)
    assert [c["card_id"] for c in chosen] == [candidates[i]["card_id"] for i in reference.ranking]
    np.testing.assert_allclose(audit["gains"], reference.gains, atol=1e-12)
    assert chosen[0]["card_id"] != "a"
    assert all(c["card_id"] != "past" for c in chosen)


def test_empty_and_insufficient_pools_are_explicit(apricot):
    assert select_feature_cards([], 5)[0] == []
    chosen, audit = select_feature_cards([{"card_id": "a"}, {"card_id": "b"}], 5)
    assert [c["card_id"] for c in chosen] == ["a", "b"]
    assert audit["gains"] == [0, 0] and audit["n_featureless_candidates"] == 2


def test_progressive_preserves_molecules_caps_and_no_backlog(apricot):
    def analog(aid, similarity, cards):
        return {"analog_id": aid, "similarity": similarity,
                "cards": {c["card_id"]: c for c in cards}}
    previous = {"active": analog("active", .9, [card("a"), card("skipped")])}
    active, _ = select_initial_evidence(previous, card_limit=1, card_selection_policy="feature_based.v1")
    frozen = copy.deepcopy(active)
    current = {"active": analog("active", .9, [card("a"), card("skipped"), card("new_a"), card("new_b", "thyroid tumor")]),
               "near": analog("near", .8, [card("c"), card("d", "bile flow")]),
               "far": analog("far", .7, [card("far")])}
    baseline = select_progressive_delta(previous, current, active, level=2, card_limit=1, combined_molecule_limit=2)
    new, aug, audit = select_progressive_delta(previous, current, active, level=2, card_limit=1,
                                              combined_molecule_limit=2, card_selection_policy="feature_based.v1")
    assert set(new) == set(baseline[0]) == {"near"}
    assert set(aug) == set(baseline[1]) == {"active"}
    assert all(len(a["cards"]) == 1 for a in [*new.values(), *aug.values()])
    merged = append_evidence(active, new, aug)
    assert active == frozen and "a" in merged["active"]["cards"]
    assert "skipped" not in merged["active"]["cards"]
    assert audit["feature_selection"][0]["n_covered_cards"] == 1
    assert audit["feature_selection"][1]["n_covered_cards"] == 2
    assert all("features" not in c for a in merged.values() for c in a["cards"].values())


def test_condition_buckets_remain_hard_priority(apricot):
    pool = {"a": {"analog_id": "a", "similarity": .8, "cards": {
        "exact": card("exact"), "other": card("other", "different tumor", "positive")}}}
    chosen, _ = select_initial_evidence(pool, card_limit=1, query_condition="human",
                                       card_condition_groups={"exact": ["human"], "other": ["mouse"]},
                                       card_selection_policy="feature_based.v1")
    assert set(chosen["a"]["cards"]) == {"exact"}


def test_shared_matrix_and_kernel_are_safe_for_parallel_query_selection(apricot):
    from concurrent.futures import ThreadPoolExecutor
    pool = [card("a"), card("b", "thyroid tumor", "positive"), card("c", "bile flow")]
    history = [card("past")]
    expected, _ = select_feature_cards(pool, 2, covered_cards=history)
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: select_feature_cards(pool, 2, covered_cards=history)[0], range(16)))
    assert all(result == expected for result in results)
