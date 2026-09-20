import optimization.grid_search as grid_search
from optimization.grid_search import (
    COMPONENTS,
    anchor_names,
    fixed_gated_anchor_names,
    fixed_gated_grid_profiles,
    gated_grid_profiles,
    grid_profiles,
    normalized_gated_grid_profiles,
    screen_by_uid_overlap,
    screen_profiles,
)
from optimization.select_records import Profile


def test_test_query_set_reads_test_ranked_cache(monkeypatch) -> None:
    observed = {}

    def fake_load(release_index, *, task, subset, levels, queries):
        observed["subset"] = subset
        return ({level: {query_id: [] for query_id in queries} for level in levels}, {}, {})

    monkeypatch.setattr(grid_search, "load_ranked_universe", fake_load)
    monkeypatch.setattr(grid_search, "_semantic_rows", lambda task: ({}, {}))

    query_ids, _, audit = grid_search._task_inputs("bbb_martins", query_set="test")

    assert len(query_ids) == 393
    assert observed["subset"] == "test"
    assert audit["evaluation_subset"] == "test"


def test_grid_has_384_profiles_and_nine_requested_anchors() -> None:
    profiles = grid_profiles()

    assert len(profiles) == 384
    assert len({profile.name for profile in profiles}) == 384
    assert len(anchor_names()) == 9
    assert set(anchor_names()).issubset({profile.name for profile in profiles})


def test_gated_grid_has_324_profiles() -> None:
    profiles = gated_grid_profiles()

    assert len(profiles) == 324
    assert len({profile.name for profile in profiles}) == 324
    assert {profile.assay_lambda for profile in profiles} == {0, 0.1, 0.25}
    assert {profile.gated_assay_lambda for profile in profiles} == {0, 0.1, 0.25, 0.5}


def test_fixed_gated_grid_has_81_profiles_and_six_anchors() -> None:
    profiles = fixed_gated_grid_profiles()

    assert len(profiles) == 81
    assert len({profile.name for profile in profiles}) == 81
    assert all(profile.gated_assay_lambda == 1 for profile in profiles)
    assert {
        value
        for profile in profiles
        for value in (
            profile.assay_lambda,
            profile.molecular_lambda,
            profile.semantic_relevance_lambda,
            profile.semantic_diversity_lambda,
        )
    } == {0, 0.25, 0.5}
    assert len(fixed_gated_anchor_names()) == 6
    assert set(fixed_gated_anchor_names()).issubset({profile.name for profile in profiles})


def test_normalized_gated_grid_has_36_crossed_profiles_and_three_controls() -> None:
    profiles = normalized_gated_grid_profiles()

    assert len(profiles) == 39
    assert len({profile.name for profile in profiles}) == 39
    controls = [
        profile for profile in profiles
        if profile.molecular_lambda == 0
        and profile.semantic_relevance_lambda == 0
        and profile.semantic_diversity_lambda == 0
    ]
    assert len(controls) == 3
    assert {profile.gated_assay_lambda for profile in profiles} == {0.75, 1.0, 1.25}
    assert {profile.molecular_lambda for profile in profiles if profile not in controls} == {
        0.25, 0.5, 0.75,
    }
    assert {
        profile.semantic_relevance_lambda for profile in profiles if profile not in controls
    } == {0.05, 0.1}
    assert {
        profile.semantic_diversity_lambda for profile in profiles if profile not in controls
    } == {0.05, 0.1}

    high_semantic = normalized_gated_grid_profiles((0.25, 0.5))
    assert len(high_semantic) == 39
    assert {
        profile.semantic_relevance_lambda
        for profile in high_semantic if profile.molecular_lambda
    } == {0.25, 0.5}
    assert {
        profile.semantic_diversity_lambda
        for profile in high_semantic if profile.molecular_lambda
    } == {0.25, 0.5}


def test_screen_is_deterministic_keeps_anchors_and_collapses_aliases() -> None:
    profiles = grid_profiles()
    signatures = {profile.name: profile.name for profile in profiles}
    duplicate = profiles[-1].name
    signatures[duplicate] = signatures[profiles[0].name]
    components = {
        profile.name: {
            component: ((index + offset) % len(profiles)) / len(profiles)
            for offset, component in enumerate(COMPONENTS)
        }
        for index, profile in enumerate(profiles)
    }

    selected, aliases, ranks = screen_profiles(profiles, signatures, components)
    repeated, _, _ = screen_profiles(profiles, signatures, components)

    assert selected == repeated
    assert len(selected) == 32
    assert aliases[duplicate] == profiles[0].name
    assert set(aliases[name] for name in anchor_names()).issubset(selected)
    assert all(ranks[name] >= 1 for name in selected)


def test_overlap_screen_prefers_morgan_then_assay_within_strata() -> None:
    profiles = [
        Profile("baseline", assay_lambda=0),
        Profile("morgan_low", assay_lambda=0),
        Profile("morgan_high", assay_lambda=0),
        Profile("assay_base", assay_lambda=0.25),
        Profile("assay_low", assay_lambda=0.25),
        Profile("assay_high", assay_lambda=0.25),
        Profile("assay_050_a", assay_lambda=0.5),
        Profile("assay_050_b", assay_lambda=0.5),
    ]
    uid_sets = {
        "baseline": ("a", "b"),
        "morgan_low": ("c", "d"),
        "morgan_high": ("c", "d"),
        "assay_base": ("e", "f"),
        "assay_low": ("g", "h"),
        "assay_high": ("g", "h"),
        "assay_050_a": ("i", "j"),
        "assay_050_b": ("k", "l"),
    }
    panel = ("task", "query", "L2")
    panels = {
        name: {panel: frozenset(uids)} for name, uids in uid_sets.items()
    }
    components = {
        profile.name: {"morgan": 0.5, "assay": 0.5} for profile in profiles
    }
    components["morgan_low"]["morgan"] = 0.1
    components["morgan_high"]["morgan"] = 0.9
    components["assay_low"]["assay"] = 0.1
    components["assay_high"]["assay"] = 0.9

    selected, _, edges, details = screen_by_uid_overlap(
        profiles,
        panels,
        components,
        ["baseline"],
        k=2,
        threshold=0.8,
        profiles_per_assay_stratum=2,
    )

    assert len(selected) == 6
    assert details["selected_stratum_counts"] == {"0": 2, "0.25": 2, "0.5": 2}
    assert {row["left_profile"] for row in edges} == {"morgan_low", "assay_low"}
    assert "morgan_high" in selected and "morgan_low" not in selected
    assert "assay_high" in selected and "assay_low" not in selected


def test_overlap_screen_keeps_conflicting_anchors_as_an_explicit_exception() -> None:
    profiles = [
        Profile("zero", assay_lambda=0),
        Profile("quarter", assay_lambda=0.25),
        Profile("half", assay_lambda=0.5),
    ]
    panel = ("task", "query", "L2")
    panels = {
        "zero": {panel: frozenset(("a", "b"))},
        "quarter": {panel: frozenset(("a", "b"))},
        "half": {panel: frozenset(("c", "d"))},
    }
    components = {
        profile.name: {"morgan": 1.0, "assay": 1.0} for profile in profiles
    }

    selected, _, edges, details = screen_by_uid_overlap(
        profiles, panels, components, ["zero", "quarter"], k=2, threshold=0.8
    )

    assert selected == ["zero", "quarter", "half"]
    assert len(edges) == 1 and edges[0]["anchor_exception"] is True
    assert details["anchor_exceptions"] == edges
