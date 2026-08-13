from tools.chembl_tool.paper_experiments.matched_train_label_agent.bbb_property_compatibility import (
    ionization_class,
    property_distance,
    robust_feature_scales,
    select_compatible_neighbors,
)
from tools.chembl_tool.paper_experiments.matched_train_label_agent import (
    molecule_property_profiles,
)


def _features(
    neutral: float,
    logd: float,
    mw: float,
    tpsa: float,
    hbd: float,
    rotors: float,
    acidic: float = 0,
    basic: float = 0,
) -> dict[str, float]:
    return {
        "pka__fraction_neutral": neutral,
        "pka__logd_estimate": logd,
        "pka__num_acidic_sites": acidic,
        "pka__num_basic_sites": basic,
        "rdkit__MolWt": mw,
        "rdkit__TPSA": tpsa,
        "rdkit__NumHDonors": hbd,
        "rdkit__NumRotatableBonds": rotors,
    }


def test_ionization_class_separates_neutral_acid_base_and_ampholyte() -> None:
    assert ionization_class(_features(0.9, 2, 300, 40, 0, 3)) == "mostly_neutral"
    assert ionization_class(_features(0.1, 2, 300, 40, 0, 3, acidic=1)) == "mostly_ionized_acidic"
    assert ionization_class(_features(0.1, 2, 300, 40, 0, 3, basic=1)) == "mostly_ionized_basic"
    assert (
        ionization_class(_features(0.1, 2, 300, 40, 0, 3, acidic=1, basic=1))
        == "mostly_ionized_ampholytic"
    )


def test_property_distance_uses_train_only_robust_scales() -> None:
    train = [
        _features(0.0, 0, 100, 10, 0, 0),
        _features(0.5, 1, 200, 20, 1, 2),
        _features(1.0, 2, 300, 30, 2, 4),
    ]
    scales = robust_feature_scales(train)
    assert property_distance(train[1], train[1], scales) == 0
    assert property_distance(train[0], train[2], scales) > 0


def test_compatible_selection_respects_similarity_floor_and_ignores_labels() -> None:
    query = _features(0.1, 1.0, 200, 30, 1, 2, basic=1)
    train = [
        _features(0.9, 4.0, 500, 100, 3, 10),
        _features(0.1, 1.1, 205, 31, 1, 2, basic=1),
        _features(0.1, 1.2, 210, 32, 1, 3, basic=1),
        _features(0.1, 1.3, 215, 33, 1, 3, basic=1),
    ]
    scales = robust_feature_scales([query, *train])
    candidates = [
        {"train_index": 0, "Y": 1, "similarity": 0.60, "morgan_rank": 1},
        {"train_index": 1, "Y": 0, "similarity": 0.58, "morgan_rank": 2},
        {"train_index": 2, "Y": 1, "similarity": 0.55, "morgan_rank": 3},
        {"train_index": 3, "Y": 0, "similarity": 0.51, "morgan_rank": 4},
    ]

    selected = select_compatible_neighbors(
        candidates,
        query_features=query,
        train_features=train,
        scales=scales,
        top_k=3,
        similarity_floor=0.50,
    )

    assert [row["train_index"] for row in selected] == [1, 2, 3]
    flipped_labels = [{**row, "Y": 1 - row["Y"]} for row in candidates]
    selected_after_label_flip = select_compatible_neighbors(
        flipped_labels,
        query_features=query,
        train_features=train,
        scales=scales,
        top_k=3,
        similarity_floor=0.50,
    )
    assert [row["train_index"] for row in selected_after_label_flip] == [1, 2, 3]


def test_profile_materializer_records_explicit_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        molecule_property_profiles,
        "_invoke_batch",
        lambda _url, _smiles: [
            {"status": "error", "errors": [{"code": "pka_failure"}]}
        ],
    )
    fallback = _features(0.1, 1.0, 200, 30, 1, 2, basic=1)
    profiles, statuses = molecule_property_profiles.load_or_fetch_profiles(
        [("train:0", "CCN")],
        path=tmp_path / "profiles.jsonl",
        required_features=set(fallback),
        tool_service_url="unused",
        batch_size=1,
        fallback=lambda _smiles: fallback,
    )

    assert profiles["train:0"] == fallback
    assert statuses == {"rdkit_fallback_pka_unavailable": 1}
