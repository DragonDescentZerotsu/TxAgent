from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.audit_skin_seed import (
    select_balanced_seed,
)
from tools.chembl_tool.paper_experiments.skin_causal_panel_seed.skin_contract import (
    causal_panel_retrieval_config,
    consensus_direction,
)


def test_consensus_direction_requires_strict_70_percent_agreement() -> None:
    assert consensus_direction(["positive"] * 7 + ["negative"] * 3) == "positive"
    assert consensus_direction(["positive"] * 6 + ["negative"] * 4) is None
    assert consensus_direction(["equivocal", ""]) is None


def test_balanced_seed_selection_is_direction_balanced_and_label_blind() -> None:
    rows = [
        {
            "query_index": index,
            "top_causal_direction": "negative" if index < 4 else "positive",
            "top_causal_similarity": 1.0 - index / 100,
            # This intentionally contradictory field must not affect selection.
            "label": index % 2,
        }
        for index in range(10)
    ]
    selected = select_balanced_seed(rows, seed_size=6)
    assert selected == [0, 1, 2, 4, 5, 6]


def test_skin_seed_source_config_keeps_tier1_and_replaces_raw_aop() -> None:
    config = causal_panel_retrieval_config()
    assert [group.group_id for group in config.mechanism_groups] == [
        "Mechanism.tier_1",
        "Mechanism.causal_panel",
    ]
    assert config.mechanism_groups[0].source_groups == ("Direct.skin_reaction",)
    assert config.mechanism_groups[1].source_groups == (
        "Mechanism.sensitization_causal_panel",
    )
