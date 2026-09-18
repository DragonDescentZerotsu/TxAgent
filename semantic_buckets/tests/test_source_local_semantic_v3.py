import json

import pandas as pd

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import source_local_semantic_v3 as workflow


def test_readout_columns_are_varying_unconsumed_pair_columns(monkeypatch) -> None:
    monkeypatch.setattr(core, "REFINEMENT_COLUMNS", {"source": ("first", "second")})
    lookup = {
        "a": {"values": {"first": "same", "second": "x"}},
        "b": {"values": {"first": "same", "second": "y"}},
    }

    assert workflow._varying_columns("source", ["a", "b"], lookup, []) == ["second"]
    assert workflow._varying_columns("source", ["a", "b"], lookup, ["second"]) == []


def test_readout_roots_never_cross_source_or_semantic_parent() -> None:
    mapping = pd.DataFrame(
        [
            {
                "level": "L2",
                "source_id": "left",
                "semantic_bucket_id": "semantic-left",
                "atom_id": "a",
            },
            {
                "level": "L2",
                "source_id": "right",
                "semantic_bucket_id": "semantic-right",
                "atom_id": "b",
            },
        ]
    )

    roots = workflow._root_branches(mapping)

    assert [(row["semantic_bucket_id"], row["source_id"], row["atom_ids"]) for row in roots] == [
        ("semantic-left", "left", ["a"]),
        ("semantic-right", "right", ["b"]),
    ]
    assert all(row["depth"] == 0 and row["consumed_columns"] == [] for row in roots)


def test_source_local_readout_prompt_renders_valid_json_payload(monkeypatch) -> None:
    monkeypatch.setattr(core, "REFINEMENT_COLUMNS", {"source": ("endpoint",)})
    monkeypatch.setattr(core, "PROMPT_DIMENSION_COLUMNS", {"source": ("endpoint",)})
    monkeypatch.setattr(core, "COLUMN_DESCRIPTIONS", {"endpoint": "measured endpoint"})
    monkeypatch.setattr(core, "TASK_NAME", "Task")
    lookup = {
        "a": {"source_id": "source", "record_count": 1, "values": {"endpoint": "x"}},
        "b": {"source_id": "source", "record_count": 1, "values": {"endpoint": "y"}},
    }
    cards = {"a": {"endpoint": "x"}, "b": {"endpoint": "y"}}
    branch = {
        "semantic_bucket_id": "semantic",
        "branch_id": "readout",
        "level": "L2",
        "source_id": "source",
        "depth": 0,
        "consumed_columns": [],
        "atom_ids": ["a", "b"],
    }

    prompt, candidates = workflow._coherence_prompt(branch, lookup, cards)
    payload = json.loads(prompt[prompt.index("{") : prompt.rindex("}") + 1].split("\n\nReturn only", 1)[0])

    assert candidates == ["endpoint"]
    assert payload["candidate_columns"][0]["column"] == "endpoint"
    assert payload["sample_records"] == [{"endpoint": "x"}, {"endpoint": "y"}]

    merge_prompt, items = workflow._merge_prompt(branch, "endpoint", lookup, cards)
    merge_payload = json.loads(
        merge_prompt[merge_prompt.index("{") : merge_prompt.rindex("}") + 1].split(
            "\n\nReturn only", 1
        )[0]
    )
    assert sorted(items.values()) == [["a"], ["b"]]
    assert merge_payload["sample_records"] == [{"endpoint": "x"}, {"endpoint": "y"}]


def test_final_profiles_restore_consumed_columns_and_are_exact(monkeypatch) -> None:
    monkeypatch.setattr(core, "REFINEMENT_COLUMNS", {"source": ("first", "second")})
    monkeypatch.setattr(core, "PROMPT_DIMENSION_COLUMNS", {"source": ("first", "second")})
    monkeypatch.setattr(
        core,
        "COLUMN_DESCRIPTIONS",
        {"first": "first dimension", "second": "second dimension"},
    )
    monkeypatch.setattr(core, "TASK_NAME", "Task")
    lookup = {
        "a": {"source_id": "source", "record_count": 1, "values": {"first": "x", "second": "m"}},
        "b": {"source_id": "source", "record_count": 1, "values": {"first": "y", "second": "m"}},
        "c": {"source_id": "source", "record_count": 1, "values": {"first": "y", "second": "n"}},
    }
    branch = {
        "semantic_bucket_id": "semantic",
        "branch_id": "readout",
        "level": "L2",
        "source_id": "source",
        "depth": 1,
        "consumed_columns": ["first"],
        "atom_ids": ["a", "b", "c"],
    }

    prompt, items, columns = workflow._profile_prompt(branch, "second", lookup, lookup)
    payload = json.loads(prompt[prompt.index("{") : prompt.rindex("}") + 1].split("\n\nReturn only", 1)[0])

    assert columns == ["second", "first"]
    assert {tuple(sorted(row["canonical_dimensions"].items())) for row in payload["profiles"]} == {
        (("first", "x"), ("second", "m")),
        (("first", "y"), ("second", "m")),
        (("first", "y"), ("second", "n")),
    }
    assert sorted(items.values()) == [["a"], ["b"], ["c"]]
    assert workflow._exact_profile_groups(branch, lookup) == [["a"], ["b"], ["c"]]
