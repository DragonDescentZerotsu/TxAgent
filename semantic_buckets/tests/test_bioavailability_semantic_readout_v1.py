import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest
import pandas as pd

from semantic_buckets import bioavailability_semantic_readout_v1 as workflow
from semantic_buckets import bioavailability_semantic_readout_v2 as oral_v2
from semantic_buckets import (
    bioavailability_semantic_readout_v3_l4_refinement as l4_refinement,
)
from semantic_buckets import bbb_semantic_readout_v2 as bbb_v2


def test_pair_bucket_parser_uses_source_contract() -> None:
    parsed = workflow.parse_pair_bucket(
        "fa", '["fa","solubility","mg/mL","kinetic","human"]'
    )
    assert parsed == {
        "canonical_endpoint_concept": "solubility",
        "canonical_unit_text": "mg/mL",
        "canonical_assay_context": "kinetic",
        "canonical_species_context": "human",
    }
    with pytest.raises(ValueError, match="expected"):
        workflow.parse_pair_bucket("fa", '["fa","solubility"]')

    oral = workflow.parse_pair_bucket(
        "oral_exposure",
        '["oral_exposure","auc","ng*h/mL","human","plasma","mass|absolute|mg|5"]',
    )
    assert oral["canonical_direct_condition_group"] == "__not_present_in_pair_key__"


def test_merge_sets_preserve_membership_and_default_to_singletons() -> None:
    items = {"a": ["atom-a"], "b": ["atom-b"], "c": ["atom-c"]}
    merged = workflow.apply_merge_sets(
        items,
        {"merge_sets": [{"member_ids": ["a", "c"]}]},
    )
    assert sorted(map(tuple, merged.values())) == [("atom-a", "atom-c"), ("atom-b",)]


def test_merge_sets_reject_overlap_and_unknown_ids() -> None:
    items = {"a": ["atom-a"], "b": ["atom-b"], "c": ["atom-c"]}
    with pytest.raises(ValueError, match="multiple merge sets"):
        workflow.apply_merge_sets(
            items,
            {
                "merge_sets": [
                    {"member_ids": ["a", "b"]},
                    {"member_ids": ["b", "c"]},
                ]
            },
        )
    with pytest.raises(ValueError, match="unknown IDs"):
        workflow.apply_merge_sets(
            items, {"merge_sets": [{"member_ids": ["a", "missing"]}]}
        )


def test_l4_refinement_replaces_only_target_group() -> None:
    base = pd.DataFrame(
        [
            {
                "level": "L3",
                "source_id": "fa",
                "source_semantic_bucket_id": "unchanged-level",
                "atom_id": "a",
            },
            {
                "level": "L4",
                "source_id": "fg",
                "source_semantic_bucket_id": "unchanged-source",
                "atom_id": "b",
            },
            {
                "level": "L4",
                "source_id": "fa",
                "source_semantic_bucket_id": "old-c",
                "atom_id": "c",
            },
            {
                "level": "L4",
                "source_id": "fa",
                "source_semantic_bucket_id": "old-d",
                "atom_id": "d",
            },
        ]
    )

    result = l4_refinement.replace_l4_mapping(base, {"new": ["c", "d"]})

    assert result["source_semantic_bucket_id"].tolist() == [
        "unchanged-level",
        "unchanged-source",
        "new",
        "new",
    ]
    with pytest.raises(ValueError, match="coverage changed"):
        l4_refinement.replace_l4_mapping(base, {"new": ["c"]})

    with pytest.raises(ValueError, match="overlap"):
        l4_refinement.replace_l4_mapping(base, {"new": ["c", "d", "d"]})


def test_keep_is_terminal_while_deferred_and_split_children_stay_active() -> None:
    active = {"keep": ["a"], "defer": ["d"], "split": ["b", "c"]}
    children = {
        "split": {
            workflow.bucket_id(["b"]): ["b"],
            workflow.bucket_id(["c"]): ["c"],
        }
    }
    next_active, terminal = workflow.apply_split_decisions(
        active, {"keep": "keep", "defer": "defer", "split": "split"}, children
    )
    assert terminal == {"keep": ["a"]}
    assert sorted(next_active.values()) == [["b"], ["c"], ["d"]]


def test_deferred_bucket_skips_new_child_merge(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "REFINEMENT_COLUMNS", {"source": ("column",)})
    monkeypatch.setattr(workflow, "COLUMN_DESCRIPTIONS", {"column": "test column"})
    monkeypatch.setattr(workflow, "DEFER_TECHNICAL_BRANCHES", True)
    monkeypatch.setattr(workflow, "MAX_SOURCE_ROUNDS", 1)
    queued = {}

    def queue(_connection, *, kind, phase, **_kwargs):
        request_id = f"{kind}|{phase}"
        queued[request_id] = kind
        return request_id

    def response(_connection, request_id):
        kind = queued[request_id]
        if kind == "select_column":
            return {"decision": "select", "column": "column"}
        if kind == "split_bucket":
            return {"decision": "split"}
        return {"merge_sets": []}

    merge_calls = []

    def merge_round(_connection, *, buckets, phase, **_kwargs):
        merge_calls.append((phase, set(buckets)))
        return {key: list(value) for key, value in buckets.items()}

    monkeypatch.setattr(workflow, "_render", lambda name, _payload: name)
    monkeypatch.setattr(workflow, "_token_count", lambda _prompt: 1)
    monkeypatch.setattr(workflow, "_queue_checked_request", queue)
    monkeypatch.setattr(workflow, "_run_pending", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "_response", response)
    monkeypatch.setattr(workflow, "_merge_bucket_round", merge_round)

    group = {
        "level": "L2",
        "source_id": "source",
        "round": 0,
        "consumed_columns": [],
        "active": {"deferred-parent": ["a"], "split-parent": ["b", "c"]},
        "terminal": {},
        "termination_reasons": {},
        "history": [],
        "finished": False,
        "final_buckets": None,
    }
    lookup = {
        atom: {
            "source_id": "source",
            "record_count": 1,
            "values": {"column": value},
        }
        for atom, value in (("a", "x"), ("b", "x"), ("c", "y"))
    }
    with pytest.raises(RuntimeError, match="three-round cap with 3 active branches"):
        workflow._run_source_semantic(
            object(),
            group,
            lookup=lookup,
            sample_cards={},
            maximum_context=1_000_000,
            parallelism=1,
            client=object(),
            request_slots=threading.BoundedSemaphore(1),
            checkpoint=lambda: None,
        )
    child_merges = [items for phase, items in merge_calls if "new_children_merge" in phase]
    assert len(child_merges) == 2
    assert all("deferred-parent" not in items for items in child_merges)
    assert group["history"][0]["deferred_count"] == 1
    assert group["blocked_reason"] == "three_source_round_cap_with_active_branches"
    assert sorted(group["active"].values()) == [["a"], ["b"], ["c"]]


def test_v2_refinement_columns_are_pair_columns_without_direct_conditions() -> None:
    for configured, ignored in (
        (oral_v2, "canonical_direct_condition_group"),
        (bbb_v2, "condition_group"),
    ):
        pair_columns = (
            workflow.PAIR_COLUMNS
            if configured is oral_v2
            else bbb_v2.workflow.PAIR_COLUMNS
        )
        for source, columns in configured.REFINEMENT_COLUMNS.items():
            assert set(columns) <= set(pair_columns[source])
            assert ignored not in columns


def test_bucket_payload_samples_use_only_prompt_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "PROMPT_DIMENSION_COLUMNS", {"source": ("allowed",)})
    lookup = {
        "atom": {
            "source_id": "source",
            "record_count": 1,
            "values": {"allowed": "value"},
        }
    }
    cards = {
        "atom": {
            "allowed": "value",
            "canonical_record_id": "must-not-leak",
            "assay_model": "must-not-leak",
        }
    }

    payload = workflow._bucket_payload("bucket", ["atom"], lookup, cards)

    assert payload["sample_records"] == [{"allowed": "value"}]


def test_cross_source_candidates_anchor_every_non_dominant_bucket(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "CROSS_SOURCE_LEVELS", ("L2",))
    monkeypatch.setattr(workflow, "PROMPT_DIMENSION_COLUMNS", {source: ("identity",) for source in "ABC"})
    atoms = pd.DataFrame(
        [
            {
                "atom_id": atom,
                "source_id": source,
                "record_count": 1,
                "values_json": json.dumps({"identity": identity}),
            }
            for atom, source, identity in (
                ("a1", "A", "brain uptake"),
                ("a2", "A", "efflux transport"),
                ("a3", "A", "unrelated"),
                ("b1", "B", "brain uptake"),
                ("b2", "B", "passive permeability"),
                ("c1", "C", "carrier influx"),
            )
        ]
    )
    source_map = pd.DataFrame(
        [
            {
                "level": "L2",
                "source_id": row.source_id,
                "source_semantic_bucket_id": f"bucket-{row.atom_id}",
                "atom_id": row.atom_id,
            }
            for row in atoms.itertuples(index=False)
        ]
    )
    cards = {
        row.atom_id: {"identity": json.loads(row.values_json)["identity"]}
        for row in atoms.itertuples(index=False)
    }

    batches = workflow.cross_source_candidate_batches(
        source_map,
        atoms,
        cards,
        batch_size=5,
        exploration_per_source=1,
    )

    assert {batch["anchor_bucket_id"] for batch in batches} == {
        "bucket-b1",
        "bucket-b2",
        "bucket-c1",
    }
    brain_batch = next(
        batch for batch in batches if batch["anchor_bucket_id"] == "bucket-b1"
    )
    assert "bucket-a1" in {bucket["bucket_id"] for bucket in brain_batch["buckets"]}
    assert all(len(batch["buckets"]) <= 5 for batch in batches)
    assert all(
        bucket["source_id"]
        != next(
            item["source_id"]
            for item in batch["buckets"]
            if item["bucket_id"] == batch["anchor_bucket_id"]
        )
        for batch in batches
        for bucket in batch["buckets"][1:]
    )


def test_cross_source_merge_requires_one_anchor_spanning_sources() -> None:
    batch = {
        "anchor_bucket_id": "a",
        "buckets": [
            {"bucket_id": "a", "source_id": "A"},
            {"bucket_id": "b", "source_id": "B"},
            {"bucket_id": "c", "source_id": "C"},
        ],
    }
    response = {
        "merge_sets": [
            {"member_ids": ["a", "b"], "label": "same", "rationale": "same identity"}
        ],
        "rationale": "reviewed",
    }
    assert workflow.validate_cross_source_merge_response(response, batch) == [["a", "b"]]

    response["merge_sets"][0]["member_ids"] = ["b", "c"]
    with pytest.raises(ValueError, match="anchor"):
        workflow.validate_cross_source_merge_response(response, batch)

    validation = {
        "valid_ids": ["a", "b", "c"],
        "anchor_bucket_id": "a",
        "source_by_id": {"a": "A", "b": "B", "c": "C"},
    }
    with pytest.raises(ValueError, match="anchor"):
        workflow._validate_model_response(
            "cross_source_merge", response, validation
        )


def test_cross_source_components_take_transitive_closure() -> None:
    source = pd.DataFrame(
        [
            {"source_semantic_bucket_id": "a", "source_id": "A", "level": "L2"},
            {"source_semantic_bucket_id": "b", "source_id": "B", "level": "L2"},
            {"source_semantic_bucket_id": "c", "source_id": "C", "level": "L2"},
        ]
    )
    decisions = [
        {
            "request_id": "r1",
            "response": {
                "merge_sets": [
                    {"member_ids": ["a", "b"], "label": "x", "rationale": "ab"}
                ]
            },
        },
        {
            "request_id": "r2",
            "response": {
                "merge_sets": [
                    {"member_ids": ["b", "c"], "label": "x", "rationale": "bc"}
                ]
            },
        },
    ]

    components = workflow._cross_source_components(source, decisions)

    assert len(components) == 1
    assert components[0]["member_ids"] == ["a", "b", "c"]
    assert components[0]["decision_request_ids"] == ["r1", "r2"]


def test_cross_source_run_is_review_gated_and_publishes_proposals(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "semantic"
    output.mkdir()
    monkeypatch.setattr(workflow, "VERSION", "test.v1")
    monkeypatch.setattr(workflow, "TASK_NAME", "test")
    monkeypatch.setattr(workflow, "CROSS_SOURCE_LEVELS", ("L2",))
    monkeypatch.setattr(
        workflow,
        "PROMPT_DIMENSION_COLUMNS",
        {"A": ("identity",), "B": ("identity",)},
    )
    monkeypatch.setattr(
        workflow,
        "_endpoint_contract",
        lambda: (1_048_576, [workflow.MODEL]),
    )
    monkeypatch.setattr(workflow, "_token_count", lambda prompt: len(prompt.split()))

    atoms = pd.DataFrame(
        [
            {
                "atom_id": atom,
                "level": "L2",
                "source_id": source,
                "record_count": 1,
                "values_json": json.dumps({"identity": identity}),
            }
            for atom, source, identity in (
                ("a1", "A", "brain uptake"),
                ("a2", "A", "other"),
                ("b1", "B", "brain uptake"),
            )
        ]
    )
    source_map = pd.DataFrame(
        [
            {
                "level": "L2",
                "source_id": row.source_id,
                "source_semantic_bucket_id": f"bucket-{row.atom_id}",
                "atom_id": row.atom_id,
            }
            for row in atoms.itertuples(index=False)
        ]
    )
    source_path = output / "source_semantic_bucket_map.parquet"
    source_map.to_parquet(source_path, index=False)
    atoms.to_parquet(output / "input_atoms.parquet", index=False)
    workflow._write_gzip_json(
        output / "sample_cards.json.gz",
        {
            row.atom_id: {"identity": json.loads(row.values_json)["identity"]}
            for row in atoms.itertuples(index=False)
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "awaiting_cross_source_mapping",
                "source_semantic_bucket_map": {
                    "sha256": workflow._file_sha256(source_path)
                },
            }
        )
    )
    review_output = tmp_path / "review"
    workflow.prepare_cross_source_prompt_review(
        output, review_output=review_output
    )

    class FakeClient:
        @staticmethod
        def snapshot():
            return [{"name": "fake", "completed": 1}]

    monkeypatch.setattr(
        workflow,
        "_build_completion_pool",
        lambda parallelism: (FakeClient(), [{"name": "fake"}], parallelism),
    )

    def complete_requests(connection, request_ids, **_kwargs):
        for request_id in request_ids:
            validation = json.loads(
                connection.execute(
                    "SELECT validation_json FROM requests WHERE request_id=?",
                    (request_id,),
                ).fetchone()[0]
            )
            response = {
                "merge_sets": [
                    {
                        "member_ids": validation["valid_ids"][:2],
                        "label": "brain uptake",
                        "rationale": "same scientific identity",
                    }
                ],
                "rationale": "reviewed",
            }
            connection.execute(
                """
                UPDATE requests SET status='complete',attempts=1,response_json=?,
                    served_model=?,provider_name='fake',provider_base_url='fake'
                WHERE request_id=?
                """,
                (workflow._canonical_json(response), workflow.MODEL, request_id),
            )
        connection.commit()

    monkeypatch.setattr(workflow, "_run_pending", complete_requests)
    review_manifest = review_output / "manifest.json"
    manifest = workflow.run_cross_source_merges(
        output,
        review_manifest_path=review_manifest,
        approved_review_sha256=workflow._file_sha256(review_manifest),
        parallelism=1,
    )

    proposal = json.loads((output / "cross_source_model_proposals.json").read_text())
    assert manifest["status"] == "awaiting_agentic_cross_source_review"
    assert manifest["cross_source_request_summary"]["complete"] == 1
    assert proposal["components"][0]["source_ids"] == ["A", "B"]


def test_seed_request_cache_reuses_only_the_exact_contract(tmp_path: Path) -> None:
    source_path = tmp_path / "source.sqlite3"
    source = workflow._request_database(source_path)
    request_id = workflow._request_id("split_bucket", "phase", "prompt")
    workflow._queue_request(
        source,
        request_id=request_id,
        kind="split_bucket",
        phase="phase",
        prompt="prompt",
        reasoning_effort="low",
        max_tokens=100,
        validation={},
    )
    source.execute(
        "UPDATE requests SET status='complete',response_json='{}',served_model=?",
        (workflow.MODEL,),
    )
    source.commit()
    source.close()

    target_path = tmp_path / "target.sqlite3"
    receipt = workflow._seed_request_database(target_path, source_path)
    assert receipt is not None
    assert receipt["complete_requests_copied"] == 1

    target = workflow._request_database(target_path)
    assert target.execute(
        "SELECT status FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()[0] == "complete"
    with pytest.raises(ValueError, match="cached request contract changed"):
        workflow._queue_request(
            target,
            request_id=request_id,
            kind="split_bucket",
            phase="phase",
            prompt="prompt",
            reasoning_effort="low",
            max_tokens=100,
            validation={"changed": True},
        )
    target.close()


def test_seed_request_cache_excludes_noncomplete_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "source.sqlite3"
    source = workflow._request_database(source_path)
    for request_id in ("complete", "pending"):
        workflow._queue_request(
            source,
            request_id=request_id,
            kind="split_bucket",
            phase=request_id,
            prompt=request_id,
            reasoning_effort="low",
            max_tokens=100,
            validation={},
        )
    source.execute(
        "UPDATE requests SET status='complete',response_json='{}',served_model=? "
        "WHERE request_id='complete'",
        (workflow.MODEL,),
    )
    source.commit()
    source.close()

    target_path = tmp_path / "target.sqlite3"
    receipt = workflow._seed_request_database(target_path, source_path)
    assert receipt["complete_requests_copied"] == 1
    assert receipt["noncomplete_requests_excluded"] == 1
    assert receipt["source_status_counts"] == {"complete": 1, "pending": 1}

    target = workflow._request_database(target_path)
    assert [
        tuple(row)
        for row in target.execute("SELECT request_id,status FROM requests")
    ] == [("complete", "complete")]
    target.close()


def test_semantic_size_audit_flags_large_and_top_buckets(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "REFINEMENT_COLUMNS", {"source": ("column",)})
    pd.DataFrame(
        [
            {
                "level": "L2",
                "source_id": "source",
                "source_semantic_bucket_id": "source-large",
                "semantic_bucket_id": "large",
                "atom_id": "a",
            },
            {
                "level": "L2",
                "source_id": "source",
                "source_semantic_bucket_id": "source-small",
                "semantic_bucket_id": "small",
                "atom_id": "b",
            },
        ]
    ).to_parquet(tmp_path / "semantic_bucket_map.parquet", index=False)
    pd.DataFrame(
        [
            {"atom_id": "a", "record_count": 9, "values_json": '{"column":"x"}'},
            {"atom_id": "b", "record_count": 1, "values_json": '{"column":"y"}'},
        ]
    ).to_parquet(tmp_path / "input_atoms.parquet", index=False)
    workflow._write_gzip_json(
        tmp_path / "sample_cards.json.gz", {"a": {"id": "a"}, "b": {"id": "b"}}
    )
    (tmp_path / "semantic_state.json").write_text(
        json.dumps(
            {
                "groups": {
                    "L2|source": {
                        "terminal": {"terminal-a": ["a"], "terminal-b": ["b"]},
                        "termination_reasons": {
                            "terminal-a": "deepseek_keep",
                            "terminal-b": "three_source_round_cap",
                        },
                    }
                }
            }
        )
    )
    manifest = workflow.audit_semantic_bucket_sizes(
        tmp_path, fraction_threshold=0.5, top_per_level=1
    )
    assert manifest["flagged_bucket_count"] == 1
    assert manifest["over_fraction_threshold_count"] == 1
    audit = pd.read_parquet(tmp_path / "semantic_bucket_size_audit.parquet")
    assert audit.loc[audit["semantic_bucket_id"] == "large", "review_required"].item()
    assert not audit.loc[audit["semantic_bucket_id"] == "small", "review_required"].item()

    map_sha256 = workflow._file_sha256(tmp_path / "semantic_bucket_map.parquet")
    workflow.write_json_atomic(
        tmp_path / "semantic_bucket_map_manifest.json",
        {
            "status": "awaiting_size_review",
            "semantic_bucket_map_sha256": map_sha256,
        },
    )
    rejected_review = {
        "version": f"{workflow.VERSION}.semantic_bucket_size_review.v1",
        "reviewer": "test-reviewer",
        "semantic_bucket_map_sha256": map_sha256,
        "bucket_reviews": [
            {
                "level": "L2",
                "semantic_bucket_id": "large",
                "decision": "needs_refinement",
                "rationale": "The bucket mixes distinct readouts.",
            }
        ],
    }
    rejected_path = tmp_path / "rejected_review.json"
    workflow.write_json_atomic(rejected_path, rejected_review)
    rejected = workflow.approve_semantic_bucket_sizes(
        tmp_path, review_path=rejected_path
    )
    assert rejected["status"] == "refinement_required"
    assert json.loads(
        (tmp_path / "semantic_bucket_map_manifest.json").read_text()
    )["status"] == "awaiting_size_review"

    approved_review = rejected_review.copy()
    approved_review["bucket_reviews"] = [
        {
            **rejected_review["bucket_reviews"][0],
            "decision": "coherent",
            "rationale": "The bucket is one scientifically coherent readout.",
        }
    ]
    approved_path = tmp_path / "approved_review.json"
    workflow.write_json_atomic(approved_path, approved_review)
    approved = workflow.approve_semantic_bucket_sizes(
        tmp_path, review_path=approved_path
    )
    assert approved["status"] == "complete_reviewed"
    assert json.loads(
        (tmp_path / "semantic_bucket_map_manifest.json").read_text()
    )["status"] == "complete_reviewed"


def test_split_children_must_partition_parent() -> None:
    with pytest.raises(ValueError, match="partition"):
        workflow.apply_split_decisions(
            {"parent": ["a", "b"]},
            {"parent": "split"},
            {"parent": {"child-a": ["a"], "child-c": ["c"]}},
        )


def test_high_reasoning_merge_can_reverse_a_split_to_keep() -> None:
    decisions = {"parent": "split"}
    children = {"parent": {"only-child": ["a", "b"]}}
    reasons = {}

    assert workflow.reverse_collapsed_splits(decisions, children, reasons) == 1
    assert decisions == {"parent": "keep"}
    assert children == {}
    assert reasons == {"parent": "high_reasoning_merge_reversed_split"}
    assert workflow.apply_split_decisions(
        {"parent": ["a", "b"]}, decisions, children
    ) == ({}, {"parent": ["a", "b"]})


def test_randomized_batches_are_seeded_and_cover_each_item_once() -> None:
    items = [f"b{index:03d}" for index in range(85)]
    first = workflow.deterministic_batches(items, seed="round-1")
    repeated = workflow.deterministic_batches(list(reversed(items)), seed="round-1")
    second = workflow.deterministic_batches(items, seed="round-2")
    assert first == repeated
    assert first != second
    assert sorted(item for batch in first for item in batch) == items
    assert max(map(len, first)) == 40


def test_degree25_schedule_is_exact_or_parity_balanced() -> None:
    even = workflow.degree_edges(
        [f"b{index:02d}" for index in range(28)], level="L2"
    )
    even_degree = Counter(bucket for edge in even for bucket in edge)
    assert set(even_degree.values()) == {25}
    assert len({tuple(sorted(edge)) for edge in even}) == len(even)
    even_first = Counter(left for left, _ in even)
    assert max(abs(2 * even_first[key] - even_degree[key]) for key in even_degree) == 1

    odd = workflow.degree_edges(
        [f"b{index:02d}" for index in range(29)], level="L3"
    )
    odd_degree = Counter(bucket for edge in odd for bucket in edge)
    assert Counter(odd_degree.values()) == {24: 1, 25: 28}
    odd_first = Counter(left for left, _ in odd)
    assert max(abs(2 * odd_first[key] - odd_degree[key]) for key in odd_degree) == 1

    complete = workflow.degree_edges(
        [f"b{index:02d}" for index in range(20)], level="L4"
    )
    assert len(complete) == 20 * 19 // 2


def test_readout_coherence_stops_branch_and_root_becomes_depth_one() -> None:
    result = workflow.apply_readout_decision(
        ["a", "b"], decision="coherent", depth=0
    )
    assert result["active"] == {}
    assert result["terminal_depth"] == 1
    assert result["termination_reason"] == "coherent_readout"


def test_readout_split_recurses_only_within_parent_and_stops_at_cap() -> None:
    children = {
        workflow.bucket_id(["a"]): ["a"],
        workflow.bucket_id(["b"]): ["b"],
    }
    split = workflow.apply_readout_decision(
        ["a", "b"], decision="split", depth=0, children=children
    )
    assert split["active_depth"] == 1
    assert sorted(split["active"].values()) == [["a"], ["b"]]

    capped = workflow.apply_readout_decision(
        ["a", "b"], decision="split", depth=2
    )
    assert capped["active"] == {}
    assert capped["terminal"] == {}
    assert sorted(capped["blocked"].values()) == [["a", "b"]]
    assert capped["blocked_depth"] == 2
    assert capped["termination_reason"] == "incoherent_at_depth_cap"


def test_response_validation_enforces_exact_ids_and_ranking_shape() -> None:
    ranking = workflow._validate_model_response(
        "ranking", {"winner_bucket_id": "b"}, {"candidate_bucket_ids": ["a", "b"]}
    )
    assert ranking == {"winner_bucket_id": "b"}

    with pytest.raises(ValueError, match="winner was not supplied"):
        workflow._validate_model_response(
            "ranking",
            {"winner_bucket_id": "missing"},
            {"candidate_bucket_ids": ["a", "b"]},
        )
    with pytest.raises(ValueError, match="unexpected fields"):
        workflow._validate_model_response(
            "split_bucket",
            {"decision": "keep", "rationale": "same readout", "extra": True},
            {},
        )


def test_cross_source_review_merges_only_l2_across_sources() -> None:
    source = pd.DataFrame(
        [
            {"level": "L2", "source_id": "fa", "source_semantic_bucket_id": "fa-a", "atom_id": "a"},
            {"level": "L2", "source_id": "fg", "source_semantic_bucket_id": "fg-b", "atom_id": "b"},
            {"level": "L2", "source_id": "fa", "source_semantic_bucket_id": "fa-c", "atom_id": "c"},
            {"level": "L3", "source_id": "oral_exposure", "source_semantic_bucket_id": "oral-d", "atom_id": "d"},
        ]
    )
    atoms = pd.DataFrame({"atom_id": ["a", "b", "c", "d"]})
    mapping = workflow.apply_cross_source_review(
        source,
        atoms,
        [
            {
                "member_ids": ["fa-a", "fg-b"],
                "label": "shared concept",
                "rationale": "same biological evidence across sources",
            }
        ],
    )
    assert mapping["a"] == mapping["b"]
    assert mapping["c"] == workflow.bucket_id(["c"])
    assert mapping["d"] == "oral-d"

    with pytest.raises(ValueError, match="does not cross sources"):
        workflow.apply_cross_source_review(
            source,
            atoms,
            [
                {
                    "member_ids": ["fa-a", "fa-c"],
                    "label": "invalid",
                    "rationale": "same source",
                }
            ],
        )


def test_oversized_two_bucket_merge_fails_closed_to_singletons(
    tmp_path, monkeypatch
) -> None:
    connection = workflow._request_database(tmp_path / "requests.sqlite3")
    values = {
        "canonical_endpoint_concept": "endpoint",
        "canonical_unit_text": "unit",
        "canonical_assay_context": "context",
        "canonical_species_context": "human",
    }
    lookup = {
        "a": {"source_id": "fa", "record_count": 1, "values": values},
        "b": {"source_id": "fa", "record_count": 1, "values": values},
    }
    cards = {
        "a": {"canonical_endpoint_concept": "endpoint-a"},
        "b": {"canonical_endpoint_concept": "endpoint-b"},
    }
    monkeypatch.setattr(workflow, "_token_count", lambda _prompt: 2)
    result = workflow._merge_bucket_round(
        connection,
        level="L2",
        source="fa",
        buckets={"left": ["a"], "right": ["b"]},
        lookup=lookup,
        sample_cards=cards,
        phase="test",
        maximum_context=workflow.HIGH_MAX_TOKENS + 1,
        parallelism=2,
        client=object(),
        request_slots=threading.BoundedSemaphore(2),
    )
    assert sorted(result.values()) == [["a"], ["b"]]
    assert connection.execute("select count(*) from requests").fetchone()[0] == 0
    connection.close()


def test_request_pool_honors_one_global_inflight_cap(tmp_path) -> None:
    connection = workflow._request_database(tmp_path / "requests.sqlite3")
    request_ids = []
    for index in range(10):
        request_id = f"request-{index}"
        workflow._queue_request(
            connection,
            request_id=request_id,
            kind="split_bucket",
            phase=f"phase-{index}",
            prompt=f"prompt-{index}",
            reasoning_effort="low",
            max_tokens=100,
            validation={},
        )
        request_ids.append(request_id)

    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def create(**_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        message = SimpleNamespace(
            content=json.dumps({"decision": "keep", "rationale": "one readout"}),
            reasoning_content="brief reasoning",
            reasoning=None,
            model_extra={},
        )
        return SimpleNamespace(
            id="generation",
            model=workflow.MODEL,
            usage=None,
            choices=[SimpleNamespace(message=message)],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    workflow._run_pending(
        connection,
        request_ids,
        parallelism=10,
        client=client,
        request_slots=threading.BoundedSemaphore(3),
    )
    connection.close()
    assert maximum_active == 3


def test_weight_request_uses_its_qualified_model_and_route(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(workflow, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(workflow, "WEIGHT_REQUEST_EXTRA_BODY", {"thinking": {"type": "enabled"}})
    connection = workflow._request_database(tmp_path / "requests.sqlite3")
    validation = {
        "candidate_aliases": ["Candidate 1"], "candidate_bucket_ids": ["bucket"],
        "anchor_bucket_ids": [], "requested_model": "requested-model",
        "allowed_served_models": ["requested-model", "canonical-model"],
        "selected_provider_route": "provider/fp8",
        "provider_pool_snapshot_sha256": "snapshot",
        "provider_routing": {"order": ["provider/fp8"], "allow_fallbacks": False,
                             "omit_response_format": True},
    }
    workflow._queue_request(
        connection, request_id="request", kind="weight_assignment", phase="phase",
        prompt="prompt", reasoning_effort="high", max_tokens=100, validation=validation,
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(
            content=json.dumps({"scores": [{"candidate": "Candidate 1", "weight": 0.5,
                                             "rationale": "valid"}]}),
            reasoning_content="reasoning", reasoning=None, model_extra={},
        )
        return SimpleNamespace(id="generation", model="canonical-model", usage=None,
                               choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    workflow._run_pending(connection, ["request"], parallelism=1, client=client,
                          request_slots=threading.BoundedSemaphore(1))

    receipt = json.loads(connection.execute(
        "SELECT receipt_json FROM request_attempt_receipts WHERE request_id='request'"
    ).fetchone()[0])
    connection.close()
    assert captured["model"] == "requested-model"
    assert captured["extra_body"]["provider"]["order"] == ["provider/fp8"]
    assert "response_format" not in captured
    assert receipt["requested_model"] == "requested-model"
    assert receipt["provider_pool_snapshot_sha256"] == "snapshot"


def test_independent_endpoint_pools_each_honor_their_capacity() -> None:
    lock = threading.Lock()
    active = Counter()
    maximum = Counter()

    def client(name):
        def create(**_kwargs):
            with lock:
                active[name] += 1
                maximum[name] = max(maximum[name], active[name])
            time.sleep(0.02)
            with lock:
                active[name] -= 1
            return SimpleNamespace(
                id=name,
                model=workflow.MODEL,
                usage=None,
                choices=[],
            )

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    pool = workflow._IndependentCompletionPool(
        [
            ("first", "http://first/v1", client("first"), 2),
            ("second", "http://second/v1", client("second"), 2),
        ]
    )
    with workflow.ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(lambda _: pool.create(), range(8)))

    assert Counter(response.provider_name for response in responses) == {
        "first": 4,
        "second": 4,
    }
    assert maximum == {"first": 2, "second": 2}


def test_endpoint_pool_opens_a_circuit_after_transport_failure() -> None:
    failing = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("502"))
            )
        )
    )
    healthy_completion = SimpleNamespace(
        id="healthy",
        model=workflow.MODEL,
        usage=None,
        choices=[],
    )
    healthy = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: healthy_completion)
        )
    )
    pool = workflow._IndependentCompletionPool(
        [
            ("failing", "http://failing/v1", failing, 1),
            ("healthy", "http://healthy/v1", healthy, 1),
        ],
        failure_threshold=1,
        cooldown_seconds=60,
    )

    with pytest.raises(RuntimeError, match="502"):
        pool.create()
    assert pool.create().provider_name == "healthy"
    assert pool.create().provider_name == "healthy"
    snapshot = {row["name"]: row for row in pool.snapshot()}
    assert snapshot["failing"]["failures"] == 1
    assert snapshot["failing"]["circuit_open"] is True


def test_invalid_response_is_preserved_in_attempt_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(workflow, "MAX_ATTEMPTS", 1)
    connection = workflow._request_database(tmp_path / "requests.sqlite3")
    workflow._queue_request(
        connection,
        request_id="request-invalid",
        kind="ranking",
        phase="ranking",
        prompt="prompt",
        reasoning_effort="low",
        max_tokens=100,
        validation={"candidate_bucket_ids": ["a", "b"]},
    )
    message = SimpleNamespace(
        content=json.dumps({"winner_bucket_id": "tie"}),
        reasoning_content=None,
        reasoning=None,
        model_extra={},
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    id="generation",
                    model=workflow.MODEL,
                    provider_name="test-provider",
                    provider_base_url="http://test-provider/v1",
                    usage=None,
                    choices=[SimpleNamespace(message=message)],
                )
            )
        )
    )

    with pytest.raises(RuntimeError, match="ranking winner was not supplied"):
        workflow._run_pending(
            connection,
            ["request-invalid"],
            parallelism=1,
            client=client,
            request_slots=threading.BoundedSemaphore(1),
        )

    receipt = connection.execute(
        "SELECT receipt_json FROM request_attempt_receipts WHERE request_id=?",
        ("request-invalid",),
    ).fetchone()
    connection.close()
    receipt = json.loads(receipt[0])
    assert receipt["raw_response"] == message.content
    assert receipt["provider_name"] == "test-provider"
    assert receipt["provider_base_url"] == "http://test-provider/v1"


def test_provider_pool_adapter_preserves_actual_route_provenance() -> None:
    spec = SimpleNamespace(
        model="deepseek/deepseek-v4.1-flash",
        request_extra_body={
            "allowed_served_models": ["deepseek/deepseek-v4.1-flash-20260910"]
        },
    )
    pool = SimpleNamespace(
        config=SimpleNamespace(providers=(spec,)),
        chat_json=lambda *_args, **_kwargs: {
            "content": {"merge_sets": []},
            "reasoning_content": "reasoning",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "deepseek/deepseek-v4.1-flash-20260910",
            "id": "generation",
            "execution_provider": {
                "provider": "openrouter_route",
                "base_url": "https://openrouter.ai/api/v1",
                "requested_model": "deepseek/deepseek-v4.1-flash",
            },
            "execution_provider_attempts": [{"status": "ok"}],
        },
        snapshot=lambda: {"providers": []},
    )

    completion = workflow._ProviderPoolCompletionAdapter(pool).create(
        messages=[{"role": "user", "content": "prompt"}], max_tokens=100
    )

    assert completion.model == "deepseek/deepseek-v4.1-flash-20260910"
    assert completion.requested_model == "deepseek/deepseek-v4.1-flash"
    assert completion.provider_name == "openrouter_route"
    assert completion.usage.model_dump()["completion_tokens"] == 20
    assert set(completion.allowed_served_models) == {
        "deepseek/deepseek-v4.1-flash",
        "deepseek/deepseek-v4.1-flash-20260910",
    }


def test_composite_pool_reports_each_independent_capacity() -> None:
    def completion(name):
        return SimpleNamespace(provider_name=name)

    local = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion("local"))
        ),
        snapshot=lambda: {"name": "local"},
    )
    paid = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: completion("paid"))
        ),
        snapshot=lambda: {"name": "paid"},
    )
    pool = workflow._CompositeCompletionPool(
        (("local", local, 1), ("openrouter", paid, 2))
    )

    snapshot = pool.snapshot()
    assert [row["name"] for row in snapshot] == ["local", "openrouter"]
    assert [row["capacity"] for row in snapshot] == [1, 2]


def test_selector_profile_limit_keeps_full_population_statistics(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "PAIR_COLUMNS", {"source": ("used", "candidate")})
    monkeypatch.setattr(
        workflow, "REFINEMENT_COLUMNS", {"source": ("used", "candidate")}
    )
    monkeypatch.setattr(workflow, "COLUMN_DESCRIPTIONS", {"candidate": "candidate"})
    monkeypatch.setattr(workflow, "SELECTOR_PROFILE_LIMIT", 1)
    lookup = {
        "a": {"values": {"used": "x", "candidate": "one"}, "record_count": 1},
        "b": {"values": {"used": "x", "candidate": "two"}, "record_count": 1},
        "c": {"values": {"used": "y", "candidate": "three"}, "record_count": 1},
    }

    payload = workflow._column_selection_payload(
        "L2", "source", {"bucket-a": ["a", "b"], "bucket-b": ["c"]}, lookup, ["used"]
    )

    assert payload["active_bucket_count"] == 2
    assert payload["profiled_bucket_count"] == 1
    assert payload["candidate_columns"][0]["buckets_with_multiple_values"] == 1
    assert payload["candidate_columns"][0]["maximum_distinct_values_in_one_bucket"] == 2
