"""Selection and rendering checks for the L1 parent-condition cache."""
import copy
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from predict.retrieval.assay_reranking.cache_matched import (
    load_candidates as load_dispatched_candidates,
)
from predict.retrieval.assay_reranking.build_l1_context_cache import (
    _context_by_uid,
    _schema,
)
from predict.retrieval.assay_reranking.l1_context_cache import (
    ACTIVE_CONTEXT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    load_candidates,
)
from predict.retrieval.assay_reranking.runtime import (
    ACTIVE_CACHE_ROOT,
    cache_profile_root,
)
from predict.utils.json import sha256_file


def test_v10_4_l1_width_profiles_publish_as_active():
    for width in (25, 50, 100):
        profile = f"l1_context_morgan{width}_v10_4_gold_v2_v1"
        assert cache_profile_root(profile).parent == ACTIVE_CACHE_ROOT


def test_v10_4_uses_exact_gold_v2_voter_membership(tmp_path):
    membership = tmp_path / "voter_membership.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "task": "BBB_Martins",
        "source_row_uid": "uid-1",
        "molecule_identity_key": "parent-1",
        "condition_group": "condition-1",
        "condition_atoms": ["atom-1"],
        "aggregate_label": 1,
        "benchmark_row_id": "context-1",
        "split": "train",
        "aggregate_status": "published",
    }]), membership)
    manifest = tmp_path / "voter_membership.manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "gold_voter_membership.v1",
        "path": membership.name,
        "sha256": sha256_file(membership),
    }))
    gold = {"context-1": {
        "molecule_identity_key": "parent-1",
        "condition_group": "condition-1",
        "condition_atoms": ["atom-1"],
        "Y": 1,
    }}
    mapped, retired, provenance = _context_by_uid(
        "bbb_martins", gold,
        {"voter_membership": membership, "voter_membership_manifest": manifest},
        "v10_4",
    )
    assert mapped == {"uid-1": "context-1"}
    assert retired == set()
    assert "v10_stage3" in provenance

    mismatched = copy.deepcopy(gold)
    mismatched["context-1"]["Y"] = 0
    with pytest.raises(ValueError, match="voter identity"):
        _context_by_uid(
            "bbb_martins", mismatched,
            {"voter_membership": membership, "voter_membership_manifest": manifest},
            "v10_4",
        )


def _payload(record_id: str) -> str:
    return json.dumps({
        "record_id": record_id,
        "source_row_uid": f"uid-{record_id}",
        "source_id": "direct_bbb",
        "task_id": "bbb_martins",
        "progressive_level": "L1",
        "family_key": "direct_brain_exposure",
        "canonical_smiles": "CCN",
        "measurement_kind": "continuous",
        "source_fields": {"measurement_text": record_id},
        "source_contract": {
            "contract_version": "source_column_contract.v1",
            "source_or_simply_cleaned": {"measurement_text": True},
        },
    })


def _fixture(
    tmp_path, *, fallback_only=False, primary_width=25,
    candidate_widths=None, labels=None, schema_version=SCHEMA_VERSION,
):
    database = tmp_path / "retrieval.sqlite3"
    with sqlite3.connect(database) as connection:
        _schema(connection)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", schema_version), ("content_id", "fixture"),
        ])
        connection.execute("INSERT INTO queries VALUES (1,'query-parent','CCO')")
        connection.execute("INSERT INTO benchmark_queries VALUES ('q','CCO',1)")
        labels = labels or [1, 1, 1, 1, 1, 0, 0]
        for key, label in enumerate(labels, 1):
            parent = "shared-parent" if key in {1, 2} else f"p{key}"
            condition = f"condition-{key}"
            connection.execute(
                "INSERT INTO gold_contexts VALUES (?,?,?,?,?,?,?,?,?)",
                (key, f"ctx-{key}", parent, "CCN", condition, "external",
                 "[]", label, 1),
            )
            connection.execute(
                "INSERT INTO records VALUES (?,?,?)", (key, f"r{key}", _payload(f"r{key}"))
            )
            connection.execute("INSERT INTO context_records VALUES (?,?,1)", (key, key))
            parent_rank = 26 if key == 7 and fallback_only else key
            connection.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?,?)",
                (1, key, parent_rank, .8 - key / 100, 1 - key / (len(labels) + 1),
                 key if parent_rank <= 25 else None, key),
            )
    manifest = {
        "schema_version": schema_version,
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "database": database.name,
        "content_id": "fixture",
        "tie_seed": 0,
        "morgan_primary_parent_width": primary_width,
        "morgan_fallback_parent_width": 100,
        "capacities": {"l1_contexts": len(labels), "voter_records_per_context": 1},
        "assignment_counts": {},
        "record_count": len(labels),
    }
    if candidate_widths is not None:
        manifest["morgan_candidate_parent_widths"] = candidate_widths
    version = tmp_path / "VERSION.json"
    version.write_text(json.dumps(manifest))
    return {
        "selection_contract": schema_version,
        "reranking": "assay-transfer-contrastive",
        "stages": {"L1": "assay_transfer_contrastive"},
        "cache_manifest": str(version),
        "inputs": {},
    }


def test_active_context_v2_m0_stays_inside_compacted_top10(tmp_path):
    policy = _fixture(
        tmp_path,
        primary_width=10,
        candidate_widths=[10, 15, 25, 100],
        labels=[1] * 10 + [0, 0],
        schema_version=ACTIVE_CONTEXT_SCHEMA_VERSION,
    )
    molecules, _, audit = load_dispatched_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=10, l1_limit=1, min_contrast=0,
    )
    selected = audit["query_audits"]["q"]["L1"]
    assert len(molecules["q"]) == 10
    assert selected["replacement_count"] == 0
    assert not selected["fallback_beyond_primary"]
    assert selected["selected_unique_parents"] == 9
    assert selected["selected_repeated_parent_cards"] == 1


def test_l1_only_accepts_per_level_limit_mapping(tmp_path):
    policy = _fixture(tmp_path, labels=[1, 1, 1, 0, 0, 0])

    molecules, later, _ = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=3, l1_limit=1, later_limit={}, min_contrast=1,
    )

    assert len(molecules["q"]) == 3
    assert later == {"q": {}}


def test_active_context_v2_uses_wider_pool_only_for_label_balance(tmp_path):
    policy = _fixture(
        tmp_path,
        primary_width=10,
        candidate_widths=[10, 15, 25, 100],
        labels=[1] * 10 + [0, 0],
        schema_version=ACTIVE_CONTEXT_SCHEMA_VERSION,
    )
    _, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=10, l1_limit=1, min_contrast=2,
    )
    selected = audit["query_audits"]["q"]["L1"]
    assert selected["selected_label_counts"] == {"0": 2, "1": 8}
    assert selected["replacement_count"] == 2
    assert selected["fallback_beyond_primary"]


@pytest.mark.parametrize(
    ("primary_width", "candidate_widths"),
    [(50, [50, 100]), (100, [100])],
)
def test_active_context_v2_supports_wider_primary_pools(
    tmp_path, primary_width, candidate_widths,
):
    policy = _fixture(
        tmp_path,
        primary_width=primary_width,
        candidate_widths=candidate_widths,
        labels=[1] * 6 + [0],
        schema_version=ACTIVE_CONTEXT_SCHEMA_VERSION,
    )
    _, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=6, l1_limit=1, min_contrast=1,
    )
    selected = audit["query_audits"]["q"]["L1"]
    assert selected["morgan_primary_parent_width"] == primary_width
    assert selected["morgan_candidate_parent_widths"] == candidate_widths
    assert selected["selected_label_counts"] == {"0": 1, "1": 5}












def test_contrastive_selection_prefers_top25_and_keeps_context_cards_separate(tmp_path):
    policy = _fixture(tmp_path)
    molecules, later, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=6, l1_limit=1, min_contrast=2,
    )
    selected = molecules["q"]
    assert later == {"q": {}}
    assert [row["context_card_id"] for row in selected] == [
        "ctx-1", "ctx-2", "ctx-3", "ctx-4", "ctx-6", "ctx-7"
    ]
    assert audit["query_audits"]["q"]["L1"]["selected_label_counts"] == {
        "0": 2, "1": 4
    }
    assert not audit["query_audits"]["q"]["L1"]["fallback_beyond_primary"]





































































































def test_contrastive_selection_uses_top100_only_after_top25_is_exhausted(tmp_path):
    policy = _fixture(tmp_path, fallback_only=True)
    _, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=6, l1_limit=1, min_contrast=2,
    )
    assert audit["query_audits"]["q"]["L1"]["fallback_beyond_primary"]


def test_morgan100_uses_the_whole_primary_pool_without_fallback(tmp_path):
    policy = _fixture(tmp_path, fallback_only=True, primary_width=100)
    _, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=6, l1_limit=1, min_contrast=2,
    )
    selection = audit["query_audits"]["q"]["L1"]
    assert selection["morgan_primary_parent_width"] == 100
    assert not selection["fallback_beyond_primary"]


def test_morgan_context_selection_uses_parent_rank_not_transfer_score(tmp_path):
    policy = _fixture(tmp_path)
    policy.update(reranking="morgan", stages={"L1": "morgan"})
    database = policy["cache_manifest"].replace("VERSION.json", "retrieval.sqlite3")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE candidates SET transfer_score=.999 WHERE context_key=7")
    molecules, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=3, l1_limit=1,
    )
    assert [row["context_card_id"] for row in molecules["q"]] == [
        "ctx-1", "ctx-2", "ctx-3"
    ]
    assert all(row["ranking_method"] == "morgan" for row in molecules["q"])
    assert audit["contract"]["morgan_primary_parent_width"] == 25


def test_assay_transfer_reorders_only_the_frozen_morgan_top_k(tmp_path):
    morgan_policy = _fixture(tmp_path)
    morgan_policy.update(reranking="morgan", stages={"L1": "morgan"})
    database = morgan_policy["cache_manifest"].replace(
        "VERSION.json", "retrieval.sqlite3"
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE candidates SET transfer_score=context_key/10.0"
        )
        connection.execute(
            "UPDATE candidates SET transfer_score=.999 WHERE context_key=7"
        )
    assay_policy = dict(morgan_policy)
    assay_policy.update(
        reranking="assay-transfer-within-morgan",
        stages={"L1": "assay_transfer_within_morgan"},
    )
    morgan, _, _ = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid",
        policy=morgan_policy, molecule_limit=6, l1_limit=1,
    )
    assay, _, _ = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid",
        policy=assay_policy, molecule_limit=6, l1_limit=1,
    )
    morgan_ids = [row["context_card_id"] for row in morgan["q"]]
    assay_ids = [row["context_card_id"] for row in assay["q"]]
    assert morgan_ids == [f"ctx-{index}" for index in range(1, 7)]
    assert assay_ids == [f"ctx-{index}" for index in range(6, 0, -1)]
    assert set(assay_ids) == set(morgan_ids)
    assert "ctx-7" not in assay_ids

    rendered = []



























def test_morgan_contrastive_rebalances_by_morgan_rank_and_hides_transfer(tmp_path):
    policy = _fixture(tmp_path)
    policy.update(
        reranking="morgan-contrastive", stages={"L1": "morgan_contrastive"}
    )
    database = policy["cache_manifest"].replace("VERSION.json", "retrieval.sqlite3")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE candidates SET transfer_score=1-transfer_score")
    molecules, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=6, l1_limit=1, min_contrast=2,
    )
    selected = molecules["q"]
    assert [row["context_card_id"] for row in selected] == [
        "ctx-1", "ctx-2", "ctx-3", "ctx-4", "ctx-6", "ctx-7"
    ]
    assert audit["query_audits"]["q"]["L1"]["selected_label_counts"] == {
        "0": 2, "1": 4
    }























def test_contrastive_shape_fails_closed(tmp_path):
    policy = _fixture(tmp_path)
    with pytest.raises(ValueError, match=r"K >= 2\*M"):
        load_candidates(
            {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
            molecule_limit=5, l1_limit=1, min_contrast=3,
        )


def test_zero_contrast_is_exact_unbalanced_assay_transfer_top_k(tmp_path):
    contrastive = _fixture(tmp_path)
    plain = dict(contrastive)
    plain.update(reranking="assay-transfer", stages={"L1": "assay_transfer"})
    selected = []
    for policy in (contrastive, plain):
        molecules, _, audit = load_candidates(
            {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
            molecule_limit=6, l1_limit=1, min_contrast=0,
        )
        selected.append([row["context_card_id"] for row in molecules["q"]])
        assert audit["query_audits"]["q"]["L1"]["replacement_count"] == 0
    assert selected[0] == selected[1]


def test_assay_transfer_fallback_uses_top25_before_higher_scored_top100(tmp_path):
    policy = _fixture(
        tmp_path, primary_width=15, candidate_widths=[15, 25, 100],
        labels=[1] * 6 + [0, 0],
    )
    database = policy["cache_manifest"].replace("VERSION.json", "retrieval.sqlite3")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE candidates SET morgan_parent_rank=20,transfer_score=.20 "
            "WHERE context_key=7"
        )
        connection.execute(
            "UPDATE candidates SET morgan_parent_rank=26,transfer_score=.99 "
            "WHERE context_key=8"
        )
    molecules, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=6, l1_limit=1, min_contrast=1,
    )
    assert "ctx-7" in [row["context_card_id"] for row in molecules["q"]]
    assert "ctx-8" not in [row["context_card_id"] for row in molecules["q"]]
    assert audit["query_audits"]["q"]["L1"]["replacements"][0][
        "added_parent_width"
    ] == 25


def test_maximum_contrast_produces_exact_balance(tmp_path):
    policy = _fixture(tmp_path, labels=[1] * 8 + [0] * 5)
    _, _, audit = load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=10, l1_limit=1, min_contrast=5,
    )
    assert audit["query_audits"]["q"]["L1"]["selected_label_counts"] == {
        "0": 5, "1": 5
    }


def test_negative_contrast_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        load_candidates(
            {"q": "CCO"}, task="bbb_martins", subset="valid",
            policy=_fixture(tmp_path), molecule_limit=6, l1_limit=1,
            min_contrast=-1,
        )
