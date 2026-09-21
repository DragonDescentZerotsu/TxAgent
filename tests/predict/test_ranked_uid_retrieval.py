import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from predict.retrieval.assay_reranking import cache_matched
from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder
from predict.retrieval.assay_reranking import ranked_uid_retrieval
from predict.retrieval.assay_reranking.build_ranked_uid_retrieval import finalize_level
from predict.retrieval.assay_reranking.ranked_uid_retrieval import (
    _select_assay_contrastive,
    load_candidates,
    load_ranked_panels,
    load_ranked_universe,
)
from predict.utils.json import sha256_file


def test_cache_matched_default_uses_ranked_uid_v4() -> None:
    assert cache_matched.DEFAULT_CACHE_BUNDLE.name == "ranked_level_retrieval_v4.yaml"


def test_addon_v2_profile_is_active_data_cache() -> None:
    assert builder.runtime.cache_profile_root(
        "ranked_level_retrieval_gold_v1_addon_v2"
    ).parts[-3:] == (
        "assay_reranking", "active", "ranked_level_retrieval_gold_v1_addon_v2"
    )


def test_dili_queries_recover_frozen_identity_from_drug() -> None:
    _, rows, queries = builder._queries("dili", "valid")
    assert len(queries) == len(rows)
    assert all(parent == row["molecule_identity_key"] for row, (_, _, parent, _) in zip(rows, queries))


def test_projection_reader_materializes_only_requested_level(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist([
        {
            "source_row_uid": "u2", "external_record_id": "r2",
            "parent_id": "p2", "parent_smiles": "CC", "level": "L2",
            "payload": json.dumps({"progressive_level": "L2"}),
        },
        {
            "source_row_uid": "u3", "external_record_id": "r3",
            "parent_id": "p3", "parent_smiles": "CCC", "level": "L3",
            "payload": json.dumps({"progressive_level": "L3"}),
        },
    ]), records)
    manifest = tmp_path / "VERSION.json"
    manifest.write_text(json.dumps({"records": records.name}))

    selected, grouped = builder._records(manifest, "L3")

    assert set(selected) == {"u3"}
    assert set(grouped) == {"L3"}


def test_trusted_predecessor_rows_reuse_published_score_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    level = tmp_path / "bbb_martins/scaffold/valid/L2"
    level.mkdir(parents=True)
    database = level / "rankings.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE rankings(benchmark_row_id TEXT,item_id TEXT,"
            "score_key TEXT,assay_transfer_score REAL)"
        )
        connection.execute("INSERT INTO rankings VALUES ('q','u','key',0.75)")
    (level / "VERSION.json").write_text(json.dumps({"database": database.name}))
    monkeypatch.setattr(builder, "SCORE_REUSE_ROOTS", (tmp_path,))
    monkeypatch.setattr(builder, "TRUST_PREDECESSOR_ROWS", True)

    assert builder._predecessor_rows("bbb_martins", "valid", "L2") == {
        ("q", "u"): ("key", 0.75)
    }


def test_cache_matched_wrapper_forwards_preselected_uids(monkeypatch) -> None:
    sentinel = {"q": {"L2": ["uid"]}}
    observed = {}

    def fake_load(*args, **kwargs):
        observed.update(kwargs)
        return {}, {}, {}

    monkeypatch.setattr(ranked_uid_retrieval, "load_candidates", fake_load)
    cache_matched.load_candidates(
        {"q": "CC"}, task="bbb_martins", subset="valid",
        policy={"selection_contract": "ranked_uid_retrieval.v1"},
        preselected_uids=sentinel,
    )

    assert observed["preselected_uids"] is sentinel


def _ranking(root: Path, level: str, rows: list[tuple], *, contexts: bool = False) -> Path:
    directory = root / level
    directory.mkdir(parents=True)
    database = directory / "rankings.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE queries(
                benchmark_row_id TEXT PRIMARY KEY,drug TEXT NOT NULL,
                query_parent_id TEXT NOT NULL,query_parent_smiles TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE rankings(
                benchmark_row_id TEXT NOT NULL,item_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,
                morgan_similarity REAL NOT NULL,parent_morgan_rank INTEGER NOT NULL,
                within_parent_rank INTEGER NOT NULL,morgan_rank INTEGER NOT NULL,
                assay_transfer_score REAL,assay_rank INTEGER,
                morgan_context_id TEXT,assay_context_id TEXT,
                morgan_member_count INTEGER,assay_member_count INTEGER,score_key TEXT,
                PRIMARY KEY(benchmark_row_id,item_id)
            ) WITHOUT ROWID;
            CREATE TABLE contexts(
                context_id TEXT PRIMARY KEY,parent_id TEXT NOT NULL,
                condition_group TEXT NOT NULL,gold_label INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE context_records(
                context_id TEXT NOT NULL,source_row_uid TEXT NOT NULL,
                within_context_rank INTEGER NOT NULL,
                PRIMARY KEY(context_id,within_context_rank)
            ) WITHOUT ROWID;
        """)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", "ranked_uid_retrieval.v1"),
            ("content_id", level),
        ])
        connection.execute("INSERT INTO queries VALUES ('q1','CCC','query-parent','CCC')")
        connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        if contexts:
            connection.execute("INSERT INTO contexts VALUES ('ctx','card-parent','species=rat',1)")
            connection.executemany("INSERT INTO context_records VALUES (?,?,?)", [
                ("ctx", "u1", 1), ("ctx", "u2", 2),
            ])
    manifest = directory / "VERSION.json"
    manifest.write_text(json.dumps({
        "schema_version": "ranked_uid_retrieval.v1",
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "level": level,
        "pool": "fixed" if level == "L1" else "all",
        "parent_capacity": 100,
        "content_id": level,
        "database": database.name,
        "query_counts": {"q1": {"candidate_parents": 2, "candidate_records": len(rows)}},
    }))
    return manifest


def _release(tmp_path: Path) -> tuple[dict, Path]:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    records = evidence_dir / "records.parquet"
    pq.write_table(pa.Table.from_pylist([
        {
            "source_row_uid": f"u{index}",
            "external_record_id": f"r{index}",
            "payload": json.dumps({"canonical_smiles": f"physical-{index}"}),
        }
        for index in range(1, 111)
    ]), records)
    evidence_manifest = evidence_dir / "VERSION.json"
    evidence_manifest.write_text(json.dumps({
        "schema_version": "ranked_evidence_projection.v1",
        "status": "complete", "task_id": "bbb_martins",
        "content_id": "evidence", "records": records.name,
    }))
    l1 = _ranking(tmp_path, "L1", [
        ("q1", "card-parent", "card-parent", "CCO", 0.9, 1, 1, 1, 0.8, 1,
         "ctx", "ctx", 2, 2, None),
    ], contexts=True)
    l2 = _ranking(tmp_path, "L2", [
        ("q1", "u3", "p1", "CCN", 0.8, 1, 1, 1, 0.2, 5, None, None, None, None, "k3"),
        ("q1", "u4", "p1", "CCN", 0.8, 1, 2, 2, 0.9, 1, None, None, None, None, "k4"),
        ("q1", "u5", "p1", "CCN", 0.8, 1, 3, 3, 0.4, 3, None, None, None, None, "k5"),
        ("q1", "u6", "p2", "CO", 0.7, 2, 1, 4, 0.8, 2, None, None, None, None, "k6"),
        ("q1", "u7", "p2", "CO", 0.7, 2, 2, 5, 0.3, 4, None, None, None, None, "k7"),
    ])
    index = tmp_path / "RELEASE_INDEX.json"
    index.write_text(json.dumps({
        "schema_version": "ranked_uid_task_release_index.v1",
        "status": "complete", "task_id": "bbb_martins",
        "profile": "ranked_level_retrieval_v3", "gold_release": "v1",
        "pool": "all", "parent_capacity": 100,
        "later_candidate_universe": "all_uids_under_morgan_top_100_parents",
        "evidence": {
            "manifest": "evidence/VERSION.json", "content_id": "evidence",
            "manifest_sha256": sha256_file(evidence_manifest),
        },
        "splits": {"valid": {"levels": {
            "L1": {
                "manifest": str(l1.relative_to(tmp_path)),
                "manifest_sha256": sha256_file(l1), "content_id": "L1",
            },
            "L2": {
                "manifest": str(l2.relative_to(tmp_path)),
                "manifest_sha256": sha256_file(l2), "content_id": "L2",
            },
        }}},
        "neighbor_identity_policy_by_level": {
            "L1": "scaffold_disjoint", "L2": "parent_disjoint",
        },
    }))
    return {
        "stages": {"L1": "assay_transfer", "L2": "assay_transfer"},
        "cache_manifests": {"L1": str(l1), "L2": str(l2)},
        "cache_index": str(index),
    }, l2


def test_load_ranked_universe_reads_all_rows_before_hydration(tmp_path: Path) -> None:
    policy, manifest = _release(tmp_path)

    ranked, evidence_manifest, audit = load_ranked_universe(
        Path(policy["cache_index"]), task="bbb_martins", subset="valid",
        levels=("L2",), queries={"q1": "CCC"},
    )

    assert [row["item_id"] for row in ranked["L2"]["q1"]] == [
        "u3", "u4", "u5", "u6", "u7",
    ]
    assert evidence_manifest.name == "VERSION.json"
    assert audit["parent_capacity"] == 100
    assert audit["level_manifests"]["L2"]["path"] == str(manifest.resolve())


def test_expanded_parent_universe_selects_records_then_hydrates_once(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)
    molecules, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 4}, cache_pool="all",
    )
    assert [row["record_id"] for row in molecules["q1"][0]["l1_records"]] == ["r1", "r2"]
    assert [row["record_id"] for row in later["q1"]["L2"]["records"]] == [
        "r4", "r6", "r5", "r7",
    ]
    assert later["q1"]["L2"]["available_record_count"] == 5
    assert audit["neighbor_identity_policy_by_level"]["L2"] == "parent_disjoint"
    assert audit["cache_capacities"] == {"L1": 100, "L2": 5}
    assert audit["cache_content_ids"] == {"L1": "L1", "L2": "L2"}


def test_preselected_uids_preserve_order_and_use_authoritative_payloads(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)

    _, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 2}, cache_pool="all",
        preselected_uids={"q1": {"L2": ["u6", "u3"]}},
    )

    records = later["q1"]["L2"]["records"]
    assert [row["record_id"] for row in records] == ["r6", "r3"]
    assert [row["payload"]["source_canonical_smiles"] for row in records] == [
        "physical-6", "physical-3",
    ]
    assert audit["contract"]["later_selection"] == "preselected_uid_order"


def test_preselected_uids_fail_when_not_in_query_level_universe(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)

    with pytest.raises(ValueError, match="absent from q1/L2"):
        load_candidates(
            {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
            molecule_limit=1, l1_limit=10, later_limit={"L2": 2}, cache_pool="all",
            preselected_uids={"q1": {"L2": ["u3", "u99"]}},
        )


def test_assay_records_are_reranked_inside_morgan_parent_width(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)
    _, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 3}, cache_pool="all",
        morgan_primary_parent_width=1,
    )

    assert [row["record_id"] for row in later["q1"]["L2"]["records"]] == [
        "r4", "r5", "r3",
    ]
    assert audit["contract"]["morgan_primary_parent_width"] == 1


def test_contrastive_selection_widens_only_for_missing_label() -> None:
    rows = [
        {
            "item_id": f"p{rank}", "morgan_rank": rank, "assay_rank": rank,
            "assay_context_id": f"c{rank}",
        }
        for rank in range(1, 101)
    ]
    contexts = {
        f"c{rank}": {"gold_label": 1 if rank == 16 else 0}
        for rank in range(1, 101)
    }

    selected, audit = _select_assay_contrastive(
        rows, contexts, molecule_limit=10, primary_width=15, min_contrast=1,
    )

    assert [row["morgan_rank"] for row in selected] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 16]
    assert audit["selected_label_counts"] == {"0": 9, "1": 1}
    assert audit["fallback_beyond_primary"] is True
    assert audit["replacements"][0]["effective_parent_width"] == 25


def test_contrastive_selection_uses_scored_alternate_context() -> None:
    rows = [
        {
            "item_id": f"p{rank}", "parent_id": f"p{rank}",
            "morgan_rank": rank, "assay_rank": rank,
            "assay_transfer_score": 1 - rank / 1000,
            "assay_context_id": f"positive-{rank}",
            "assay_member_count": 1, "score_key": f"positive-key-{rank}",
        }
        for rank in range(1, 101)
    ]
    contexts = {
        **{f"positive-{rank}": {"gold_label": 1} for rank in range(1, 101)},
        "negative-11": {"gold_label": 0},
    }
    selected, audit = _select_assay_contrastive(
        rows, contexts, molecule_limit=10, primary_width=25, min_contrast=1,
        context_scores=[{
            "parent_id": "p11", "context_id": "negative-11",
            "assay_transfer_score": 0.25, "assay_rank": 101,
            "member_count": 2, "score_key": "negative-key-11",
        }],
    )

    negative = next(row for row in selected if row["assay_context_id"] == "negative-11")
    assert negative["item_id"] == "p11"
    assert negative["assay_member_count"] == 2
    assert len({row["item_id"] for row in selected}) == 10
    assert audit["selected_label_counts"] == {"0": 1, "1": 9}


def test_l1_only_policy_does_not_open_later_manifest(tmp_path: Path) -> None:
    policy, later_manifest = _release(tmp_path)
    policy["stages"] = {"L1": "assay_transfer"}
    policy["cache_manifests"] = {"L1": policy["cache_manifests"]["L1"]}
    later_manifest.unlink()

    molecules, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={}, cache_pool="all",
    )

    assert len(molecules["q1"]) == 1
    assert later == {"q1": {}}
    assert audit["cache_content_ids"] == {"L1": "L1"}


def test_two_panels_share_one_deduplicated_hydration_union(tmp_path: Path) -> None:
    _, manifest = _release(tmp_path)
    panels, union = load_ranked_panels(
        manifest, task="bbb_martins", subset="valid", level="L2",
        queries={"q1": "CCC"}, morgan_limit=3, assay_limit=3,
    )
    assert [row["item_id"] for row in panels["morgan"]["q1"]] == ["u3", "u4", "u5"]
    assert [row["item_id"] for row in panels["assay_transfer"]["q1"]] == ["u4", "u6", "u5"]
    assert union["q1"] == ["u3", "u4", "u5", "u6"]


def test_request_over_available_expanded_rows_fails(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)
    with pytest.raises(ValueError, match="Requested 6 rows.*has 5"):
        load_candidates(
            {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
            molecule_limit=1, l1_limit=1, later_limit={"L2": 6}, cache_pool="all",
        )


def test_release_index_rejects_changed_level_manifest(tmp_path: Path) -> None:
    policy, manifest = _release(tmp_path)
    document = json.loads(manifest.read_text())
    document["unexpected_change"] = True
    manifest.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="L2 cache differs"):
        load_candidates(
            {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
            molecule_limit=1, l1_limit=1, later_limit={"L2": 1}, cache_pool="all",
        )


def test_later_level_can_select_more_than_one_hundred_expanded_uids(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)
    l3 = _ranking(tmp_path, "L3", [
        (
            "q1", f"u{index}", "p3", "CCF", 0.6, 1, index - 7,
            index - 7, 0.5, index - 7, None, None, None, None, f"k{index}",
        )
        for index in range(8, 109)
    ])
    policy["stages"] = {"L1": "morgan", "L3": "morgan"}
    policy["cache_manifests"] = {
        "L1": policy["cache_manifests"]["L1"], "L3": str(l3),
    }
    index_path = Path(policy["cache_index"])
    index = json.loads(index_path.read_text())
    index["splits"]["valid"]["levels"] = {
        "L1": index["splits"]["valid"]["levels"]["L1"],
        "L3": {
            "manifest": str(l3.relative_to(tmp_path)),
            "manifest_sha256": sha256_file(l3), "content_id": "L3",
        },
    }
    index["neighbor_identity_policy_by_level"] = {
        "L1": "scaffold_disjoint", "L3": "parent_disjoint",
    }
    index_path.write_text(json.dumps(index))

    _, later, _ = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=1, later_limit={"L3": 101}, cache_pool="all",
    )

    assert len(later["q1"]["L3"]["records"]) == 101


def test_levels_do_not_deduplicate_each_other(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)
    l3 = _ranking(tmp_path, "L3", [
        ("q1", "u3", "p3", "CCF", 0.6, 1, 1, 1, 0.5, 1,
         None, None, None, None, "same-uid-new-level"),
    ])
    policy["stages"] = {"L1": "morgan", "L2": "morgan", "L3": "morgan"}
    policy["cache_manifests"]["L3"] = str(l3)
    index_path = Path(policy["cache_index"])
    index = json.loads(index_path.read_text())
    index["splits"]["valid"]["levels"]["L3"] = {
        "manifest": str(l3.relative_to(tmp_path)),
        "manifest_sha256": sha256_file(l3), "content_id": "L3",
    }
    index["neighbor_identity_policy_by_level"]["L3"] = "parent_disjoint"
    index_path.write_text(json.dumps(index))

    _, later, _ = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=1, later_limit={"L2": 1, "L3": 1},
        cache_pool="all",
    )

    assert later["q1"]["L2"]["records"][0]["record_id"] == "r3"
    assert later["q1"]["L3"]["records"][0]["record_id"] == "r3"


def test_finalize_merges_each_score_key_once_and_cleans_journal(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    manifest = _ranking(
        output / "bbb_martins" / "scaffold" / "valid", "L2", [
            ("q1", "u3", "p1", "CCN", 0.8, 1, 1, 1, None, None,
             None, None, None, None, "shared"),
            ("q1", "u4", "p1", "CCN", 0.8, 1, 2, 2, None, None,
             None, None, None, None, "shared"),
            ("q1", "u5", "p2", "CO", 0.7, 2, 1, 3, None, None,
             None, None, None, None, "other"),
        ],
    )
    document = json.loads(manifest.read_text())
    document.update(status="prepared", score_counts={"exact_reuse": 0, "pending": 2})
    manifest.write_text(json.dumps(document))
    journal = manifest.parent / ".scores" / "00-of-01.jsonl"
    journal.parent.mkdir()
    journal.write_text(
        '\n'.join([
            json.dumps({"score_key": "shared", "assay_transfer_score": 0.2}),
            json.dumps({"score_key": "other", "assay_transfer_score": 0.9}),
        ]) + '\n'
    )

    result = finalize_level("bbb_martins", "valid", "L2", output)

    assert result["status"] == "complete"
    assert result["score_counts"] == {"exact_reuse": 0, "fresh": 2}
    assert not journal.parent.exists()
    with sqlite3.connect(manifest.parent / "rankings.sqlite3") as connection:
        assert connection.execute(
            "SELECT item_id,assay_transfer_score,assay_rank FROM rankings "
            "ORDER BY item_id"
        ).fetchall() == [
            ("u3", 0.2, 2), ("u4", 0.2, 3), ("u5", 0.9, 1),
        ]
