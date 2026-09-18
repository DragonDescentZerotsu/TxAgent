"""Executable pooled-L5 and deterministic-rank checks for cache v3."""
import json
from pathlib import Path
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from predict.retrieval.assay_reranking import cache_matched
from predict.retrieval.assay_reranking.build_cache_matched_v2 import _schema
from predict.retrieval.assay_reranking.build_cache_matched_v3 import (
    _rank_pool,
    build_l1_successor,
)
from predict.retrieval.assay_reranking.cache_matched_v3 import SCHEMA_VERSION
from predict.retrieval.assay_reranking import v9
from predict.utils.json import sha256_file


def _payload(record_id):
    return json.dumps({
        "record_id": record_id,
        "source_row_uid": f"uid-{record_id}",
        "source_id": "toy",
        "task_id": "bbb_martins",
        "progressive_level": "L1" if record_id == "l1" else "L5",
        "family_key": "toy",
        "canonical_smiles": "CCN",
        "measurement_kind": "continuous",
        "source_fields": {"measurement_text": record_id},
        "source_contract": {
            "contract_version": "source_column_contract.v1",
            "source_or_simply_cleaned": {"measurement_text": True},
        },
    })


def test_v3_l5_uses_the_selected_pool_and_sqlite_only(tmp_path, monkeypatch):
    database = tmp_path / "retrieval.sqlite3"
    with sqlite3.connect(database) as connection:
        _schema(connection)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", SCHEMA_VERSION), ("content_id", "fixture"),
        ])
        connection.execute("INSERT INTO queries VALUES (1,'query-parent','CCO')")
        connection.execute("INSERT INTO benchmark_queries VALUES ('q','CCO',1)")
        connection.execute("INSERT INTO records VALUES (1,'l1','L1',?)", (_payload("l1"),))
        connection.execute(
            "INSERT INTO assignments VALUES (1,'fixed','L1','p1','CCN',1,1,.5,1,.6,1)"
        )
        connection.execute("INSERT INTO selection_counts VALUES (1,'fixed','L1',1,1)")
        for key, pool in enumerate(("tool-accepted", "tool-compatible", "all"), 2):
            record_id = f"l5-{pool}"
            connection.execute(
                "INSERT INTO records VALUES (?,?,?,?)",
                (key, record_id, "L5", _payload(record_id)),
            )
            connection.execute(
                "INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (1, pool, "L5", f"p{key}", "CCN", key, 1, .4, 1, None, None),
            )
            connection.execute(
                "INSERT INTO selection_counts VALUES (?,?,?,?,?)", (1, pool, "L5", 1, 1)
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "gold_release": "v2",
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "database": database.name,
        "content_id": "fixture",
        "tie_seed": 0,
        "pools": ["tool-accepted", "tool-compatible", "all"],
        "fixed_pool_levels": ["L1"],
        "joint_materialized": False,
        "assignment_counts": {},
        "record_count": 4,
        "capacities": {
            "l1_molecules": 1,
            "l1_records_per_molecule": 1,
            "later_records_per_level": 1,
        },
    }
    version = tmp_path / "VERSION.json"
    version.write_text(json.dumps(manifest))
    policy = {
        "selection_contract": SCHEMA_VERSION,
        "stages": {"L1": "morgan", "L5": "morgan"},
        "cache_manifest": str(version),
        "inputs": {},
    }
    monkeypatch.setattr(
        cache_matched, "sha256_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("v3 runtime must not hash or rebuild sources")
        ),
    )
    for pool in ("tool-accepted", "tool-compatible", "all"):
        _, later, audit = cache_matched.load_candidates(
            {"q": "CCO"}, task="bbb_martins", subset="valid", policy=policy,
            molecule_limit=1, l1_limit=1, later_limit=1, cache_pool=pool,
        )
        row = later["q"]["L5"]["records"][0]
        assert row["record_id"] == f"l5-{pool}"
        assert row["assay_rank"] is None
        assert "transfer_likelihood" not in row
        assert audit["selection_policy"] == SCHEMA_VERSION


def test_v3_same_parent_ties_are_deterministic_unique_record_ranks():
    groups = {
        "same-parent": [
            {"record_id": "r2", "canonical_smiles": "CCN"},
            {"record_id": "r1", "canonical_smiles": "CCN"},
        ],
        "lower-parent": [{"record_id": "r3", "canonical_smiles": "CCC"}],
    }
    first = _rank_pool(
        groups, {"same-parent": .8, "lower-parent": .2},
        task="bbb_martins", benchmark_row_id="q", limit=3,
    )
    second = _rank_pool(
        groups, {"same-parent": .8, "lower-parent": .2},
        task="bbb_martins", benchmark_row_id="q", limit=3,
    )
    assert [(row["payload"]["record_id"], row["morgan_rank"]) for row in first] == [
        (row["payload"]["record_id"], row["morgan_rank"]) for row in second
    ]
    assert [row["similarity"] for row in first[:2]] == [.8, .8]
    assert [row["morgan_rank"] for row in first] == [1, 2, 3]
    assert [row["within_parent_rank"] for row in first[:2]] == [1, 2]


def test_l1_successor_replaces_only_l1_assay_ranks(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    database = base / "retrieval.sqlite3"
    with sqlite3.connect(database) as connection:
        _schema(connection)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", SCHEMA_VERSION), ("content_id", "base"),
        ])
        connection.execute("INSERT INTO queries VALUES (1,'query-parent','CCO')")
        connection.execute("INSERT INTO benchmark_queries VALUES ('q','CCO',1)")
        for index in range(100):
            key = index + 1
            parent = f"p{index:03d}"
            connection.execute(
                "INSERT INTO records VALUES (?,?,?,?)", (key, f"r{index}", "L1", _payload("l1"))
            )
            connection.execute(
                "INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (1, "fixed", "L1", parent, "CCN", key, 1, index / 100, key, .1, key),
            )
        connection.execute("INSERT INTO records VALUES (101,'later','L2',?)", (_payload("later"),))
        connection.execute(
            "INSERT INTO assignments VALUES (1,'all','L2','later-parent','CCC',101,1,.2,1,.7,1)"
        )
    base_manifest = base / "VERSION.json"
    base_manifest.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "gold_release": "v2", "status": "complete",
        "task_id": "bbb_martins", "subset": "valid", "content_id": "base",
        "database": database.name, "database_sha256": sha256_file(database),
        "inputs": {}, "legacy_selection_audit": {},
    }))

    ranking_root = tmp_path / "ranking"
    ranking_root.mkdir()
    ranking_rows = [{
        "query_record_id": "q", "retrieval_record_id": f"gold-{index}",
        "retrieval_molecule_identity_key": f"p{index:03d}",
        "morgan_tanimoto_similarity": index / 100,
        "model_rank": 99 - index, "prob_transfer": index / 100,
    } for index in range(100)]
    ranking_rows[0]["morgan_tanimoto_similarity"] = .123
    rankings = ranking_root / "rankings.parquet"
    pq.write_table(pa.Table.from_pylist(ranking_rows), rankings)
    ranking_manifest = ranking_root / "VERSION.json"
    ranking_manifest.write_text(json.dumps({
        "schema_version": v9.RANKING_SCHEMA_VERSION, "status": "complete",
        "task_id": "bbb_martins", "model_lineage": "v10_3",
        "model": v9.model_profile("bbb_martins", "v10_3"),
        "prompt_assets": v9.verify_vendored_assets(),
        "reference_provenance": v9.reference_provenance("bbb_martins", "v10_3"),
        "rankings_sha256": sha256_file(rankings), "morgan_pool_size": 100,
        "candidate_policy": "morgan_top100_distinct_gold_train_parents_then_all_context_rows",
    }))

    result = build_l1_successor(
        task="bbb_martins", subset="valid", base_manifest=base_manifest,
        ranking_manifest=ranking_manifest, output_root=tmp_path / "successor",
    )
    with sqlite3.connect(tmp_path / "successor/retrieval.sqlite3") as connection:
        later = connection.execute(
            "SELECT assay_transfer_score,assay_rank FROM assignments WHERE level='L2'"
        ).fetchone()
        best = connection.execute(
            "SELECT parent_id,assay_transfer_score FROM assignments "
            "WHERE level='L1' AND assay_rank=1"
        ).fetchone()
    assert later == (.7, 1)
    assert best == ("p099", .99)
    assert result["l1_successor"]["contract"].endswith(".v2")
    assert result["l1_successor"]["ranking_context_similarity_mismatch_count"] == 1
