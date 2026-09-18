"""Executable contract tests for the SQLite-only cache-matched v2 reader."""
import json
import sqlite3

import pytest

from predict.retrieval.assay_reranking import cache_matched
from predict.retrieval.assay_reranking import build_cache_matched_v2 as publisher
from predict.retrieval.assay_reranking.build_cache_matched_v2 import _schema
from predict.retrieval.assay_reranking.build_cache_matched_v2 import _rank_later_rows
from predict.retrieval.assay_reranking.cache_matched_v2 import POOLS, SCHEMA_VERSION
from predict.retrieval.assay_reranking.cache_matched_v2 import _select_l1
from predict.retrieval.assay_reranking.cache_matched_v2 import _require_current_gold
from predict.retrieval import policies
from predict.retrieval.policies import normalize_molecule_identity


def _payload(record_id, level, smiles):
    return json.dumps({
        "record_id": record_id,
        "source_row_uid": "uid-" + record_id,
        "source_id": "toy",
        "task_id": "bbb_martins",
        "progressive_level": level,
        "family_key": "toy_family",
        "canonical_smiles": smiles,
        "measurement_kind": "continuous",
        "source_fields": {"smiles": smiles, "measurement_text": record_id},
        "source_contract": {
            "contract_version": "source_column_contract.v2",
            "source_or_simply_cleaned": {"smiles": True, "measurement_text": True},
        },
    })


def test_cache_manifest_fails_closed_for_superseded_gold():
    with pytest.raises(ValueError, match="Cache is stale for active BBB_Martins v2"):
        _require_current_gold({
            "inputs": {"query_ledger": {
                "path": "/repo/data/gold_labels/BBB_Martins/v1/scaffold/valid.jsonl"
            }}
        }, "bbb_martins")
    _require_current_gold({"gold_release": "v2"}, "bbb_martins")


@pytest.fixture
def v2_policy(tmp_path):
    database = tmp_path / "retrieval.sqlite3"
    with sqlite3.connect(database) as connection:
        _schema(connection)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", SCHEMA_VERSION), ("content_id", "fixture"),
        ])
        connection.execute("INSERT INTO queries VALUES (1,'query-parent-id','query-parent')")
        connection.execute("INSERT INTO benchmark_queries VALUES ('q','raw-query',1)")
        connection.execute("INSERT INTO selection_counts VALUES (1,'fixed','L1',6,6)")
        record_key = 0
        for index in range(1, 7):
            record_key += 1
            record_id = f"l1-{index}"
            connection.execute(
                "INSERT INTO records VALUES (?,?,?,?)",
                (record_key, record_id, "L1", _payload(record_id, "L1", f"source-{index}")),
            )
            connection.execute(
                "INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (1, "fixed", "L1", f"parent-{index}", f"parent-smiles-{index}",
                 record_key, 1, 1 - index / 10, index, index / 10, 7 - index),
            )
        for index, (similarity, assay_score) in enumerate(((0.2, 0.9), (0.8, 0.1)), 1):
            record_key += 1
            record_id = f"l2-{index}"
            connection.execute(
                "INSERT INTO records VALUES (?,?,?,?)",
                (record_key, record_id, "L2", _payload(record_id, "L2", f"conflicting-source-{index}")),
            )
            for pool in POOLS:
                connection.execute(
                    "INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (1, pool, "L2", "shared-parent", "cached-parent-smiles", record_key,
                     index, similarity, 3 - index, assay_score, index),
                )
        connection.executemany(
            "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
            [(1, pool, "L2", 2, 1) for pool in POOLS],
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
        "pools": list(POOLS),
        "joint_materialized": False,
        "assignment_counts": {"fixed/L1": 6, "all/L2": 2},
        "record_count": 8,
        "capacities": {
            "l1_molecules": 10,
            "l1_records_per_molecule": 10,
            "later_records_per_level": 50,
        },
    }
    manifest_path = tmp_path / "VERSION.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "selection_contract": SCHEMA_VERSION,
        "reranking": "joint",
        "stages": {"L1": "joint", "L2": "assay_transfer"},
        "cache_manifest": str(manifest_path),
        "inputs": {},
    }


def test_v2_is_sqlite_only_and_joint_merges_without_refill(v2_policy, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("v2 runtime must not inspect source files or recompute chemistry")

    monkeypatch.setattr(cache_matched, "sha256_file", forbidden)
    monkeypatch.setattr(policies, "normalize_molecule_identity", forbidden)
    monkeypatch.setattr(policies, "standardize_smiles_and_fp", forbidden)
    molecules, later, audit = cache_matched.load_candidates(
        {"q": "raw-query"}, task="bbb_martins", subset="valid",
        library="does-not-exist", mapper="does-not-exist", policy=v2_policy,
        molecule_limit=10, l1_limit=10, later_limit=2, cache_pool="all",
    )
    assert len(molecules["q"]) == 6  # top five from each native list; overlap gets no refill
    assert molecules["q"][0]["morgan_rank"] == 1
    assert molecules["q"][0]["assay_rank"] == 6
    assert [row["record_id"] for row in later["q"]["L2"]["records"]] == ["l2-1", "l2-2"]
    assert {row["morgan_rank"] for row in later["q"]["L2"]["records"]} == {1, 2}
    assert [row["assay_rank"] for row in later["q"]["L2"]["records"]] == [1, 2]
    assert {row["payload"]["canonical_smiles"] for row in later["q"]["L2"]["records"]} == {
        "cached-parent-smiles"
    }
    assert {row["morgan_similarity"] for row in later["q"]["L2"]["records"]} == {0.2, 0.8}
    assert audit["selection_policy"] == SCHEMA_VERSION
    assert audit["query_identities"]["q"] == {
        "parent_id": "query-parent-id",
        "parent_smiles": "query-parent",
    }


def test_asymmetric_joint_is_assay_first_and_preserves_native_ranks():
    connection = sqlite3.connect(":memory:")
    _schema(connection)
    connection.execute("INSERT INTO queries VALUES (1,'query-parent','CCO')")
    connection.execute("INSERT INTO selection_counts VALUES (1,'fixed','L1',10,10)")
    morgan_order = [3, 4, 5, 6, 7, 8, 9, 10, 2, 1]
    morgan_ranks = {parent: rank for rank, parent in enumerate(morgan_order, 1)}
    for parent in range(1, 11):
        record_id = f"r{parent}"
        connection.execute(
            "INSERT INTO records VALUES (?,?,?,?)",
            (parent, record_id, "L1", _payload(record_id, "L1", f"C{parent}")),
        )
        connection.execute(
            "INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (1, "fixed", "L1", f"p{parent}", f"C{parent}", parent, 1,
             1 - parent / 100, morgan_ranks[parent], 1 - parent / 100, parent),
        )

    asymmetric, audit = _select_l1(
        connection, 1, "joint", 10, 1, joint_panel_sizes=(3, 7)
    )
    assert [row["reference_molecule_id"] for row in asymmetric] == [
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"
    ]
    assert asymmetric[0]["assay_transfer_panel_rank"] == 1
    assert asymmetric[0]["morgan_panel_rank"] == "not_selected_in_panel"
    assert asymmetric[2]["assay_transfer_panel_rank"] == 3
    assert asymmetric[2]["morgan_panel_rank"] == 1
    assert audit["joint_panel_order"] == ["assay_transfer", "morgan"]
    assert audit["joint_panel_sizes"] == {"assay_transfer": 3, "morgan": 7}
    assert audit["joint_overlap_molecules"] == 1
    assert audit["joint_overlap_refill"] is False

    historical, _ = _select_l1(connection, 1, "joint", 10, 1)
    assert [row["reference_molecule_id"] for row in historical] == [
        "p3", "p4", "p5", "p6", "p7", "p1", "p2"
    ]
    assert "morgan_top5_rank" in historical[0]
    assert "morgan_panel_rank" not in historical[0]


def test_v2_rank_queries_use_covering_indices(v2_policy):
    manifest = json.loads(open(v2_policy["cache_manifest"]).read())
    database = __import__("pathlib").Path(v2_policy["cache_manifest"]).with_name(manifest["database"])
    with sqlite3.connect(database) as connection:
        plans = {
            method: " ".join(str(value) for row in connection.execute(
                f"EXPLAIN QUERY PLAN SELECT record_key FROM assignments "
                f"WHERE query_id=1 AND pool='all' AND level='L2' ORDER BY {column}",
            ) for value in row)
            for method, column in (("morgan", "morgan_rank"), ("assay", "assay_rank"))
        }
    assert "assignments_morgan_rank" in plans["morgan"]
    assert "assignments_assay_rank" in plans["assay"]


def test_publisher_ranks_parent_scoped_records_and_retains_native_windows():
    parent_id = normalize_molecule_identity("CCO").parent_inchi_key
    rows = [
        ("r1", parent_id, {
            "canonical_smiles": "CCO",
            "identity_smiles_forms": ["CCO"],
        }, 0.1),
        ("r2", parent_id, {
            "canonical_smiles": "C(C)O",
            "identity_smiles_forms": ["C(C)O"],
        }, 0.9),
    ]
    ranked = _rank_later_rows(
        rows, query_smiles="CCN", task="bbb_martins",
        benchmark_row_id="q", level="L2",
    )
    assert {row["parent_smiles"] for row in ranked} == {"CCO"}
    assert len({row["similarity"] for row in ranked}) == 1
    assert {row["morgan_rank"] for row in ranked} == {1, 2}
    assert {row["assay_rank"] for row in ranked} == {1, 2}


def test_publisher_writes_a_complete_directly_readable_database(tmp_path, monkeypatch):
    query = {"benchmark_row_id": "q", "drug": "CCO"}
    compact = tmp_path / "valid.jsonl"
    detailed = tmp_path / "valid_molecule_condition_labels.jsonl"
    compact.write_text(json.dumps(query) + "\n")
    detailed.write_text(json.dumps(query) + "\n")
    monkeypatch.setattr(publisher, "split_path", lambda task, subset: compact)
    monkeypatch.setattr(publisher, "L1_MOLECULE_CAPACITY", 1)
    monkeypatch.setattr(publisher, "L1_RECORD_CAPACITY", 1)
    monkeypatch.setattr(publisher, "LATER_RECORD_CAPACITY", 1)

    legacy_bundle = tmp_path / "legacy.yaml"
    legacy_bundle.write_text("fixture\n")
    later_root = tmp_path / "later"
    later_root.mkdir()
    later_manifest = later_root / "VERSION.json"
    later_manifest.write_text(json.dumps({
        "schema_version": "assay_transfer_three_pools.v1",
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "models": {"L2": {}},
        "cache_sha256": "frozen-fixture",
    }))
    parent = normalize_molecule_identity("CCN")
    parent_id = parent.parent_inchi_key
    source = json.loads(_payload("l2", "L2", parent.parent_smiles))
    source["identity_smiles_forms"] = [parent.parent_smiles]
    with sqlite3.connect(later_root / "scores.sqlite3") as connection:
        connection.executescript("""
            CREATE TABLE benchmark_queries(query_id,benchmark_row_id,drug);
            CREATE TABLE records(record_key,external_record_id,parent_id,payload);
            CREATE TABLE scores(score_key,transfer_probability);
            CREATE TABLE assignments(query_id,pool,level,record_key,score_key);
        """)
        connection.execute("INSERT INTO benchmark_queries VALUES (1,'q','CCO')")
        connection.execute(
            "INSERT INTO records VALUES (1,'l2',?,?)", (parent_id, json.dumps(source))
        )
        connection.execute("INSERT INTO scores VALUES (1,0.75)")
        connection.executemany(
            "INSERT INTO assignments VALUES (1,?,'L2',1,1)",
            [(pool,) for pool in POOLS],
        )
    l1_payload = json.loads(_payload("l1", "L1", parent.parent_smiles))
    l5_payload = json.loads(_payload("l5", "L5", parent.parent_smiles))
    legacy_result = (
        {"q": [{
            "reference_molecule_id": parent_id,
            "canonical_smiles": parent.parent_smiles,
            "morgan_top5_rank": 1,
            "assay_transfer_top5_rank": 1,
            "morgan_similarity": 0.5,
            "transfer_likelihood": 0.8,
            "l1_records": [{"payload": l1_payload}],
        }]},
        {"q": {"L5": {"records": [{
            "reference_molecule_id": parent_id,
            "reference_parent_smiles": parent.parent_smiles,
            "morgan_similarity": 0.5,
            "payload": l5_payload,
        }]}}},
        {
            "selection_policy": "cache_matched_retrieval.v1",
            "inputs": {},
            "v9": {},
            "gold_context_mapping": None,
            "neighbor_identity_policy": "scaffold_disjoint",
            "_parent_smiles_by_id": {parent_id: parent.parent_smiles},
            "query_audits": {"q": {"L5": {
                "candidate_records": 1, "candidate_molecules": 1,
            }}},
        },
    )
    policy = {
        "stages": {"L1": "joint", "L2": "assay_transfer", "L5": "morgan"},
        "cache_manifests": {"L1": str(tmp_path / "L1.json"), "L2": str(later_manifest)},
    }
    monkeypatch.setattr(publisher, "load_cache_policy", lambda *args: policy)
    monkeypatch.setattr(publisher, "_load_candidates_v1", lambda *args, **kwargs: legacy_result)
    output = tmp_path / "published"
    manifest = publisher.build(
        task="bbb_martins", subset="valid", library=tmp_path,
        mapper=tmp_path / "mapper", output_root=output,
        legacy_bundle=legacy_bundle,
    )
    assert manifest["status"] == "complete"
    direct_policy = {
        "selection_contract": SCHEMA_VERSION,
        "stages": {**policy["stages"], "L1": "assay_transfer"},
        "cache_manifest": str(output / "VERSION.json"),
        "inputs": {},
    }
    molecules, later, _ = cache_matched.load_candidates(
        {"q": "CCO"}, task="bbb_martins", subset="valid",
        policy=direct_policy, molecule_limit=1, l1_limit=1, later_limit=1,
    )
    assert molecules["q"][0]["morgan_rank"] == 1
    assert later["q"]["L2"]["records"][0]["assay_rank"] == 1
    assert later["q"]["L5"]["records"][0]["morgan_rank"] == 1
