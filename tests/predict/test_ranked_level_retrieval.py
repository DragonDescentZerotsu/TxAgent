import json
from pathlib import Path
import sqlite3

import pytest
import yaml

from predict.retrieval.assay_reranking.cache_matched import load_cache_policy
from predict.retrieval.assay_reranking.ranked_level_retrieval import load_candidates


def _ranking(root: Path, level: str, method: str, rows: list[tuple]) -> Path:
    directory = root / "scaffold/valid" / level / method
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
                benchmark_row_id TEXT NOT NULL,rank INTEGER NOT NULL,item_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,parent_smiles TEXT NOT NULL,
                morgan_similarity REAL NOT NULL,transfer_likelihood REAL,
                member_count INTEGER NOT NULL,
                PRIMARY KEY(benchmark_row_id,rank),UNIQUE(benchmark_row_id,item_id)
            ) WITHOUT ROWID;
        """)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", "ranked_level_retrieval.v2"),
            ("content_id", f"{level}-{method}"),
        ])
        connection.execute("INSERT INTO queries VALUES ('q1','CCC','query-parent','CCC')")
        connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?)", rows)
    manifest = directory / "VERSION.json"
    manifest.write_text(json.dumps({
        "schema_version": "ranked_level_retrieval.v2",
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "level": level,
        "ranking_method": method,
        "pool": "fixed" if level == "L1" else "all",
        "capacity": 100,
        "content_id": f"{level}-{method}",
        "database": database.name,
        "query_counts": {
            "q1": {"candidate_records": len(rows), "candidate_parents": len(rows)}
        },
    }))
    return manifest


def _release(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "bbb_martins"
    evidence_dir = task / "evidence"
    evidence_dir.mkdir(parents=True)
    evidence = evidence_dir / "evidence.sqlite3"
    with sqlite3.connect(evidence) as connection:
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE records(
                source_row_uid TEXT PRIMARY KEY,external_record_id TEXT NOT NULL,
                payload TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE contexts(
                context_id TEXT PRIMARY KEY,parent_id TEXT NOT NULL,
                condition_group TEXT NOT NULL,gold_label INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE context_records(
                context_id TEXT NOT NULL,source_row_uid TEXT NOT NULL,
                within_context_rank INTEGER NOT NULL
            );
        """)
        connection.execute("INSERT INTO metadata VALUES ('content_id','evidence-id')")
        connection.executemany("INSERT INTO records VALUES (?,?,?)", [
            ("u1", "r1", json.dumps({"canonical_smiles": "physical-1"})),
            ("u2", "r2", json.dumps({"canonical_smiles": "physical-2"})),
            ("u3", "r3", json.dumps({
                "canonical_smiles": "physical-3",
                "source_fields": {"measurement_text": "12", "unit_text": "%"},
            })),
            ("u4", "r4", json.dumps({"canonical_smiles": "physical-4"})),
        ])
        connection.execute("INSERT INTO contexts VALUES ('ctx1','card-parent','species=rat',1)")
        connection.executemany("INSERT INTO context_records VALUES (?,?,?)", [
            ("ctx1", "u1", 1), ("ctx1", "u2", 2),
        ])
    (evidence_dir / "VERSION.json").write_text(json.dumps({
        "status": "complete", "content_id": "evidence-id", "database": evidence.name,
    }))

    l1 = _ranking(task, "L1", "assay_transfer", [
        ("q1", 1, "ctx1", "card-parent", "CCO", 0.8, 0.9, 2),
    ])
    l2 = _ranking(task, "L2", "assay_transfer", [
        ("q1", 1, "u3", "p3", "CCN", 0.7, 0.8, 1),
        ("q1", 2, "u4", "p4", "CO", 0.6, 0.7, 1),
    ])
    index = task / "RELEASE_INDEX.json"
    index.write_text(json.dumps({
        "schema_version": "ranked_level_task_release_index.v2",
        "status": "complete", "task_id": "bbb_martins", "pool": "all",
        "evidence": {"manifest": "evidence/VERSION.json", "content_id": "evidence-id"},
        "splits": {"valid": {"levels": {
            "L1": {"assay_transfer": {"manifest": str(l1.relative_to(task))}},
            "L2": {"assay_transfer": {"manifest": str(l2.relative_to(task))}},
        }}},
    }))
    bundle = tmp_path / "bundle.yaml"
    bundle.write_text(yaml.safe_dump({
        "version": 18, "caches": {"bbb_martins": "bbb_martins/RELEASE_INDEX.json"}
    }))
    return bundle, index


def test_independent_cache_expands_all_l1_members_and_per_level_k(tmp_path: Path) -> None:
    bundle, _ = _release(tmp_path)
    policy = load_cache_policy(bundle, "bbb_martins", "valid", "assay-transfer", 2)
    molecules, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 1}, cache_pool="all",
    )
    card = molecules["q1"][0]
    assert card["context_card_id"] == "ctx1"
    assert [record["record_id"] for record in card["l1_records"]] == ["r1", "r2"]
    assert card["l1_records"][0]["payload"]["evidence_parent_smiles"] == "physical-1"
    assert card["l1_records"][0]["payload"]["canonical_smiles"] == "CCO"
    assert [record["record_id"] for record in later["q1"]["L2"]["records"]] == ["r3"]
    assert later["q1"]["L2"]["records"][0]["payload"]["source_contract"] == {
        "contract_version": "ranked_evidence_projection.v1",
        "source_or_simply_cleaned": {"measurement_text": True, "unit_text": True},
    }
    assert audit["neighbor_identity_policy_by_level"] == {
        "L1": "scaffold_disjoint", "L2": "parent_disjoint"
    }


def test_independent_cache_rejects_over_capacity_before_rendering(tmp_path: Path) -> None:
    bundle, _ = _release(tmp_path)
    policy = load_cache_policy(bundle, "bbb_martins", "valid", "assay-transfer", 2)
    with pytest.raises(ValueError, match="capacity 100"):
        load_candidates(
            {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
            later_limit={"L2": 101}, cache_pool="all",
        )


def test_independent_cache_rejects_joint_and_legacy_pool(tmp_path: Path) -> None:
    bundle, _ = _release(tmp_path)
    with pytest.raises(ValueError, match="does not support joint"):
        load_cache_policy(bundle, "bbb_martins", "valid", "joint", 2)
    policy = load_cache_policy(bundle, "bbb_martins", "valid", "assay-transfer", 2)
    with pytest.raises(ValueError, match="pool=all"):
        load_candidates(
            {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
            later_limit={"L2": 1}, cache_pool="tool-accepted",
        )
