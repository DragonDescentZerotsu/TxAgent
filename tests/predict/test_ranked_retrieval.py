import json
from pathlib import Path
import sqlite3

from predict.retrieval.assay_reranking.build_ranked_retrieval import _display_fields
from predict.retrieval.assay_reranking.ranked_retrieval import load_candidates


def test_display_adapter_keeps_measurement_unit_origins_atomic():
    display = _display_fields(
        "bioavailability_ma",
        {
            "canonical_measurement_scale_id": None,
            "canonical_measurement_text": "12",
            "canonical_unit_text": None,
            "measurement_kind": "continuous",
        },
        {"measurement_text": "12 mg", "unit_text": "mg"},
    )
    assert display == {
        "measurement_text": "12 mg",
        "unit_text": "mg",
        "origin": "source_pair",
        "adapter": "canonical_first_atomic_pair.v1",
    }


def test_display_adapter_uses_oral_semantic_category():
    display = _display_fields(
        "bioavailability_ma",
        {
            "canonical_measurement_scale_id": "fg_substrate_status_binary.v1",
            "canonical_category_id": "not_substrate",
        },
        {"measurement_text": "0"},
    )
    assert (display["measurement_text"], display["unit_text"], display["origin"]) == (
        "not substrate", "transporter substrate status", "semantic_category"
    )


def _write_runtime_fixture(root: Path) -> Path:
    evidence = root / "evidence.sqlite3"
    with sqlite3.connect(evidence) as connection:
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE records(
                source_row_uid TEXT PRIMARY KEY,external_record_id TEXT NOT NULL,
                payload TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE contexts(
                context_id TEXT PRIMARY KEY,parent_id TEXT,condition_group TEXT,gold_label INTEGER
            ) WITHOUT ROWID;
        """)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", "ranked_evidence_retrieval.v1"),
            ("content_id", "evidence-id"),
        ])
        connection.executemany("INSERT INTO records VALUES (?,?,?)", [
            ("u1", "r1", json.dumps({"canonical_smiles": "N#N", "display_measurement_text": "high"})),
            ("u2", "r2", json.dumps({"canonical_smiles": "CCN", "display_measurement_text": "3"})),
        ])
        connection.execute("INSERT INTO contexts VALUES ('ctx1','p1','species=rat',1)")

    rankings = root / "rankings.sqlite3"
    with sqlite3.connect(rankings) as connection:
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE queries(query_id INTEGER PRIMARY KEY,query_parent_id TEXT,query_parent_smiles TEXT);
            CREATE TABLE benchmark_queries(benchmark_row_id TEXT PRIMARY KEY,drug TEXT,query_id INTEGER);
            CREATE TABLE rankings(
                query_id INTEGER,pool TEXT,level TEXT,source_row_uid TEXT,parent_id TEXT,
                parent_smiles TEXT,context_id TEXT,within_parent_rank INTEGER,
                morgan_similarity REAL,morgan_rank INTEGER,assay_transfer_score REAL,assay_rank INTEGER
            );
            CREATE TABLE selection_counts(
                query_id INTEGER,pool TEXT,level TEXT,candidate_record_count INTEGER,
                candidate_parent_count INTEGER
            );
            CREATE TABLE l1_parent_contexts(
                query_id INTEGER,parent_id TEXT,morgan_context_id TEXT,assay_context_id TEXT
            );
        """)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", "ranked_evidence_retrieval.v1"),
            ("content_id", "ranking-id"),
        ])
        connection.execute("INSERT INTO queries VALUES (1,'query-parent','CCC')")
        connection.execute("INSERT INTO benchmark_queries VALUES ('q1','CCC',1)")
        connection.execute("INSERT INTO l1_parent_contexts VALUES (1,'p1','ctx1','ctx1')")
        connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (1, "fixed", "L1", "u1", "p1", "CCO", "ctx1", 1, 0.7, 1, 0.9, 1),
            (1, "tool-accepted", "L2", "u2", "p2", "CCN", None, 1, 0.6, 1, 0.8, 1),
        ])
        connection.executemany("INSERT INTO selection_counts VALUES (?,?,?,?,?)", [
            (1, "fixed", "L1", 1, 1),
            (1, "tool-accepted", "L2", 1, 1),
        ])

    manifest = root / "VERSION.json"
    manifest.write_text(json.dumps({
        "schema_version": "ranked_evidence_retrieval.v1",
        "status": "complete",
        "task_id": "bbb_martins",
        "subset": "valid",
        "gold_release": "v1",
        "neighbor_identity_policy": "l1_voter_membership_later_parent_disjoint",
        "joint_materialized": False,
        "pools": ["tool-accepted", "tool-compatible", "all"],
        "rankings_database": "rankings.sqlite3",
        "evidence_database": "evidence.sqlite3",
        "content_id": "ranking-id",
        "evidence_content_id": "evidence-id",
        "capacities": {
            "l1_molecules": 1, "l1_records_per_molecule": 1,
            "later_records_per_level": 1,
        },
        "assignment_counts": {"fixed/L1": 1, "tool-accepted/L2": 1},
        "evidence_record_count": 2,
    }))
    return manifest


def test_runtime_reads_rankings_and_evidence_indexes(tmp_path):
    manifest = _write_runtime_fixture(tmp_path)
    policy = {
        "selection_contract": "ranked_evidence_retrieval.v1",
        "cache_manifest": str(manifest),
        "stages": {"L1": "assay_transfer", "L2": "morgan"},
        "inputs": {},
    }
    molecules, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=1, later_limit=1,
    )
    assert molecules["q1"][0]["l1_records"][0]["payload"]["display_measurement_text"] == "high"
    assert molecules["q1"][0]["canonical_smiles"] == "CCO"
    assert molecules["q1"][0]["l1_records"][0]["payload"]["canonical_smiles"] == "CCO"
    assert molecules["q1"][0]["l1_records"][0]["payload"]["evidence_parent_smiles"] == "N#N"
    assert molecules["q1"][0]["selected_condition"] == "species=rat"
    assert molecules["q1"][0]["_diagnostic_label"] == 1
    assert later["q1"]["L2"]["records"][0]["record_id"] == "r2"
    assert audit["neighbor_identity_policy"] == "l1_voter_membership_later_parent_disjoint"
