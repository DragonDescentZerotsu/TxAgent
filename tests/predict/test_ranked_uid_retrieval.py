import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from predict.retrieval.assay_reranking import cache_matched
from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder
from predict.retrieval.assay_reranking import ranked_uid_retrieval
from predict.retrieval.assay_reranking.build_ranked_uid_retrieval import finalize_level
from predict.retrieval.assay_reranking.ranked_uid_retrieval import (
    HIDDEN_PARTIAL_METHOD,
    HYBRID_METHOD,
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


@pytest.mark.parametrize("benchmark,task", [("gold", "carcinogens"), ("tdc", "ames")])
def test_partial_snapshot_profiles_are_active(benchmark: str, task: str) -> None:
    from predict.retrieval.assay_reranking.build_safety_v27_ranked_retrieval import profile

    cache = profile(task, benchmark, 40, snapshot=True)
    assert cache in builder.runtime.ACTIVE_CACHE_PROFILES
    assert builder.runtime.cache_profile_root(cache).is_relative_to(
        builder.runtime.DATA_ACTIVE_CACHE_ROOT
    )


@pytest.mark.parametrize("benchmark,task", [("gold", "carcinogens"), ("tdc", "ames")])
def test_hybrid_profiles_are_active(benchmark: str, task: str) -> None:
    from predict.retrieval.assay_reranking.build_safety_v27_ranked_retrieval import profile

    cache = profile(task, benchmark, 40, hybrid=True)
    assert cache in builder.runtime.ACTIVE_CACHE_PROFILES
    assert "/l2plus/hybrid/v27/" in cache


def test_normal_validator_rejects_partial_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "ames"
    root.mkdir()
    (root / "RELEASE_INDEX.json").write_text(json.dumps({
        "schema_version": "ranked_uid_task_release_index.v1",
        "status": "complete", "task_id": "ames",
        "score_coverage": "partial_snapshot",
        "profile": "flat_v5/tdc_v1/ames/l2plus/assay_transfer/v27/"
                   "general_candidate_copy_shared_parent40_partial_snapshot_v1",
    }))
    with pytest.raises(ValueError, match="requires explicit validation"):
        builder.validate_release("ames", tmp_path, tmp_path / "evidence.json")


def test_dili_queries_recover_frozen_identity_from_drug() -> None:
    _, rows, queries = builder._queries("dili", "valid")
    assert len(queries) == len(rows)
    assert all(parent == row["molecule_identity_key"] for row, (_, _, parent, _) in zip(rows, queries))


def test_frozen_candidate_query_identity_does_not_renormalize(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data/gold_labels/Skin_Reaction/v1/scaffold"
    root.mkdir(parents=True)
    query = root / "test_molecule_condition_labels.jsonl"
    query.write_text(json.dumps({"benchmark_row_id": "q", "drug": "*CC"}) + "\n")
    level = tmp_path / "base/skin_reaction/scaffold/test/L2"
    level.mkdir(parents=True)
    with sqlite3.connect(level / "rankings.sqlite3") as connection:
        connection.execute("CREATE TABLE queries(benchmark_row_id TEXT,drug TEXT,query_parent_id TEXT,query_parent_smiles TEXT)")
        connection.execute("INSERT INTO queries VALUES ('q','*CC','','')")
    (level / "VERSION.json").write_text(json.dumps({
        "database": "rankings.sqlite3", "inputs": {"query_sha256": sha256_file(query)},
    }))
    monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(builder, "BASE_RANKING_ROOT", tmp_path / "base")
    monkeypatch.setattr(builder, "QUERY_BENCHMARK", "gold")
    _, _, frozen = builder._queries("skin_reaction", "test", "L2")
    assert frozen == [("q", "*CC", "", "")]


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
    assert builder._records(manifest, "L3", {"p3"})[0].keys() == selected.keys()
    assert builder._records(manifest, "L3", {"p2"}) == ({}, {})


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


def test_prepared_score_journal_reuses_exact_keys_and_checks_prefix(
    tmp_path: Path, monkeypatch,
) -> None:
    level = tmp_path / "skin_reaction/scaffold/valid/L3"
    journal_dir = level / ".scores"
    journal_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"score_key": "a", "prompt": "short"},
        {"score_key": "b", "prompt": "longer"},
    ]), level / "prompts.parquet")
    (level / "VERSION.json").write_text(json.dumps({
        "status": "prepared", "model": {"revision": "pinned"},
        "prompts": "prompts.parquet",
    }))
    journal = journal_dir / "00-of-01.jsonl"
    journal.write_text(json.dumps({"score_key": "a", "assay_transfer_score": 0.7}) + "\n")
    monkeypatch.setattr(builder, "REUSE_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(builder, "SCORE_REUSE_ROOTS", (tmp_path,))
    monkeypatch.setattr(builder, "_model_spec", lambda *_: {"revision": "pinned"})

    assert builder._reuse_scores("skin_reaction", "valid", "L3", {"a", "b"}) == {"a": 0.7}
    journal.write_text(json.dumps({"score_key": "b", "assay_transfer_score": 0.7}) + "\n")
    with pytest.raises(ValueError, match="exact shard prefix"):
        builder._reuse_scores("skin_reaction", "valid", "L3", {"a", "b"})


def test_pretokenized_shards_match_live_answer_prefixes(tmp_path: Path, monkeypatch) -> None:
    from transformers import AutoTokenizer

    class Tokenizer:
        def apply_chat_template(self, messages, **_):
            return "user: " + messages[0]["content"] + " answer: "

        def __call__(self, value, **_):
            values = value if isinstance(value, list) else [value]
            ids = [[ord(char) for char in item] for item in values]
            return SimpleNamespace(input_ids=ids[0]) if isinstance(value, str) else {"input_ids": ids}

    source = tmp_path / "prepared/ames/scaffold/valid/L2"
    source.mkdir(parents=True)
    prompts = [
        {"score_key": "b", "prompt": "long candidate"},
        {"score_key": "a", "prompt": "short"},
        {"score_key": "c", "prompt": "mid"},
    ]
    pq.write_table(pa.Table.from_pylist(prompts), source / "prompts.parquet")
    spec = {"model": "test-model", "revision": "pinned"}
    (source / "VERSION.json").write_text(json.dumps({
        "status": "prepared", "prompts": "prompts.parquet", "model": spec,
    }))
    tokenizer = Tokenizer()
    monkeypatch.setattr(builder, "_model_spec", lambda *_: spec)
    monkeypatch.setattr(builder.runtime, "resolve_model_snapshot", lambda *_, **__: "snapshot")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_, **__: tokenizer)

    result = builder.pretokenize_level("ames", "valid", "L2", tmp_path / "prepared", tmp_path / "tokens", 2, 2)
    ordered = sorted(prompts, key=lambda row: (len(row["prompt"]), row["score_key"]))
    for shard in range(2):
        name = f"{shard:02d}-of-02.parquet"
        actual = pq.read_table(tmp_path / "tokens/ames/scaffold/valid/L2" / name).to_pylist()
        expected = ordered[shard::2]
        assert [row["score_key"] for row in actual] == [row["score_key"] for row in expected]
        prefixes, a_tokens, b_tokens = builder.runtime._answer_prefixes(
            tokenizer, [SimpleNamespace(prompt=row["prompt"]) for row in expected],
        )
        assert [(row["prefix_ids"], row["a_token"], row["b_token"]) for row in actual] == list(zip(
            prefixes, a_tokens, b_tokens,
        ))
        assert result["files"][name] == sha256_file(tmp_path / "tokens/ames/scaffold/valid/L2" / name)

    parallel = builder.pretokenize_level(
        "ames", "valid", "L2", tmp_path / "prepared", tmp_path / "parallel", 2, 2, 2,
    )
    assert parallel["counts"] == result["counts"]
    assert parallel["files"] == result["files"]
    wide = builder.pretokenize_level(
        "ames", "valid", "L2", tmp_path / "prepared", tmp_path / "wide", 8, 2, 8,
    )
    assert sum(wide["counts"].values()) == len(prompts)
    assert len(wide["files"]) == 8
    for shard in range(8):
        name = f"{shard:02d}-of-08.parquet"
        actual = pq.read_table(tmp_path / "wide/ames/scaffold/valid/L2" / name).to_pylist()
        assert [row["score_key"] for row in actual] == [row["score_key"] for row in ordered[shard::8]]


def test_score_level_uses_matching_pretokenized_shard(tmp_path: Path, monkeypatch) -> None:
    level = tmp_path / "prepared/ames/scaffold/valid/L2"
    level.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        {"score_key": "b", "prompt": "longer", "projection_hash": "p"},
        {"score_key": "a", "prompt": "short", "projection_hash": "p"},
    ]), level / "prompts.parquet")
    (level / "VERSION.json").write_text(json.dumps({
        "status": "prepared", "database": "rankings.sqlite3", "prompts": "prompts.parquet",
    }))
    token_dir = tmp_path / "tokens/ames/scaffold/valid/L2"
    token_dir.mkdir(parents=True)
    token_file = token_dir / "00-of-01.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"score_key": "a", "prefix_ids": [1, 2], "a_token": 3, "b_token": 4},
        {"score_key": "b", "prefix_ids": [5, 6], "a_token": 7, "b_token": 8},
    ]), token_file)
    spec = {"model": "test-model", "revision": "pinned"}
    (token_dir / "VERSION.json").write_text(json.dumps({
        "prompts_sha256": sha256_file(level / "prompts.parquet"), "model": spec,
        "num_shards": 1, "files": {token_file.name: sha256_file(token_file)},
    }))
    seen = []
    monkeypatch.setattr(builder, "_model_spec", lambda *_: spec)
    monkeypatch.setattr(builder, "_renderer", lambda *_: SimpleNamespace(template_hash="template"))
    monkeypatch.setattr(builder.runtime, "resolve_model_snapshot", lambda *_, **__: "snapshot")
    monkeypatch.setattr(builder.runtime, "load_model", lambda *_, **__: (object(), object()))

    def fake_score(_model, _tokenizer, tasks, *, device, tokenized):
        seen.append((device, tokenized))
        return [SimpleNamespace(cache_key=task.cache_key, transfer_probability=0.75) for task in tasks]

    monkeypatch.setattr(builder.runtime, "score_prompt_batch", fake_score)
    result = builder.score_level("ames", "valid", "L2", tmp_path / "prepared", 0, 1, 0, 2, tmp_path / "tokens")

    assert result["scores"] == 2
    assert seen == [(0, ([[1, 2], [5, 6]], [3, 7], [4, 8]))]
    assert [row["score_key"] for row in builder.read_jsonl(Path(result["journal"]))] == ["a", "b"]
    Path(result["journal"]).unlink()
    pq.write_table(pa.Table.from_pylist([
        {"score_key": "a", "prompt": "changed", "projection_hash": "p"},
    ]), level / "prompts.parquet")
    with pytest.raises(ValueError, match="Pretokenized score inputs"):
        builder.score_level("ames", "valid", "L2", tmp_path / "prepared", 0, 1, 0, 2, tmp_path / "tokens")


def test_shared_l2plus_parent_universe_expands_only_matching_later_records(
    tmp_path: Path, monkeypatch,
) -> None:
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text("{}\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n")
    shared = tmp_path / "out/skin_reaction/scaffold/valid/PARENT_UNIVERSE.json"
    shared.parent.mkdir(parents=True)
    shared.write_text(json.dumps({
        "schema_version": "shared_later_parent_universe.v1",
        "task_id": "skin_reaction", "subset": "valid", "parent_capacity": 2,
        "query_sha256": sha256_file(query_path),
        "evidence_manifest_sha256": sha256_file(evidence),
        "parents_by_query": {"q": [["p1", "CCC", 0.8], ["p2", "CCCC", 0.7]]},
    }))
    records = {
        "keep": {"parent_id": "p1", "canonical_smiles": "CCC"},
        "drop": {"parent_id": "p3", "canonical_smiles": "CCCCC"},
    }
    monkeypatch.setattr(builder, "CAPACITY", 2)
    monkeypatch.setattr(builder, "SHARED_LATER_PARENT_UNIVERSE", True)
    monkeypatch.setattr(builder, "BASE_RANKING_ROOT", None)
    monkeypatch.setattr(builder, "SCORE_REUSE_ROOTS", ())
    monkeypatch.setattr(builder, "_model_spec", lambda *_: None)
    monkeypatch.setattr(builder, "_queries", lambda *args: (
        query_path, [{"benchmark_row_id": "q", "drug": "CC"}], [("q", "CC", "query", "CC")],
    ))
    monkeypatch.setattr(builder, "_records", lambda *args: (
        records, {"L2": {"p1": ["keep"], "p3": ["drop"]}},
    ))

    result = builder.prepare_level("skin_reaction", "valid", "L2", tmp_path / "out", evidence)

    assert result["parent_universe_source"] == "shared_l2plus_morgan_2"
    assert result["query_counts"]["q"] == {"candidate_parents": 1, "candidate_records": 1}
    with sqlite3.connect(tmp_path / "out/skin_reaction/scaffold/valid/L2/rankings.sqlite3") as connection:
        assert connection.execute("SELECT item_id,parent_morgan_rank FROM rankings").fetchall() == [("keep", 1)]


def test_shared_later_universe_ranks_combined_parents_once(
    tmp_path: Path, monkeypatch,
) -> None:
    projection = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"parent_id": "p1", "parent_smiles": "CCO", "level": "L2"},
        {"parent_id": "p2", "parent_smiles": "CCN", "level": "L3"},
        {"parent_id": "p3", "parent_smiles": "N#N", "level": "L3"},
        {"parent_id": "p4", "parent_smiles": "not_a_smiles", "level": "L3"},
        {"parent_id": "p0", "parent_smiles": "O", "level": "L1"},
    ]), projection)
    evidence = tmp_path / "VERSION.json"
    evidence.write_text(json.dumps({
        "records": projection.name, "records_sha256": sha256_file(projection),
    }))
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text("{}\n")
    monkeypatch.setattr(builder, "CAPACITY", 2)
    monkeypatch.setattr(builder, "SHARED_LATER_PARENT_UNIVERSE", True)
    monkeypatch.setattr(builder, "BASE_RANKING_ROOT", None)
    monkeypatch.setattr(builder, "TASK_LEVELS", {"skin_reaction": ("L2", "L3")})
    monkeypatch.setattr(builder, "_queries", lambda *args: (
        query_path, [{"benchmark_row_id": "q", "drug": "CCO"}], [("q", "CCO", "query", "CCO")],
    ))

    builder.prepare_shared_parent_universe("skin_reaction", "valid", tmp_path / "out", evidence)

    selected = json.loads((tmp_path / "out/skin_reaction/scaffold/valid/PARENT_UNIVERSE.json").read_text())
    assert [row[0] for row in selected["parents_by_query"]["q"]] == ["p1", "p2"]
    assert selected["evidence_parent_count"] == 4
    assert selected["unrankable_parent_ids"] == ["p4"]


@pytest.mark.parametrize("offset, expected", [
    (0, ["p1", "p2"]),
    (1, ["p2", "p3"]),
])
def test_shared_parent_successor_trims_frozen_ranking(
    tmp_path: Path, monkeypatch, offset: int, expected: list[str],
) -> None:
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text("{}\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"records_sha256": "records"}))
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "schema_version": "shared_later_parent_universe.v1",
        "task_id": "ames", "subset": "valid", "parent_capacity": 3,
        "query_sha256": sha256_file(query_path),
        "evidence_manifest_sha256": sha256_file(evidence),
        "evidence_records_sha256": "records", "evidence_parent_count": 3,
        "unrankable_parent_ids": [],
        "parents_by_query": {"q": [["p1", "C", 0.9], ["p2", "CC", 0.8], ["p3", "CCC", 0.7]]},
    }))
    monkeypatch.setattr(builder, "CAPACITY", 2)
    monkeypatch.setattr(builder, "SOURCE_PARENT_OFFSET", offset)
    monkeypatch.setattr(builder, "SHARED_LATER_PARENT_UNIVERSE", True)
    monkeypatch.setattr(builder, "BASE_RANKING_ROOT", None)
    monkeypatch.setattr(builder, "TASK_LEVELS", {"ames": ("L2", "L3")})
    monkeypatch.setattr(builder, "_queries", lambda *args: (
        query_path, [{"benchmark_row_id": "q", "drug": "C"}], [("q", "C", "query", "C")],
    ))

    builder.prepare_shared_parent_universe("ames", "valid", tmp_path / "out", evidence, source)

    result = json.loads((tmp_path / "out/ames/scaffold/valid/PARENT_UNIVERSE.json").read_text())
    assert [row[0] for row in result["parents_by_query"]["q"]] == expected
    assert result["parent_capacity"] == 2
    assert result["source_parent_universe_sha256"] == sha256_file(source)
    assert result.get("source_parent_rank_start") == (offset + 1 if offset else None)


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


def test_hybrid_incomplete_level_keeps_all_records_and_hides_scores(tmp_path: Path) -> None:
    policy, manifest = _release(tmp_path)
    with sqlite3.connect(manifest.parent / "rankings.sqlite3") as connection:
        connection.execute(
            "UPDATE rankings SET assay_transfer_score=NULL,assay_rank=NULL WHERE item_id='u7'"
        )
    document = json.loads(manifest.read_text())
    document["score_coverage"] = "partial_snapshot"
    document["parent_universe_source"] = "shared_l2plus_morgan_100"
    document["query_counts"]["q1"].update(scored_records=4, missing_scores=1)
    manifest.write_text(json.dumps(document))
    index = json.loads(Path(policy["cache_index"]).read_text())
    index["splits"]["valid"]["levels"]["L2"]["manifest_sha256"] = sha256_file(manifest)
    Path(policy["cache_index"]).write_text(json.dumps(index))
    policy["stages"]["L2"] = HYBRID_METHOD
    policy["all_later_records"] = True

    _, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 1}, cache_pool="all",
    )

    records = later["q1"]["L2"]["records"]
    assert len(records) == 5
    assert records[-1]["record_id"] == "r7"
    assert {row["ranking_method"] for row in records} == {HIDDEN_PARTIAL_METHOD}
    assert all("transfer_likelihood" not in row and row["assay_rank"] is None for row in records)
    assert later["q1"]["L2"]["allow_shortfall"] is False
    level_audit = audit["query_audits"]["q1"]["L2"]
    assert level_audit["selection_mode"] == "scored_assay_then_unscored_morgan"
    assert level_audit["prompt_assay_scores"] == "hidden"


def test_hybrid_complete_small_level_keeps_assay_scores(tmp_path: Path) -> None:
    policy, manifest = _release(tmp_path)
    document = json.loads(manifest.read_text())
    document["parent_universe_source"] = "shared_l2plus_morgan_100"
    manifest.write_text(json.dumps(document))
    index = json.loads(Path(policy["cache_index"]).read_text())
    index["splits"]["valid"]["levels"]["L2"]["manifest_sha256"] = sha256_file(manifest)
    Path(policy["cache_index"]).write_text(json.dumps(index))
    policy["stages"]["L2"] = HYBRID_METHOD
    policy["all_later_records"] = True

    _, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 1}, cache_pool="all",
    )

    records = later["q1"]["L2"]["records"]
    assert len(records) == 5
    assert all(row["ranking_method"] == "assay_transfer" for row in records)
    assert all("transfer_likelihood" in row for row in records)
    assert audit["query_audits"]["q1"]["L2"]["prompt_assay_scores"] == "visible"


def test_composite_levels_hydrate_overlap_from_their_own_projection(tmp_path: Path) -> None:
    policy, later_manifest = _release(tmp_path)
    with sqlite3.connect(later_manifest.parent / "rankings.sqlite3") as connection:
        connection.execute("UPDATE rankings SET item_id='u1' WHERE item_id='u4'")
    later_evidence = tmp_path / "later_evidence"
    later_evidence.mkdir()
    pq.write_table(pa.Table.from_pylist([{
        "source_row_uid": "u1", "external_record_id": "r-later",
        "payload": json.dumps({"canonical_smiles": "later-projection"}),
    }]), later_evidence / "records.parquet")
    evidence_manifest = later_evidence / "VERSION.json"
    evidence_manifest.write_text(json.dumps({
        "schema_version": "ranked_evidence_projection.v1", "status": "complete",
        "task_id": "bbb_martins", "content_id": "later-evidence",
        "records": "records.parquet",
    }))
    later_index = tmp_path / "L2_INDEX.json"
    index = json.loads(Path(policy["cache_index"]).read_text())
    index["evidence"] = {
        "manifest": "later_evidence/VERSION.json", "content_id": "later-evidence",
        "manifest_sha256": sha256_file(evidence_manifest),
    }
    later_index.write_text(json.dumps(index))
    policy["cache_indexes"] = {
        "L1": policy["cache_index"], "L2": str(later_index),
    }

    molecules, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=2, later_limit={"L2": 1}, cache_pool="all",
    )

    assert molecules["q1"][0]["l1_records"][0]["record_id"] == "r1"
    assert later["q1"]["L2"]["records"][0]["record_id"] == "r-later"
    assert audit["evidence_manifests_by_level"] == {
        "L1": [str(tmp_path / "evidence/VERSION.json")],
        "L2": [str(evidence_manifest)],
    }


@pytest.mark.parametrize("later_id, later_smiles, accepted", [
    ("", "CCC", True), ("different", "CCC", False), ("", "CCO", False),
])
def test_level_query_identity_allows_only_missing_parent_id(
    tmp_path: Path, later_id: str, later_smiles: str, accepted: bool,
) -> None:
    policy, manifest = _release(tmp_path)
    with sqlite3.connect(manifest.parent / "rankings.sqlite3") as connection:
        connection.execute(
            "UPDATE queries SET query_parent_id=?,query_parent_smiles=?",
            (later_id, later_smiles),
        )
    request = lambda: load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 1}, cache_pool="all",
    )
    if accepted:
        request()
    else:
        with pytest.raises(ValueError, match="disagree on query identity"):
            request()


@pytest.mark.parametrize("width", [40, 50, 100])
def test_shared_parent_policy_keeps_every_later_physical_record(tmp_path: Path, width: int) -> None:
    _, later_manifest = _release(tmp_path)
    later_document = json.loads(later_manifest.read_text())
    later_document["parent_capacity"] = width
    later_document["parent_universe_source"] = f"shared_l2plus_morgan_{width}"
    later_manifest.write_text(json.dumps(later_document))
    index_path = tmp_path / "RELEASE_INDEX.json"
    index = json.loads(index_path.read_text())
    (tmp_path / "L1_INDEX.json").write_text(json.dumps(index))
    index["parent_capacity"] = width
    index["later_candidate_universe"] = f"all_uids_under_shared_l2plus_morgan_top_{width}_parents"
    index["splits"]["valid"]["levels"]["L2"]["manifest_sha256"] = sha256_file(later_manifest)
    index_path.write_text(json.dumps(index))
    config = tmp_path / "cache.yaml"
    config.write_text(
        "version: 20\ncaches:\n  bbb_martins:\n"
        "    L1: L1_INDEX.json\n    later: RELEASE_INDEX.json\n"
    )

    policy = cache_matched.load_cache_policy(
        config, "bbb_martins", "valid", "assay-transfer", 2,
    )
    ranked, _, release = load_ranked_universe(
        index_path, task="bbb_martins", subset="valid", levels=("L2",), queries={"q1": "CCC"},
    )
    _, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 1}, cache_pool="all",
        morgan_primary_parent_width=1,
    )

    assert policy["all_later_records"] is True
    assert len(ranked["L2"]["q1"]) == 5
    assert release["parent_capacity"] == width
    assert len(later["q1"]["L2"]["records"]) == 5
    assert audit["contract"]["all_later_records"] is True


@pytest.mark.parametrize("task", ["dili", "carcinogens"])
def test_shared_safety_policy_exposes_all_scored_levels(tmp_path: Path, task: str) -> None:
    index = {
        "schema_version": "ranked_uid_task_release_index.v1", "status": "complete",
        "task_id": task, "pool": "all",
        "later_candidate_universe": "all_uids_under_shared_l2plus_morgan_top_100_parents",
        "splits": {"valid": {"levels": {
            f"L{i}": {"manifest": f"L{i}/VERSION.json"} for i in range(1, 8)
        }}},
    }
    (tmp_path / "direct.json").write_text(json.dumps(index))
    (tmp_path / "later.json").write_text(json.dumps(index))
    config = tmp_path / "cache.yaml"
    config.write_text(
        f"version: 20\ncaches:\n  {task}:\n"
        "    L1: direct.json\n    later: later.json\n"
    )
    policy = cache_matched.load_cache_policy(
        config, task, "valid", "assay-transfer", 7,
    )
    assert tuple(policy["stages"]) == tuple(f"L{i}" for i in range(1, 8))
    assert policy["stages"]["L5"] == "assay_transfer"
    assert policy["all_later_records"] is True


def test_version_21_policy_requires_and_selects_hybrid_later_levels(tmp_path: Path) -> None:
    index = {
        "schema_version": "ranked_uid_task_release_index.v1", "status": "complete",
        "task_id": "ames", "pool": "all", "selection_coverage": "complete",
        "hybrid_policy": {
            "schema_version": "assay_complete_or_hidden_morgan_tail.v1",
            "scope": "query_level", "complete_order": "assay_transfer",
            "incomplete_order": "scored_assay_then_unscored_morgan",
            "incomplete_prompt_scores": "hidden",
        },
        "later_candidate_universe": "all_uids_under_shared_l2plus_morgan_top_40_parents",
        "splits": {"test": {"levels": {
            "L1": {"manifest": "L1/VERSION.json"},
            "L2": {"manifest": "L2/VERSION.json"},
        }}},
    }
    (tmp_path / "direct.json").write_text(json.dumps(index))
    (tmp_path / "later.json").write_text(json.dumps(index))
    config = tmp_path / "cache.yaml"
    config.write_text(
        "version: 21\ncaches:\n  ames:\n"
        "    L1: direct.json\n    later: later.json\n"
    )

    policy = cache_matched.load_cache_policy(
        config, "ames", "test", "assay-transfer", 2,
    )

    assert policy["stages"] == {"L1": "assay_transfer", "L2": HYBRID_METHOD}
    assert policy["all_later_records"] is True


def test_composite_direct_policy_ignores_absent_later_index(tmp_path: Path) -> None:
    (tmp_path / "direct.json").write_text(json.dumps({
        "schema_version": "ranked_uid_task_release_index.v1",
        "status": "complete", "task_id": "ames", "pool": "all",
        "splits": {"test": {"levels": {"L1": {"manifest": "L1/VERSION.json"}}}},
    }))
    config = tmp_path / "cache.yaml"
    config.write_text(
        "version: 20\ncaches:\n  ames:\n"
        "    L1: direct.json\n    later: missing.json\n"
    )
    policy = cache_matched.load_cache_policy(
        config, "ames", "test", "assay-transfer", 1,
    )
    assert tuple(policy["stages"]) == ("L1",)
    with pytest.raises(FileNotFoundError):
        cache_matched.load_cache_policy(config, "ames", "test", "assay-transfer", 2)


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


def test_shared_universe_can_use_preselected_uids(tmp_path: Path) -> None:
    policy, _ = _release(tmp_path)
    policy["all_later_records"] = True

    _, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=10, later_limit={"L2": 2}, cache_pool="all",
        preselected_uids={"q1": {"L2": ["u6", "u3"]}},
        preselected_uid_budget=2,
    )

    assert [row["record_id"] for row in later["q1"]["L2"]["records"]] == ["r6", "r3"]
    assert audit["contract"]["all_later_records"] is False


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


def test_partial_snapshot_preserves_morgan_universe_and_allows_empty_assay_level(
    tmp_path: Path,
) -> None:
    policy, manifest = _release(tmp_path)
    document = json.loads(manifest.read_text())
    document.update(status="prepared", score_counts={"exact_reuse": 0, "pending": 5},
                    snapshot_source={"manifest_sha256": "source"})
    with sqlite3.connect(manifest.parent / "rankings.sqlite3") as connection:
        connection.execute("UPDATE rankings SET assay_transfer_score=NULL,assay_rank=NULL")
    manifest.write_text(json.dumps(document))
    output = tmp_path / "snapshot"
    stage = output / "bbb_martins/scaffold/valid/L2"
    stage.mkdir(parents=True)
    shutil.copy2(manifest, stage / "VERSION.json")
    shutil.copy2(manifest.parent / "rankings.sqlite3", stage / "rankings.sqlite3")
    with pytest.raises(ValueError, match="Incomplete fresh scores"):
        finalize_level("bbb_martins", "valid", "L2", output)
    journal = stage / ".scores" / "00-of-01.jsonl"
    journal.parent.mkdir()
    journal.write_text(json.dumps({"score_key": "unknown", "assay_transfer_score": 0.5}) + "\n")
    with pytest.raises(ValueError, match="extra=1"):
        finalize_level("bbb_martins", "valid", "L2", output, partial_snapshot=True)
    journal.unlink()
    result = finalize_level("bbb_martins", "valid", "L2", output, partial_snapshot=True)
    assert result["query_counts"]["q1"]["scored_records"] == 0
    assert result["query_counts"]["q1"]["missing_scores"] == 5
    with sqlite3.connect(stage / "rankings.sqlite3") as connection:
        assert connection.execute("SELECT morgan_rank FROM rankings ORDER BY morgan_rank").fetchall() == [
            (1,), (2,), (3,), (4,), (5,),
        ]
    policy["cache_manifests"]["L2"] = str(stage / "VERSION.json")
    index_path = Path(policy["cache_index"])
    index = json.loads(index_path.read_text())
    index["splits"]["valid"]["levels"]["L2"].update(
        manifest=str((stage / "VERSION.json").relative_to(tmp_path)),
        manifest_sha256=sha256_file(stage / "VERSION.json"), content_id=result["content_id"],
    )
    index_path.write_text(json.dumps(index))
    _, later, audit = load_candidates(
        {"q1": "CCC"}, task="bbb_martins", subset="valid", policy=policy,
        molecule_limit=1, l1_limit=1, later_limit={"L2": 3}, cache_pool="all",
    )
    assert later["q1"]["L2"]["records"] == []
    assert later["q1"]["L2"]["allow_shortfall"] is True
    assert audit["query_audits"]["q1"]["L2"]["missing_scores"] == 5
    assert audit["cache_capacities"]["L2"] == 0
