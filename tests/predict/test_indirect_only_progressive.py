"""Contracts for direct-free, independent indirect-level prompts."""

from collections import Counter
import json
import sqlite3

import pytest

from predict.harnesses.progressive import _records
from predict.harnesses.progressive import runner
from predict.harnesses.progressive import record_filter
from predict.retrieval.assay_reranking import build_indirect_morgan_semantic_cache as cache
from predict.retrieval.assay_reranking import build_indirect_morgan_semantic_filter_cache as filter_cache
from predict.retrieval.assay_reranking import indirect_cache


PROMPT = "reranked_progressive_l1_context_v4_no_query_prior"


def _record(level: int, method: str) -> dict:
    return {
        "record_id": f"record-{level}",
        "reference_molecule_id": "parent-1",
        "morgan_similarity": 0.81234,
        "ranking_method": method,
        "payload": {
            "canonical_smiles": "CCN",
            "source_id": "indirect",
            "measurement_kind": "continuous",
            "source_fields": {
                "measurement_text": "12 nM",
                "assay_context": "indirect assay sentinel",
            },
        },
    }


@pytest.mark.parametrize("level", (2, 3, 4))
@pytest.mark.parametrize("mode", ("morgan-parent-control", "morgan-parent-semantic"))
def test_indirect_only_cli_fixes_level_prompt_prior_and_pool(level, mode):
    args = runner.parse_args([
        "--harness-version", runner.INDIRECT_ONLY_HARNESS,
        "--indirect-level", str(level), "--reranking", mode,
        "--tasks", "bbb_martins", "--prepare-only", "--skip-tool-prefetch",
    ])
    assert args.prompt_version == PROMPT
    assert args.query_prior == "none"
    assert args.record_pool == args.cache_pool == "all"
    assert args.indirect_record_limit_per_level == 25
    assert args.retrieval_policies["bbb_martins"]["stages"] == {
        f"L{level}": mode.replace("-", "_")
    }
    assert [row["level"] for row in runner._run_levels(args, "bbb_martins")] == [level]


@pytest.mark.parametrize("mode,limit", (
    ("morgan-parent-semantic", 25),
    ("morgan-parent-llm-semantic", 100),
))
def test_v2_routes_only_llm_semantic_through_record_filter(mode, limit):
    args = runner.parse_args([
        "--harness-version", runner.INDIRECT_FILTER_HARNESS,
        "--indirect-level", "2", "--reranking", mode,
        "--tasks", "bbb_martins", "--prepare-only", "--skip-tool-prefetch",
    ])

    assert args.indirect_record_limit_per_level == limit
    assert args.retrieval_policies["bbb_martins"]["stages"] == {
        "L2": mode.replace("-", "_")
    }


@pytest.mark.parametrize("level", (2, 3, 4))
def test_indirect_only_v4_render_has_no_l1_prior_or_state(level):
    method = "morgan_parent_semantic"
    snapshots = _records.stage_ranked_snapshots(
        [], task="bbb_martins",
        records_by_level={f"L{level}": {"records": [_record(level, method)]}},
        prompt_version=PROMPT,
    )
    levels = [row for row in _records.tianang_aligned_levels(
        "bbb_martins", level, prompt_version=PROMPT
    ) if row["level"] == level]
    messages = _records.build_tianang_aligned_messages(
        contract=runner._task_contract("bbb_martins", PROMPT), levels=levels,
        current_level=level, query_smiles="CCO", condition_sentence="",
        query_prior=None, query_tool_summary=None, active=snapshots[level],
        prior_state=None, prompt_version=PROMPT, record_limit=10,
        l2_record_limit=10, indirect_record_limit=1,
        retrieval_policy={f"L{level}": method},
    )
    rendered = messages[1]["content"]
    assert "indirect assay sentinel" in rendered
    assert "Morgan similarity: 0.8123" in rendered
    assert "Query-property prior" not in rendered
    assert "prior_state" not in rendered


def test_indirect_only_cli_rejects_non_v4_prompt():
    with pytest.raises(SystemExit):
        runner.parse_args([
            "--harness-version", runner.INDIRECT_ONLY_HARNESS,
            "--indirect-level", "3", "--reranking", "morgan-parent-control",
            "--prompt-version", "reranked_progressive_l1_context_v3",
        ])


def test_indirect_cache_keeps_one_query_identity_per_benchmark(monkeypatch):
    source = sqlite3.connect(":memory:")
    source.executescript("""
      CREATE TABLE queries(query_id INTEGER PRIMARY KEY,query_smiles TEXT UNIQUE);
      CREATE TABLE benchmark_queries(benchmark_row_id TEXT PRIMARY KEY,drug TEXT,query_id INTEGER);
      INSERT INTO queries VALUES (7,'CCO');
      INSERT INTO benchmark_queries VALUES ('a','A',7),('b','B',7);
    """)
    output = sqlite3.connect(":memory:")
    cache._create_tables(output)

    def rows(_, __, level, ___, ____, benchmark, _____):
        return [{"record_id": f"{level}-{index}", "parent_id": f"p{index // 10}",
                 "parent_smiles": "CCN", "payload": {"source_row_uid": f"u{index}"},
                 "morgan_similarity": 0.8,
                 "semantic_bucket_id": f"bucket-{index % 10 // 5}",
                 "semantic_rank": index % 10 // 5 + 1, "retrieval_eligible": True,
                 "tie_key": f"{benchmark}-{index:02d}"} for index in range(25)]

    monkeypatch.setattr(cache, "_query_rows", rows)
    summary = cache._populate(output, source, {}, "bbb_martins")
    assert summary["queries"] == 2
    assert output.execute("SELECT count(DISTINCT query_id) FROM benchmark_queries").fetchone()[0] == 2
    assert output.execute("SELECT count(DISTINCT query_parent_smiles) FROM queries").fetchone()[0] == 1
    assert output.execute("SELECT count(*) FROM assignments").fetchone()[0] == 300
    assert len(indirect_cache._selected_records(
        output, 0, "L2", "morgan_parent_control"
    )) == 25


def _filter_prepared() -> dict:
    active = {}
    for index in range(100):
        molecule = f"m{index // 10}"
        active.setdefault(molecule, {"canonical_smiles": f"CC{index // 10}",
            "morgan_similarity": 0.9 - index / 1000, "_selection_rank": index,
            "cards": {}})
        active[molecule]["cards"][f"c{index}"] = {
            "card_id": f"c{index}", "_canonical_record_id": f"record-{index}",
            "_selection_rank": 2000 + index, "_semantic_bucket_id": f"b{index // 5}",
            "_semantic_rank": index, "reference_smiles": f"CC{index // 10}",
            "endpoint": "scientific endpoint", "support_text": f"evidence {index}",
            "first_seen_level": 2, "measurement_kind": "binary",
            "source_id": "internal-source", "source_values": {"private": index},
            "evidence_family": "internal-family",
        }
    return {"task": "bbb_martins", "query_index": 1, "benchmark_row_id": "q1",
        "level": 2, "query_smiles": "CCO", "condition_sentence": "at pH 7",
        "active_evidence": active, "selection_audit": {
            "matched_control_record_ids": ["record-0", "record-50"],
            "original_semantic_record_ids": [f"record-{index}" for index in range(25)],
            "indirect_record_limit_per_level": 100,
            "indirect_record_limit_for_current_level": 100},
        "retrieval_audit": {"n_visible_records": 100}}


def test_query_filter_materializes_selected_order_and_audit():
    prepared = _filter_prepared()
    rows = record_filter._ordered_cards(prepared)
    candidates, lookup = record_filter._candidate_packet(prepared, rows)
    assert len(candidates) == 100
    assert "_semantic_rank" not in json.dumps(candidates)
    rendered = record_filter._messages(prepared, candidates)[1]["content"]
    assert rendered.count('"record_id":') == 100
    assert rendered.count("## Molecule ") == 10
    assert prepared["query_smiles"] in rendered
    assert prepared["condition_sentence"] in rendered
    for hidden in ("label_scope", "first_seen_level", "measurement_kind",
                   "source_id", "source_values", "evidence_family", "reference_smiles"):
        assert hidden not in rendered
    selected = ["R051", "R001", *[f"R{i:03d}" for i in range(2, 10)]]
    assert record_filter.selection_errors({"selected_record_ids": selected}, set(lookup)) == []
    final = record_filter._materialize(prepared, selected, lookup)
    assert final["n_active_cards"] == 10
    ordered = sorted(final["active_evidence"].values(), key=lambda row: row["_selection_rank"])
    assert ordered[0]["canonical_smiles"] == "CC5"
    receipt = record_filter._selection_receipt(prepared, selected, lookup)
    assert receipt["overlap_control25"] == 2
    assert receipt["overlap_semantic_top25"] == 9


@pytest.mark.parametrize("task", ("bbb_martins", "bioavailability_ma"))
@pytest.mark.parametrize("level", (2, 3, 4))
def test_query_filter_shows_task_level_pitfalls(task, level):
    prepared = _filter_prepared()
    candidates, _ = record_filter._candidate_packet(
        prepared, record_filter._ordered_cards(prepared)
    )
    routed = {**prepared, "task": task, "level": level}
    prompt = record_filter._messages(routed, candidates)[1]["content"]
    assert prompt.count("# PITFALLS") == 1


def test_query_filter_omits_pitfalls_outside_l2_l4():
    prepared = _filter_prepared()
    candidates, _ = record_filter._candidate_packet(
        prepared, record_filter._ordered_cards(prepared)
    )
    prompt = record_filter._messages(
        {**prepared, "level": 1}, candidates
    )[1]["content"]
    assert "# PITFALLS" not in prompt


def test_query_filter_rejects_a_saved_selection_from_another_version(tmp_path):
    prepared = _filter_prepared()
    _, lookup = record_filter._candidate_packet(
        prepared, record_filter._ordered_cards(prepared)
    )
    filter_dir = tmp_path / "filter"
    filter_dir.mkdir()
    (filter_dir / "selection.json").write_text(json.dumps({
        "status": "ok", "version": "query_conditioned_record_filter_v1",
        "records": [{"record_id": f"R{index:03d}"} for index in range(1, 11)],
    }))
    with pytest.raises(ValueError, match="version differs"):
        record_filter._resume(tmp_path, prepared, lookup)


def test_query_filter_rejects_bad_id_counts_and_duplicates():
    known = {f"R{i:03d}" for i in range(1, 101)}
    assert record_filter.selection_errors({"selected_record_ids": ["R001"]}, known)
    assert record_filter.selection_errors({"selected_record_ids": ["R001"] * 10}, known)
    assert record_filter.selection_errors(
        {"selected_record_ids": [*sorted(known)[:9], "R999"]}, known
    )


def test_v2_semantic_candidates_enforce_parent_and_bucket_caps():
    rows = []
    for parent in range(20):
        for record in range(20):
            rows.append({"record_id": f"r-{parent}-{record}", "parent_id": f"p-{parent}",
                "parent_rank": parent + 1, "semantic_bucket_id": f"b-{record // 10}",
                "semantic_rank": record // 10 + 1, "retrieval_eligible": True,
                "tie_key": f"{parent:02d}-{record:02d}"})
    selected = filter_cache._semantic_selection(rows)
    by_parent = Counter(row["parent_id"] for row in selected)
    by_cell = Counter((row["parent_id"], row["semantic_bucket_id"]) for row in selected)
    assert len(selected) == 100
    assert max(by_parent.values()) <= 15
    assert max(by_cell.values()) <= 8
