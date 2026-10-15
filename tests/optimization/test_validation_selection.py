"""Frozen direct composition and validation-only winner admission."""

import csv
import json
from pathlib import Path

import pytest

from optimization.selection import search
from predict.utils.json import sha256_file


def test_disjoint_prefilter_is_explicit_and_records_physical_exclusions(tmp_path, monkeypatch):
    from optimization.selection import pool
    from predict.retrieval.assay_reranking import ranked_uid_retrieval

    manifest = tmp_path / "VERSION.json"
    manifest.write_text(json.dumps(dict(neighbor_identity_policy="scaffold_disjoint")))
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(splits={"valid": {"levels": {"L2": {}}}})))
    rows = {"L2": {"q": [dict(item_id="bad", parent_smiles="Oc1ccccc1"),
                            dict(item_id="good", parent_smiles="CCO", assay_transfer_score=.7)]}}
    monkeypatch.setattr(ranked_uid_retrieval, "load_ranked_universe", lambda *a, **kw: (rows, manifest, {}))
    monkeypatch.setattr(pool, "indexed_manifests", lambda *a: {"L2": manifest})
    monkeypatch.setattr(pool, "_semantic_ledger", lambda *a: ({("good", "L2"): {"semantic_weight": .5}}, {}))
    with pytest.raises(ValueError, match="same_scaffold"):
        pool.load_indirect(index, task="bioavailability_ma", subset="valid", queries={"q": "c1ccccc1"})
    chosen, audit = pool.load_indirect(index, task="bioavailability_ma", subset="valid",
        queries={"q": "c1ccccc1"}, exclude_disjoint_violations=True)
    assert chosen["q"] == [dict(item_id="good", parent_smiles="CCO", assay_transfer_score=.7,
                                semantic_weight=.5, level="L2")]
    assert audit["disjoint_candidate_filter"] == dict(enabled=True, exclusions=[dict(
        benchmark_row_id="q", level="L2", source_row_uid="bad", reason="scaffold_disjoint:same_scaffold")])


def test_frozen_gold_direct_keeps_order_and_cache_scores(tmp_path, monkeypatch):
    records = tmp_path / "selected_contexts.tsv"
    with records.open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["benchmark_row_id", "selection_rank", "context_id", "parent_id", "gold_label"])
        for query in ("chosen", "other"):
            writer.writerows((query, i + 1, f"card-{9-i}", "parent", i % 2) for i in range(10))
    direct = tmp_path / "direct.json"
    from predict.harnesses.branches.flat import PRESELECTED_DIRECT_SCHEMAS
    direct.write_text(json.dumps(dict(schema_version=next(iter(PRESELECTED_DIRECT_SCHEMAS)),
        status="complete", task_id="bioavailability_ma", benchmark="gold_v1", budget=10,
        benchmark_row_ids=["chosen", "other"], objective={"fixed": True},
        records=dict(path=records.name, sha256=sha256_file(records), row_count=20))))
    indirect = tmp_path / "indirect.json"
    selected = tmp_path / "indirect.tsv"
    selected.write_text("benchmark_row_id\tlevel\tsource_row_uid\nchosen\tL2\tuid-2\n")
    child = tmp_path / "uids.json"
    child.write_text(json.dumps(dict(cache_manifests={"L2": {"path": "cache-L2"}})))
    document = dict(kind="indirect", benchmark="gold_v1", task_id="bioavailability_ma",
        subset="valid_small", benchmark_row_ids=["chosen"], budget=50,
        records={"path": selected.name}, uid_selection={"path": child.name})
    indirect.write_text(json.dumps(document))
    bundle = tmp_path / "cache.json"
    bundle.write_text("{}")
    monkeypatch.setattr(search, "validate_selection_manifest", lambda p: document)
    monkeypatch.setattr(search, "query_cohort", lambda *a: ({"chosen": "CCO"}, Path("queries")))
    monkeypatch.setattr(search, "cache_task", lambda *a: {"retrieval": {"L1": "cache-L1"}})
    monkeypatch.setattr(search, "indexed_manifests", lambda *a: {"L1": Path("cache-L1")})
    captured = {}
    def export(output, rows, **kwargs):
        captured.update(rows=rows, **kwargs)
        return output / "manifest.json"
    monkeypatch.setattr(search, "export_selection", export)
    result = search.compose(direct, indirect, tmp_path / "combined/manifest.json", cache_bundle=bundle)
    assert result.name == "manifest.json"
    assert [r["context_id"] for r in captured["rows"][:10]] == [f"card-{i}" for i in range(9, -1, -1)]
    assert captured["rows"][-1]["source_row_uid"] == "uid-2"
    assert captured["subset"] == "valid" and captured["queries"] == {"chosen": "CCO"}
    assert captured["provenance"]["score_policy"] == "unchanged_cache_scores"
    assert "optimizer_snapshot" not in captured["provenance"]
    records.write_text(records.read_text() + "corrupted\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        search.compose(direct, indirect, tmp_path / "combined/manifest.json", cache_bundle=bundle)


def test_validation_ranking_ties_and_rejects_incomplete_or_test(tmp_path):
    candidates = []
    for name, f1, accuracy in (("z", .8, .9), ("a", .8, .9), ("b", .8, .85)):
        batch = tmp_path / name
        batch.mkdir()
        (batch / "manifest.json").write_text(json.dumps(dict(schema_version="prediction_batch.v2",
            run_uid=name, batch_id="default", settings=dict(evaluation_subset="valid",
            model="openai/gpt-6-luna", benchmark_release="gold_v1"), queries={"3": "q1", "8": "q2"})))
        (batch / "metrics.json").write_text(json.dumps(dict(n_total=2, n_evaluable=2,
            n_successful=2, n_failed_runs=0, macro_f1=f1, accuracy=accuracy)))
        (batch / "predictions.jsonl").write_text("\n".join(json.dumps(dict(query_index=i,
            run_id=f"default_idx{i:05}", status="ok", final_status="ok", pred_label=1)) for i in (3, 8)))
        for i in (3, 8):
            run = batch / "runs" / f"default_idx{i:05}"
            run.mkdir(parents=True)
            (run / "final_reasoning_output.json").write_text(json.dumps(dict(
                status="ok", llm=dict(model="openai/gpt-6-luna", provider="OpenAI"))))
        candidates.append(dict(profile=name, batch=str(batch)))
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps(dict(benchmark="gold_v1", model="openai/gpt-6-luna", provider="OpenAI",
        selection_cohort="valid_small", benchmark_row_ids=["q1", "q2"], candidates=candidates)))
    output = tmp_path / "winner.json"
    assert search.rank(source, output)["winner"] == "a"
    with pytest.raises(FileExistsError):
        search.rank(source, output)
    metrics = tmp_path / "z/metrics.json"
    broken = json.loads(metrics.read_text())
    broken["n_successful"] = 1
    metrics.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Incomplete"):
        search.rank(source, tmp_path / "missing.json")
    broken["n_successful"] = 2
    metrics.write_text(json.dumps(broken))
    manifest = tmp_path / "z/manifest.json"
    broken = json.loads(manifest.read_text())
    broken["settings"]["evaluation_subset"] = "test"
    manifest.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Incomplete"):
        search.rank(source, tmp_path / "test.json")
