"""Canonical card retrieval, exact scores and physical UID export end to end."""

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
import pytest
from predict.retrieval.policies import normalize_molecule_identity
from predict.retrieval.selection import indexed_manifests, read_selection
from predict.retrieval.assay_reranking.cache_matched import load_candidates
from predict.utils.json import sha256_file
from optimization.selection import pool, search, policies, objectives


def _cards(tmp_path):
    train = tmp_path / "train.jsonl"
    query = tmp_path / "valid.jsonl"
    rows = []
    for i in range(100):
        full = "C[NH3+].[Cl-]" if i == 0 else "CN"
        rows.append(
            dict(
                benchmark_row_id=f"card{i:03}",
                drug="CN",
                molecule_identity=normalize_molecule_identity(full).to_dict(),
                Y=i % 2,
                condition_group=f"context-{i}",
                source_record_ids=[f"uid{i:03}"],
            )
        )
    train.write_text("".join((json.dumps(row) + "\n" for row in rows)))
    query.write_text(
        json.dumps(
            dict(
                benchmark_row_id="query",
                drug="CCO",
                Y=1,
                molecule_identity=normalize_molecule_identity("CCO").to_dict(),
            )
        )
        + "\n"
    )
    index = pool.prepare_l1_cache(
        "bbb_martins",
        train,
        {"valid": query},
        tmp_path / "cache",
        identity_policy="parent_disjoint",
    )
    return (train, query, index)


def test_full_molecule_cards_budget_and_prediction_hydration(tmp_path, monkeypatch):
    train, query, index = _cards(tmp_path)
    rows, audit = pool.load_direct(
        index, task="bbb_martins", subset="valid", queries={"query": "CCO"}
    )
    assert len(rows["query"]) == 100
    assert len({row["parent_id"] for row in rows["query"]}) == 1
    assert {row["gold_label"] for row in rows["query"]} == {0, 1}
    assert any(("." in row["canonical_smiles"] for row in rows["query"]))
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps(
            dict(
                name="fixture",
                kind="direct",
                diversity="log",
                profiles=[asdict(objectives.DirectProfile("fixture", 0, 0, 1))],
            )
        )
    )
    monkeypatch.setattr(search, "query_cohort", lambda *a: ({"query": "CCO"}, query))
    result = search.build(
        tmp_path / "search",
        ["bbb_martins"],
        kind="direct",
        subset="valid",
        grid=grid,
        direct_cache=index,
        budget=3,
    )
    document = json.loads(result.read_text())
    manifest = (
        result.parent / document["profiles"][0]["task_manifests"]["bbb_martins"]["path"]
    )
    selected, receipt = read_selection(
        manifest, task="bbb_martins", subset="valid", queries={"query": "CCO"}
    )
    assert len(selected) == 3 and len({row["context_id"] for row in selected}) == 3
    assert search.validate_selection_manifest(manifest)["budget"] == 3
    manifests = indexed_manifests(index, "valid", ("L1",))
    policy = dict(
        stages={"L1": "morgan"},
        cache_index=str(index),
        cache_indexes={"L1": str(index)},
        cache_manifests={level: str(path) for level, path in manifests.items()},
        selection_contract="ranked_uid_retrieval.v1",
    )
    molecules, _, audit = load_candidates(
        {"query": "CCO"},
        task="bbb_martins",
        subset="valid",
        policy=policy,
        molecule_limit=3,
        l1_limit=10,
        cache_pool="all",
        physical_selection=selected,
    )
    assert len(molecules["query"]) == 3
    assert {card["context_card_id"] for card in molecules["query"]} == {
        row["context_id"] for row in selected
    }
    for card in molecules["query"]:
        assert (
            card["canonical_smiles"]
            == card["l1_records"][0]["payload"]["canonical_smiles"]
        )
    from predict.harnesses.branches import flat

    retrieval = flat.cache_matched_flat_retrieval(
        "query",
        "CCO",
        molecules["query"],
        {},
        task="bbb_martins",
        reranking="morgan",
        query_audit=audit["query_audits"]["query"],
        prompt_version="final_full_flat",
        selection_contract="ranked_uid_retrieval.v1",
        preserve_physical_records=True,
    )
    retrieval["benchmark"] = "gold_v1"
    messages, _ = flat.build_flat_context_request(
        retrieval,
        task_id="bbb_martins",
        task_prompt_profile="meaningful_cns_access_v1",
        layout="global",
        reranking="morgan",
        query_prior=None,
        prompt_version="final_full_flat",
    )
    assert "label" in messages[-1]["content"]
    with sqlite3.connect(manifests["L1"].with_name("rankings.sqlite3")) as db:
        db.execute('UPDATE contexts SET gold_label=-1 WHERE context_id="card000"')
    with pytest.raises(ValueError, match="binary label"):
        pool.load_direct(
            index, task="bbb_martins", subset="valid", queries={"query": "CCO"}
        )


def test_exact_score_identity_missing_and_parent_mismatch(tmp_path):
    database = tmp_path / "scores.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE scores(query_id,level,item_id,parent_id,score,score_key)"
        )
        db.execute(
            "INSERT INTO scores VALUES (?,?,?,?,?,?)",
            ("q1", "L1", "card1", "parent", 0.8, "frozen-model-prompt-key"),
        )
    source = tmp_path / "source.json"
    source.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                status="complete",
                task="bbb_martins",
                database=database.name,
                database_sha256=sha256_file(database),
                score_count=1,
                inputs={str(source): sha256_file(source)},
            )
        )
    )
    scores, audit = pool.load_exact_scores(manifest, task="bbb_martins")
    rows = [
        dict(item_id="card1", parent_id="parent"),
        dict(item_id="card2", parent_id="parent"),
    ]
    pool.attach_exact_scores(rows, query_id="q1", level="L1", scores=scores)
    assert (
        rows[0]["assay_transfer_score"] == 0.8
        and rows[1]["assay_transfer_score"] is None
    )
    pool.attach_exact_scores(rows, query_id="q2", level="L1", scores=scores)
    assert all((row["assay_transfer_score"] is None for row in rows))
    with pytest.raises(ValueError, match="parent identity"):
        pool.attach_exact_scores(
            [dict(item_id="card1", parent_id="different")],
            query_id="q1",
            level="L1",
            scores=scores,
        )
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="source changed"):
        pool.load_exact_scores(manifest, task="bbb_martins")


def test_explicit_grids_and_dili_policy():
    direct, _ = policies.load_grid("direct", kind="direct")
    assert len(direct) == 39
    tied, diversity = policies.load_grid("dili_geomean_tied16", kind="indirect")
    assert len(tied) == 16
    policies.validate_indirect(
        tied[0],
        task="dili",
        diversity=diversity,
        dili_scores=Path("scores"),
        direct_manifest=Path("direct"),
    )
    with pytest.raises(ValueError, match="frozen direct"):
        policies.validate_indirect(
            tied[0],
            task="dili",
            diversity=diversity,
            dili_scores=Path("scores"),
            direct_manifest=None,
        )
