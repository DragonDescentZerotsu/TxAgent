from predict.tasks.bbb_martins.prediction_filter import prediction_only
import json
import sqlite3
from dataclasses import asdict
import pandas as pd
import pytest
from optimization.selection import pool, search, objectives
from predict.utils.json import sha256_file


def test_bbb_filter_keeps_measured_assay_outcomes_and_excludes_model_only_cards():
    cases = [
        ({"support_text": "QikProp predicted logBB = -1.2"}, True),
        (
            {"assay_context": "SwissADME", "support_text": "Predicted BBB negative"},
            True,
        ),
        (
            {
                "support_text": "Predicted CNS+; no PAMPA assay was performed",
                "experimental_details": {"assay_type": "PAMPA"},
            },
            True,
        ),
        (
            {"support_text": "PAMPA-BBB showed low Pe; classified as CNS negative"},
            False,
        ),
        (
            {
                "support_text": "Predicted permeability; measured brain uptake also reported"
            },
            False,
        ),
        (
            {
                "assay_context": "in vitro BBB model",
                "support_text": "The assay showed a permeability coefficient",
            },
            False,
        ),
    ]
    for card, expected in cases:
        assert prediction_only({"card": card}) is expected


@pytest.mark.parametrize("canonical", (True, False))
def test_filter_preparation_search_and_source_bindings(tmp_path, monkeypatch, canonical):
    from predict.harnesses.branches import flat
    from predict.retrieval.assay_reranking.build_ranked_uid_retrieval import _schema

    database = tmp_path / "rankings.sqlite3"
    from predict.retrieval.policies import normalize_molecule_identity
    query_identity = normalize_molecule_identity("CCO")
    with sqlite3.connect(database) as db:
        _schema(db, "L2")
        db.execute("INSERT INTO metadata VALUES ('content_id','fixture-level')")
        db.execute("INSERT INTO queries VALUES (?,?,?,?)", (
            "q", "CCO", query_identity.parent_inchi_key, query_identity.parent_smiles,
        ))
        for rank, smiles in enumerate(("CN", "CCl"), 1):
            identity = normalize_molecule_identity(smiles)
            db.execute("INSERT INTO rankings (benchmark_row_id,item_id,parent_id,parent_smiles,"
                       "morgan_similarity,parent_morgan_rank,within_parent_rank,morgan_rank) "
                       "VALUES (?,?,?,?,?,?,?,?)", (
                           "q", f"u{rank}", identity.parent_inchi_key, identity.parent_smiles,
                           0.5, rank, 1, rank,
                       ))
    manifest = tmp_path / "VERSION.json"
    manifest.write_text(json.dumps(dict(
        schema_version="ranked_uid_retrieval.v1", status="complete", task_id="bbb_martins",
        subset="valid", level="L2", pool="all", parent_capacity=100,
        database=database.name, database_sha256=sha256_file(database), content_id="fixture-level",
        query_counts={"q": dict(candidate_records=2, candidate_parents=2)},
    )))
    records = tmp_path / "records.parquet"
    pd.DataFrame([
        dict(source_row_uid=f"u{i}", external_record_id=f"r{i}", payload=json.dumps(
            dict(support_text=text)))
        for i, text in enumerate(("Measured brain uptake", "QikProp predicted logBB = -1.2"), 1)
    ]).to_parquet(records, index=False)
    projection = tmp_path / "evidence.json"
    projection.write_text(json.dumps(dict(
        schema_version="ranked_evidence_projection.v1", status="complete", task_id="bbb_martins",
        records=records.name, content_id="fixture-evidence",
    )))
    index = tmp_path / "RELEASE_INDEX.json"
    index.write_text(json.dumps(dict(
        schema_version="ranked_uid_task_release_index.v1", status="complete", task_id="bbb_martins",
        pool="all", parent_capacity=100,
        later_candidate_universe="all_uids_under_morgan_top_100_parents",
        neighbor_identity_policy_by_level={"L2": "parent_disjoint"},
        evidence=dict(manifest=projection.name, manifest_sha256=sha256_file(projection), content_id="fixture-evidence"),
        splits={"valid": dict(levels={"L2": dict(manifest=manifest.name,
            manifest_sha256=sha256_file(manifest), content_id="fixture-level")})},
    )))
    bundle = tmp_path / "bundle.json"
    configuration = dict(benchmark="tdc", tasks={"bbb_martins": dict(
        retrieval=dict(later=str(index)), selection=dict(direct_cache=str(index)),
    )}) if canonical else dict(caches={"bbb_martins": dict(later=str(index))})
    bundle.write_text(json.dumps(configuration))
    monkeypatch.setattr(flat, "_source_semantic_card", lambda payload, **kwargs: (payload, {}))
    semantics = {(f"u{i}", "L2"): dict(semantic_bucket_id="family", semantic_weight=0.5,
        semantic_weight_source="reviewed_membership_mean") for i in (1, 2)}
    monkeypatch.setattr(pool, "_semantic_ledger", lambda *args: (semantics, {}))
    filter_path = pool.prepare_bbb_filter("valid", tmp_path / "filter", cache_bundle=bundle)
    rows, audit = pool.load_indirect(index, task="bbb_martins", subset="valid", queries={"q": "CCO"},
        prediction_filter=filter_path, cache_bundle=bundle)
    assert [row["item_id"] for row in rows["q"]] == ["u1"]
    assert set(audit["prediction_filter"]) == {str(filter_path), str(filter_path.with_name("eligibility.tsv"))}
    query = tmp_path / "query.jsonl"
    query.write_text(json.dumps(dict(benchmark_row_id="q", drug="CCO")))
    monkeypatch.setattr(search, "query_cohort", lambda *args: ({"q": "CCO"}, query))
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps(dict(kind="indirect", diversity="log", profiles=[
        asdict(objectives.IndirectProfile("fixture", 0, 0, 1, 0, 0, ())),
    ])))
    result = search.build(tmp_path / "search", ["bbb_martins"], kind="indirect", subset="valid",
        cache_bundle=bundle, grid=grid, budget=1, prediction_filter=filter_path)
    selected = result.parent / json.loads(result.read_text())["profiles"][0]["task_manifests"]["bbb_martins"]["path"]
    assert search.validate_selection_manifest(selected)["inputs"]["pool"]["prediction_filter"] == audit["prediction_filter"]

    original_bundle = bundle.read_text()
    bundle.write_text(original_bundle + "\n")
    with pytest.raises(ValueError, match="candidate input differs"):
        pool.load_indirect(index, task="bbb_martins", subset="valid", queries={"q": "CCO"},
            prediction_filter=filter_path, cache_bundle=bundle)
    bundle.write_text(original_bundle)
    original = filter_path.read_text()
    document = json.loads(original)
    document["inputs"][str(index)] = "wrong"
    filter_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="candidate input differs"):
        pool.load_indirect(index, task="bbb_martins", subset="valid", queries={"q": "CCO"},
            prediction_filter=filter_path, cache_bundle=bundle)
    filter_path.write_text(original)
    table = filter_path.with_name("eligibility.tsv")
    table.write_text("source_row_uid\tprediction_only\nu1\t0\n")
    with pytest.raises(ValueError, match="decisions changed"):
        pool.load_indirect(index, task="bbb_martins", subset="valid", queries={"q": "CCO"},
            prediction_filter=filter_path, cache_bundle=bundle)
    with pytest.raises(ValueError, match="selection input changed"):
        search.validate_selection_manifest(selected)
    document = json.loads(original)
    document["files"]["eligibility.tsv"] = dict(sha256=sha256_file(table), rows=1)
    document["excluded_count"] = 0
    filter_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="omits candidate UIDs"):
        pool.load_indirect(index, task="bbb_martins", subset="valid", queries={"q": "CCO"},
            prediction_filter=filter_path, cache_bundle=bundle)
