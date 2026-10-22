from rdkit import Chem
import pytest

from predict.retrieval.policies import (
    MOLECULAR_REPRESENTATION_POLICY,
    decide_candidate,
    evidence_smiles,
    normalize_molecule_identity,
    source_smiles,
)


CBD = "C=C(C)[C@@H]1CCC(C)=C[C@H]1c1c(O)cc(CCCCC)cc1O"
THC = "CCCCCc1cc(O)c2c(c1)OC(C)(C)[C@@H]1CCC(C)=C[C@@H]21"


def test_source_parent_and_direct_context_are_distinct():
    source = "CC(=O)[O-]"
    record = {"parent_id": "parent", "canonical_smiles": "CC(=O)O",
              "source_fields": {"source_smiles": source}}
    assert source_smiles(record) == source
    assert evidence_smiles(source) == "CC(=O)O"
    assert evidence_smiles(source, direct=True) == source
    with pytest.raises(ValueError, match="source structure"):
        source_smiles({"parent_id": "parent", "canonical_smiles": "CCO"})


@pytest.mark.parametrize("source", [f"{CBD}.{THC}", "CC(=O)[O-].[Na+]"])
def test_every_multifragment_source_stays_whole(source):
    actual = evidence_smiles(source)
    assert len(Chem.GetMolFrags(Chem.MolFromSmiles(actual))) == 2
    assert actual == evidence_smiles(".".join(reversed(source.split("."))))


@pytest.mark.parametrize("policy", ["parent_disjoint", "scaffold_disjoint"])
def test_exact_component_match_is_retained_without_expanding_to_all_stereoisomers(policy):
    query = normalize_molecule_identity(THC)
    record = {"canonical_smiles": CBD, "source_fields": {"smiles": f"{CBD}.{THC}"},
              "molecular_representation_policy": MOLECULAR_REPRESENTATION_POLICY}
    assert decide_candidate(query, record, policy).excluded
    achiral = Chem.MolToSmiles(Chem.MolFromSmiles(THC), isomericSmiles=False)
    assert not decide_candidate(normalize_molecule_identity(achiral), record, policy).excluded


def test_secondary_scaffolds_and_shared_counterions_do_not_exclude():
    query = normalize_molecule_identity("COc1ccccc1.CCCCC")
    candidate = {"canonical_smiles": "CCc1ccccc1.CCCCCCC"}
    assert not decide_candidate(query, candidate, "parent_disjoint").excluded
    assert not decide_candidate(query, candidate, "scaffold_disjoint").excluded
    query = normalize_molecule_identity("C[NH3+].[Cl-]")
    assert not decide_candidate(query, {"canonical_smiles": "CC[NH3+].[Cl-]"}, "parent_disjoint").excluded


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("safe_sibling", [False, True])
def test_source_filter_runs_before_parent_capacity_and_keeps_eligible_siblings(tmp_path, monkeypatch, shared, safe_sibling):
    import json
    import sqlite3
    import pyarrow as pa
    import pyarrow.parquet as pq
    from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder
    from predict.utils.json import sha256_file

    query = "NC(=O)c1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c1O"
    hidden_source = "NC(=O)c1[nH+]cn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c1[O-]"
    represented = evidence_smiles(hidden_source)
    identity = normalize_molecule_identity(query)
    assert not decide_candidate(identity, {"canonical_smiles": represented}, "parent_disjoint").excluded
    assert decide_candidate(identity, {"canonical_smiles": hidden_source}, "parent_disjoint").excluded
    sources = [("hidden", "group", represented, hidden_source), ("other", "other", "CCO", "CCO")]
    if safe_sibling:
        sources.append(("sibling", "group", represented, represented))
    rows = [dict(source_row_uid=uid, external_record_id=uid, parent_id=parent, parent_smiles=display,
                 level="L2", payload=json.dumps(dict(source_fields={"smiles": source},
                     canonical_smiles=display, molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY)))
            for uid, parent, display, source in sources]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "records.parquet")
    manifest = tmp_path / "VERSION.json"
    manifest.write_text(json.dumps(dict(records="records.parquet", records_sha256=sha256_file(tmp_path / "records.parquet"),
        molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY)))
    query_path = tmp_path / "query.jsonl"
    query_path.write_text("{}\n")
    for key, value in dict(REPRESENTATION_POLICY=MOLECULAR_REPRESENTATION_POLICY, CAPACITY=1,
        TASK_LEVELS={"bioavailability_ma": ("L2",)}, SHARED_LATER_PARENT_UNIVERSE=shared,
        FILTER_PARENT_RELATIONS=True, LATER_IDENTITY_POLICY="parent_disjoint", BASE_RANKING_ROOT=None,
        QUERY_IDENTITY_ROOT=None, SOURCE_PARENT_OFFSET=0, SCORE_REUSE_ROOTS=()).items():
        monkeypatch.setattr(builder, key, value)
    monkeypatch.setattr(builder, "_queries", lambda *args: (query_path, [dict(benchmark_row_id="q", drug=query)],
        [("q", query, identity.parent_inchi_key, identity.parent_smiles)]))
    monkeypatch.setattr(builder, "_model_spec", lambda *args: None)
    output = tmp_path / "out"
    if shared:
        builder.prepare_shared_parent_universe("bioavailability_ma", "test", output, manifest)
    builder.prepare_level("bioavailability_ma", "test", "L2", output, manifest)
    with sqlite3.connect(output / "bioavailability_ma/scaffold/test/L2/rankings.sqlite3") as connection:
        assert connection.execute("SELECT item_id FROM rankings").fetchall() == [("sibling" if safe_sibling else "other",)]


def test_projection_scoring_keeps_source_but_groups_by_effective_structure(tmp_path, monkeypatch):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder

    source = "CC(=O)[O-]"
    payload = {"source_smiles": source, "source_fields": {"smiles": source},
               "finite_scalar_value": 2.0,
               "molecular_representation_policy": MOLECULAR_REPRESENTATION_POLICY}
    pq.write_table(pa.Table.from_pylist([{
        "source_row_uid": "sr_test", "external_record_id": "record", "parent_id": "parent",
        "parent_smiles": "CC(=O)O", "level": "L2", "payload": json.dumps(payload),
    }]), tmp_path / "records.parquet")
    manifest = tmp_path / "VERSION.json"
    manifest.write_text(json.dumps({"records": "records.parquet", "molecular_representation_policy": MOLECULAR_REPRESENTATION_POLICY}))
    monkeypatch.setattr(builder, "REPRESENTATION_POLICY", MOLECULAR_REPRESENTATION_POLICY)
    records, grouped = builder._records(manifest)
    assert records["sr_test"]["canonical_smiles"] == source
    assert records["sr_test"]["representation_smiles"] == "CC(=O)O"
    assert grouped == {"L2": {"parent": ["sr_test"]}}


def test_hydration_and_flat_cards_keep_multifragment_entities_whole():
    import json
    from predict.retrieval.assay_reranking.ranked_uid_retrieval import _payload
    from predict.harnesses.branches.flat import cache_matched_flat_retrieval

    sources = ["CC(=O)O", "CC(=O)[O-]", "CC(=O)[O-].[Na+]", "CC(=O)[O-].[K+]"]
    selected = []
    for index, source in enumerate(sources):
        effective = evidence_smiles(source)
        identity = normalize_molecule_identity(effective)
        payload = dict(source_row_uid=f"sr_{index}", record_id=str(index), task_id="bioavailability_ma",
                       source_id="toy", progressive_level="L2", family_key="passive",
                       canonical_smiles=effective, source_smiles=source, source_fields={"smiles": source},
                       source_contract={"contract_version": "source_column_contract.v2",
                                        "source_or_simply_cleaned": {"smiles": True}},
                       measurement_kind="continuous", molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY)
        ranked = dict(parent_id=identity.standard_inchi_key, parent_smiles=effective,
                      morgan_similarity=0.2, morgan_rank=index+1, assay_rank=None)
        selected.append(_payload(dict(external_record_id=str(index), payload=json.dumps(payload)), ranked, "morgan"))
        assert source_smiles(selected[-1]["payload"]) == source
    retrieval = cache_matched_flat_retrieval("query", "CCCC", [], {"L2": {"records": selected}},
        task="bioavailability_ma", reranking="morgan", query_audit={}, preserve_physical_records=True)
    neighbors = retrieval["groups"][0]["neighbors"]
    assert [n["canonical_smiles"] for n in neighbors] == [evidence_smiles(sources[i]) for i in (0, 2, 3)]
    assert [len(n["evidence_rows"]) for n in neighbors] == [2, 1, 1]
    assert all(n["similarity"] == 0.2 for n in neighbors)


def test_pending_score_reuse_preserves_candidates_and_original_preparation(tmp_path, monkeypatch):
    import json
    import sqlite3
    import pyarrow as pa
    import pyarrow.parquet as pq
    from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder

    level = tmp_path / "task/scaffold/test/L2"
    level.mkdir(parents=True)
    with sqlite3.connect(level / "rankings.sqlite3") as connection:
        builder._schema(connection, "L2")
        for index, key in enumerate(("reuse", "new")):
            connection.execute("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ('q', str(index), 'parent', 'CCO', .2, 1, index+1, index+1,
                 None, None, None, None, None, None, key))
    pq.write_table(pa.table(dict(score_key=["reuse", "new"], prompt=["old prompt", "new prompt"],
                                projection_hash=["projection", "projection"])), level / "prompts.parquet")
    manifest = dict(status="prepared", model={"revision": "pinned"}, database="rankings.sqlite3",
                    molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY,
                    prompts="prompts.parquet", scoring_contract_version=builder.runtime.SCORING_CONTRACT_VERSION,
                    score_counts={"exact_reuse": 0, "pending": 2})
    (level / "VERSION.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(builder, "_model_spec", lambda *args: manifest["model"])
    monkeypatch.setattr(builder, "REPRESENTATION_POLICY", MOLECULAR_REPRESENTATION_POLICY)
    monkeypatch.setattr(builder, "_reuse_scores", lambda *args, **kwargs: {"reuse": .7})
    updated = builder.reuse_prepared_scores("task", "test", "L2", tmp_path)
    assert updated["score_counts"] == {"exact_reuse": 1, "pending": 1}
    assert pq.read_table(level / "prompts.parquet")["score_key"].to_pylist() == ["new"]
    with sqlite3.connect(level / "rankings.sqlite3") as connection:
        assert connection.execute("SELECT item_id,assay_transfer_score FROM rankings ORDER BY item_id").fetchall() == [("0", .7), ("1", None)]
    with sqlite3.connect(level / "before_score_reuse/rankings.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(assay_transfer_score) FROM rankings").fetchone() == (0,)
    monkeypatch.setattr(builder, "_reuse_scores", lambda *args, **kwargs: {"new": .8})
    updated = builder.reuse_prepared_scores("task", "test", "L2", tmp_path)
    assert updated["status"] == "ready_to_finalize"
    assert updated["score_counts"] == {"exact_reuse": 2, "pending": 0}
    assert len(updated["score_reuse_history"]) == 2


def test_scoring_workset_deduplicates_policies_and_rejects_conflicting_prompts(tmp_path, monkeypatch):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from predict.retrieval.assay_reranking import rebuild_policy

    monkeypatch.setattr(rebuild_policy, "REPO_ROOT", tmp_path)
    jobs = [dict(id=policy, task="bioavailability_ma", role="l2plus", levels=["L2"],
                 output=policy, morgan_only=(policy == "parent")) for policy in ("parent", "scaffold")]
    for job, keys in zip(jobs, (["a", "b"], ["b", "c"])):
        for subset in ("test", "valid"):
            level = tmp_path / job["output"] / job["task"] / "scaffold" / subset / "L2"
            level.mkdir(parents=True)
            (level / "VERSION.json").write_text(json.dumps(dict(status="prepared", model={"revision": "pinned"},
                scoring_contract_version="contract", prompts="prompts.parquet", score_reuse_policy="first_declared_source")))
            pq.write_table(pa.Table.from_pylist([dict(score_key=k, prompt=k, projection_hash="projection") for k in keys]),
                           level / "prompts.parquet")
    output = tmp_path / "workset"
    workset = rebuild_policy.prepare_score_worksets(tmp_path, jobs, output)
    assert workset["jobs"][0]["subsets"] == ["test"]
    assert workset["jobs"][0]["morgan_only"] is False
    assert len(workset["source_preparations"]) == 4
    path = output / "bioavailability_ma/scaffold/test/L2/prompts.parquet"
    assert pq.read_table(path)["score_key"].to_pylist() == ["a", "b", "c"]
    source = tmp_path / "scaffold/bioavailability_ma/scaffold/test/L2/prompts.parquet"
    pq.write_table(pa.Table.from_pylist([dict(score_key="b", prompt="changed", projection_hash="projection")]), source)
    with pytest.raises(ValueError, match="different rendered prompts"):
        rebuild_policy.prepare_score_worksets(tmp_path, jobs, tmp_path / "corrupted_workset")


@pytest.mark.parametrize("preserve_uncomputed", [False, True])
def test_direct_reuse_proves_identical_prompt_despite_config_hash_change(tmp_path, monkeypatch, preserve_uncomputed):
    import hashlib
    import json
    import sqlite3
    from types import SimpleNamespace
    from predict.retrieval.assay_reranking import build_ranked_uid_retrieval as builder
    from predict.retrieval.assay_reranking import runtime, v9

    spec = {"model": "test/model", "revision": "a" * 40}
    renderer = SimpleNamespace(template_hash="template", projection_hash="new_config",
                               render=lambda known, query: f"{known['drug']}->{query['drug']}")
    source_query = "CC(=O)[O-]"
    query_path = tmp_path / "test_molecule_condition_labels.jsonl"
    query_path.write_text("{}\n")
    query_path.with_name("train_molecule_condition_labels.jsonl").write_text(json.dumps(
        {"benchmark_row_id": "card", "molecule_identity_key": "known", "drug": "CCO", "Y": 1}) + "\n")
    old = tmp_path / "old"
    old.mkdir()
    prompt_hash = hashlib.sha256(b"CCO->CC(=O)O").hexdigest()
    old_key = v9._cache_key(prompt_hash, spec, SimpleNamespace(template_hash="template", projection_hash="old_config"))
    with sqlite3.connect(old / "scores.sqlite3") as connection:
        connection.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
        connection.execute("INSERT INTO metadata VALUES ('content_id','old')")
        connection.execute("CREATE TABLE rankings(benchmark_row_id TEXT,score_key TEXT,assay_transfer_score REAL)")
        connection.execute("INSERT INTO rankings VALUES ('q',?,0.8)", (old_key,))
        connection.execute("INSERT INTO rankings VALUES ('uncomputed',NULL,NULL)")
    (old / "VERSION.json").write_text(json.dumps(dict(status="complete", model=spec, content_id="old",
        database="scores.sqlite3", scoring_contract_version=runtime.SCORING_CONTRACT_VERSION,
        prompt_assets={"template_sha256": "template", "projection_sha256": "old_config"})))
    index = old / "RELEASE_INDEX.json"
    index.write_text(json.dumps({"splits": {"test": {"levels": {"L1": {"manifest": "VERSION.json"}}}}}))
    root = tmp_path / "new"
    level = root / "task/scaffold/test/L1"
    level.mkdir(parents=True)
    with sqlite3.connect(level / "rankings.sqlite3") as connection:
        builder._schema(connection, "L1")
        connection.execute("INSERT INTO rankings VALUES ('q','card','known','CCO',0.1,1,1,1,NULL,NULL,'card',NULL,1,NULL,NULL)")
        if preserve_uncomputed:
            connection.execute("INSERT INTO rankings VALUES ('uncomputed','card','known','CCO',0.1,1,1,1,NULL,NULL,'card',NULL,1,NULL,NULL)")
    (level / "VERSION.json").write_text(json.dumps(dict(status="complete", model=None,
        molecular_representation_policy=MOLECULAR_REPRESENTATION_POLICY,
        database="rankings.sqlite3", inputs={}, query_counts={"q": {"morgan_candidate_records": 1}})))
    monkeypatch.setattr(builder, "REPRESENTATION_POLICY", MOLECULAR_REPRESENTATION_POLICY)
    monkeypatch.setattr(builder, "DIRECT_MODEL", spec)
    monkeypatch.setattr(builder, "SCORING_FAMILY", "default")
    monkeypatch.setattr(builder, "PRESERVE_UNCOMPUTED_QUERIES", preserve_uncomputed)
    monkeypatch.setattr(builder, "QUERY_BENCHMARK", "tdc")
    monkeypatch.setattr(builder, "REUSE_ROOT", tmp_path / "absent")
    monkeypatch.setattr(builder, "SCORE_REUSE_ROOTS", (index,))
    monkeypatch.setattr(builder, "_renderer", lambda *args: renderer)
    monkeypatch.setattr(builder, "_queries", lambda *args: (
        query_path, [{"benchmark_row_id": "q", "drug": source_query}], []))
    result = builder.prepare_direct_scores("task", "test", root)
    assert result["score_counts"] == {"exact_reuse": 1, "fresh": 0, "pending": 0}
    assert result["identical_prompt_rekey_count"] == 1
    assert result["status"] == "ready_to_finalize"
    if preserve_uncomputed:
        assert result["scored_query_ids"] == ["q"]
        with sqlite3.connect(level / "rankings.sqlite3") as connection:
            assert connection.execute("SELECT score_key,assay_transfer_score FROM rankings WHERE benchmark_row_id='uncomputed'").fetchone() == (None, None)
