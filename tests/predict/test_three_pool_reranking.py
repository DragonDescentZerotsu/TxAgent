"""Pool membership, rendering identity, deduplication and retry behavior."""
import json
import hashlib
import sqlite3

import pytest

from predict.retrieval.assay_reranking import three_pools as p


def test_historical_source_requires_exact_verified_archive(tmp_path, monkeypatch):
    live, archived, receipt = (tmp_path / n for n in ("live", "archived", "receipt.json"))
    live.write_text("old")
    archived.write_text("old")
    old_hash = hashlib.sha256(b"old").hexdigest()
    item = {"path": str(live), "sha256": old_hash}
    receipt.write_text(json.dumps({"files": {str(live): {"path": str(archived), "sha256": old_hash}}}))
    monkeypatch.setattr(p, "ARCHIVE_MANIFEST", receipt)
    assert p.verified_source(item) == live
    live.write_text("rebuilt")
    assert p.verified_source(item) == archived
    archived.write_text("corrupt")
    with pytest.raises(ValueError, match="verified archive"):
        p.verified_source(item)


def test_verify_inputs_treats_code_hashes_as_receipts(tmp_path):
    scientific, builder = tmp_path / "records.parquet", tmp_path / "builder.py"
    scientific.write_text("records")
    builder.write_text("new builder")
    version = {
        "schema_version": p.SCHEMA,
        "task_id": p.bbb.TASK_ID,
        "models": p.bbb.MODELS,
        "inputs": {
            "records": {"path": str(scientific), "sha256": p.runtime.file_sha256(scientific)},
            "builder": {"path": str(builder), "sha256": "historical-code-hash"},
        },
    }
    p.verify_inputs(version)
    scientific.write_text("changed records")
    with pytest.raises(ValueError, match="cache input changed"):
        p.verify_inputs(version)


def record(**updates):
    return dict({"record_id": "r", "parent_id": "parent", "progressive_level": "L2",
        "canonical_smiles": "CCO", "has_scalar": True, "tool_accepted": True,
        "assay_transfer_eligible": False, "measurement_kind": "continuous",
        "canonical_measurement_text": "42", "canonical_unit_text": "ng/mL",
        "canonical_pair_fields_json": "{}", "canonical_category_id": None,
        "canonical_measurement_scale_id": "", "finite_scalar_value": 42,
        "source_fields": {"source_smiles": "CCO", "measurement_text": "42", "unit_text": "ng/mL",
                          "support_text": "support 42", "extra_details": "details 42", "species": "rat"}}, **updates)


def test_membership_ignores_record_eligibility():
    assert p.membership(record()) == p.POOLS
    assert p.membership(record(tool_accepted=False)) == ("tool-compatible", "all")
    assert p.membership(record(tool_accepted=False, has_scalar=False)) == ("all",)


@pytest.mark.parametrize("name", [None, "retrieved molecule"])
def test_old_oral_projection_reused_only_when_hidden_field_absent(tmp_path, monkeypatch, name):
    paths = {"root": tmp_path, "cache": tmp_path / "scores.sqlite3", "version": tmp_path / "VERSION.json"}
    monkeypatch.setattr(p.oral, "cache_paths", lambda *args, **kwargs: paths)
    r = record()
    r["source_fields"]["molecule_name"] = name
    c = sqlite3.connect(paths["cache"])
    c.executescript("""
        CREATE TABLE records(record_key INTEGER,payload TEXT);
        CREATE TABLE queries(query_id INTEGER,query_smiles TEXT);
        CREATE TABLE assignments(query_id INTEGER,record_key INTEGER,score_key INTEGER);
        CREATE TABLE scores(score_key INTEGER,transfer_probability REAL);
        INSERT INTO queries VALUES(1,'CCN');
        INSERT INTO assignments VALUES(1,1,1);
        INSERT INTO scores VALUES(1,0.75);
    """)
    c.execute("INSERT INTO records VALUES(1,?)", (json.dumps(r),))
    c.commit()
    c.close()
    renderer = p.Renderer(p.oral.TASK_ID)
    paths["version"].write_text(json.dumps({"status":"complete", "task_id":p.oral.TASK_ID,
        "subset":"valid", "template_hash":renderer.template_hash,
        "projection_hash":"fcfed7fd5c11a789db49a6c2f75d69af19e2fe32cbeb9a5b9001a75c12f0c924",
        "scoring_contract_version":p.runtime.SCORING_CONTRACT_VERSION, "models":p.oral.MODELS,
        "inputs":{}, "cache_sha256":p.runtime.file_sha256(paths["cache"])}))
    if name:
        with pytest.raises(ValueError, match="changes rendered prompt"):
            p.reusable_scores(p.oral.TASK_ID, "valid")
    else:
        scores, receipts = p.reusable_scores(p.oral.TASK_ID, "valid")
        assert scores == {renderer.prompt_task(r, "CCN").cache_key: 0.75}
        assert all(receipt["projection_rendering_unchanged"] for receipt in receipts)


@pytest.mark.parametrize("task", tuple(p.MODULES))
def test_scalar_rendering_and_keys_stay_legacy(task):
    renderer = p.Renderer(task)
    r = record()
    prompt, projection = renderer.render(r, "CCN")
    assert prompt == renderer.legacy.render(r, "CCN")
    assert projection == renderer.legacy.projection_hash
    first = renderer.prompt_task(r, "CCN")
    assert first.cache_key == renderer.prompt_task({**r, "tool_accepted": False}, "CCN").cache_key
    assert first.cache_key != renderer.prompt_task(r, "CCC").cache_key
    assert first.cache_key != renderer.prompt_task({**r, "progressive_level": "L3"}, "CCN").cache_key


@pytest.mark.parametrize("task", tuple(p.MODULES))
def test_text_units_and_result_fields_are_not_copied(task):
    r = record(has_scalar=False, measurement_kind="non_scalar", canonical_unit_text="free-text",
               source_fields={"measurement_text": "high at 5 ng/mL", "unit_text": "free-text",
                              "support_text": "support secret", "extra_details": "detail secret",
                              "molecule_name": "reference identity", "species": "rat"})
    renderer = p.Renderer(task)
    prompt, projection = renderer.render(r, "CCN")
    known, query = prompt.split("Experiment B (query measurement hidden)")
    assert "high at 5 ng/mL" in known
    assert "free-text" not in prompt
    for text in ("high at 5 ng/mL", "support secret", "detail secret", "reference identity"):
        assert text not in query
    assert "rat" in known and "rat" in query
    assert projection != renderer.legacy.projection_hash
    r["source_fields"].pop("measurement_text")
    missing, _ = renderer.render(r, "CCN")
    assert "support secret" in missing and "high at 5 ng/mL" not in missing


def test_shared_scores_and_finalization_retry(tmp_path, monkeypatch):
    task, subset = "bbb_martins", "valid"
    monkeypatch.setattr(p, "ROOT", tmp_path)
    monkeypatch.setattr(p, "verify_inputs", lambda v: None)
    path = p.paths(task, subset)
    path["root"].mkdir(parents=True)
    path["journals"].mkdir()
    p.bbb._write_json(path["version"], {"status": "prepared", "models": p.bbb.MODELS})
    c = sqlite3.connect(path["cache"])
    c.executescript("""
        CREATE TABLE scores(score_key INTEGER PRIMARY KEY,cache_key TEXT UNIQUE,level TEXT,transfer_probability REAL,origin TEXT);
        INSERT INTO scores VALUES(1,'reused','L2',0.25,'exact_cache_reuse');
        INSERT INTO scores VALUES(2,'fresh','L2',NULL,NULL);
        CREATE TABLE prompts(score_key INTEGER PRIMARY KEY,prompt TEXT);
        INSERT INTO prompts VALUES(1,'a'); INSERT INTO prompts VALUES(2,'b');
        CREATE TABLE assignments(pool TEXT,score_key INTEGER);
        INSERT INTO assignments VALUES('tool-accepted',1),('tool-compatible',1),('all',1),('all',2);
    """)
    c.close()
    with pytest.raises(ValueError, match="incomplete"):
        p.finalize(task, subset)
    journal = path["journals"] / "L2-00-of-01.jsonl"
    journal.write_text(json.dumps({"cache_key": "fresh", "transfer_probability": float('nan')}) + "\n")
    with pytest.raises(ValueError, match="invalid"):
        p.finalize(task, subset)
    row = json.dumps({"cache_key": "fresh", "transfer_probability": .75}) + "\n"
    journal.write_text(row + row)
    with pytest.raises(sqlite3.IntegrityError):
        p.finalize(task, subset)
    journal.write_text(row)
    assert p.finalize(task, subset)["status"] == "complete"
    c = sqlite3.connect(path["cache"])
    assert c.execute("SELECT transfer_probability FROM scores ORDER BY score_key").fetchall() == [(.25,), (.75,)]
    assert c.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == 4
    assert c.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2
    assert c.execute("PRAGMA integrity_check").fetchone()[0] == 'ok'
    c.close()
    assert not path["journals"].exists()


def test_prepare_selects_parents_then_expands_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "ROOT", tmp_path / "caches")
    monkeypatch.setattr(p, "POOL_SIZE", 2)
    monkeypatch.setattr(p.bbb, "GOLD_ROOT", tmp_path)
    identity = p.normalize_molecule_identity("CO")
    query = {"benchmark_row_id": "q", "drug": "CO", "molecule_identity": {
        "parent_smiles": identity.parent_smiles, "parent_inchi_key": identity.parent_inchi_key}}
    (tmp_path / "valid_molecule_condition_labels.jsonl").write_text(json.dumps(query) + "\n")
    records = {}
    for level in p.bbb.MODELS:
        for i in range(1, 5):
            smiles = "C" * i
            key = f"{level}-{i}"
            r = record(record_id=key, parent_id=p.normalize_molecule_identity(smiles).parent_inchi_key,
                       progressive_level=level, canonical_smiles=smiles, tool_accepted=i <= 2,
                       has_scalar=i <= 3, measurement_kind="continuous" if i <= 3 else "non_scalar")
            r["source_fields"]["source_smiles"] = smiles
            records[key] = r
            if i == 1:
                records[key + "-copy"] = {**r, "record_id": key + "-copy"}
    monkeypatch.setattr(p, "load_source", lambda task: (records, {}, []))
    monkeypatch.setattr(p, "audit_models", lambda *args: {})
    monkeypatch.setattr(p, "reusable_scores", lambda *args: ({}, []))
    monkeypatch.setattr(p, "l1_reference", lambda *args: {})
    monkeypatch.setattr(p.bbb, "_runtime_metadata", lambda: {})
    version = p.prepare("bbb_martins", "valid")
    assert version["record_eligibility_tag_filters_membership"] is False
    c = sqlite3.connect(p.paths("bbb_martins", "valid")["cache"])
    rows = c.execute("SELECT pool,a.level,COUNT(DISTINCT parent_id),COUNT(*) FROM assignments a "
                     "JOIN records USING(record_key) GROUP BY pool,a.level").fetchall()
    assert len(rows) == 9 and all(n == 2 for _, _, n, _ in rows)
    assert all(n == 3 for pool, _, _, n in rows if pool == 'tool-accepted')
    assert c.execute("SELECT COUNT(*) FROM scores").fetchone()[0] < c.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    assert c.execute("SELECT COUNT(*) FROM assignments a JOIN records r USING(record_key) "
                     "WHERE pool='tool-accepted' AND json_extract(payload,'$.tool_accepted')=0").fetchone()[0] == 0
    c.close()
