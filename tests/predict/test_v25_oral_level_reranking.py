"""Behavioral checks for the Oral V25 level-cache contract."""

import json
from pathlib import Path
import sqlite3

import pytest

from predict.retrieval.assay_reranking import v25_oral_levels as oral

from predict.retrieval.assay_reranking.v25_oral_levels import (
    LEVELS,
    MODELS,
    POOL_SIZE,
    TRAINING_TEMPLATE_SHA256,
    V25OralPromptRenderer,
)


FIXTURE = Path(__file__).parent / "fixtures/v25_oral_prompt.json"


def test_oral_prompt_matches_reviewed_fixture_and_preserves_asymmetry():
    fixture = json.loads(FIXTURE.read_text())
    prompt = V25OralPromptRenderer().render(
        fixture["record"], fixture["query_smiles"]
    )
    assert prompt == fixture["expected_prompt"]
    known, query = prompt.split("Experiment B (query measurement hidden)")
    assert "Known source measurement: 12.3" in known
    assert "Source unit: cm/s" in query
    for hidden in (
        "measured for 60 minutes",
        "Known source measurement: 12.3",
        "reported permeability was 12.3",
    ):
        assert hidden not in query


def test_oral_levels_use_pinned_independent_v25_models():
    assert LEVELS == ("L4", "L6")
    assert POOL_SIZE == 75
    assert V25OralPromptRenderer().template_hash == TRAINING_TEMPLATE_SHA256
    assert MODELS["L4"]["revision"] == "0c3c134a33ef4b67cba78fad228b485ccbd0d05b"
    assert MODELS["L6"]["revision"] == "6c5f76d6afafd0517c7630e8fc374915d4c2b98a"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "invalid_probability"])
@pytest.mark.parametrize("sharded", [False, True])
def test_finalize_can_retry_after_later_level_journal_failure(tmp_path, monkeypatch, failure, sharded):
    paths = {"cache": tmp_path / "scores.sqlite3", "version": tmp_path / "VERSION.json",
             "journals": tmp_path / ".scores"}
    paths["journals"].mkdir()
    monkeypatch.setattr(oral, "cache_paths", lambda *args, **kwargs: paths)
    monkeypatch.setattr(oral, "_runtime_metadata", lambda: {})
    connection = oral._open_build_db(paths["cache"])
    for index, level in enumerate(oral.LEVELS, 1):
        connection.execute("INSERT INTO groups_dim VALUES (?,?)", (index, level))
        connection.execute("INSERT INTO prompt_tasks VALUES (?,?,?)", (index, level, "prompt"))
        connection.execute("INSERT INTO prompt_assignments VALUES (?,?,?,?,?)", (1, index, 1, index, index))
    connection.commit()
    connection.close()
    oral._write_json(paths["version"], {"status": "prepared", "level_assignment_counts": {"L4": 1, "L6": 1}})
    good = lambda level: json.dumps({"cache_key": level, "transfer_probability": 0.75}) + "\n"
    (paths["journals"] / "scores-L4.jsonl").write_text(good("L4"))
    last_journal = paths["journals"] / ("scores-L6-00-of-02.jsonl" if sharded else "scores-L6.jsonl")
    if failure == "duplicate":
        last_journal.write_text(good("L6") * 2)
    elif failure == "invalid_probability":
        last_journal.write_text(json.dumps({"cache_key": "L6", "transfer_probability": 1.5}) + "\n")
    for _ in range(2):
        with pytest.raises(ValueError, match="Oral V25 score journal"):
            oral.finalize("valid")
        with sqlite3.connect(paths["cache"]) as connection:
            assert connection.execute("SELECT COUNT(*) FROM prompt_assignments").fetchone()[0] == 2
            assert not connection.execute("SELECT name FROM sqlite_master WHERE name='prompt_scores'").fetchall()
        assert json.loads(paths["version"].read_text())["status"] == "prepared"
    last_journal.write_text(good("L6"))
    result = oral.finalize("valid")
    assert result["status"] == "complete"
    assert result["n_cached_scores"] == 2
    assert result["level_assignment_counts"] == {"L4": 1, "L6": 1}
    with sqlite3.connect(paths["cache"]) as connection:
        assert connection.execute("SELECT transfer_probability FROM scores ORDER BY score_key").fetchall() == [(0.75,), (0.75,)]


def test_l2_has_separate_cache_paths_and_rejects_mixed_reads():
    assert oral.cache_paths("valid", l2_only=True)["root"] != oral.cache_paths("valid")["root"]
    assert oral.cache_paths("test", l2_only=True)["root"] != oral.cache_paths("valid", l2_only=True)["root"]
    with pytest.raises(ValueError, match="separate L2 cache"):
        oral.load_top_ranked_records({}, subset="valid", levels=("L2", "L4"))


def test_single_level_routing_preserves_l2_and_isolates_l3():
    l2 = oral.cache_paths("valid", single_level="L2")
    assert l2 == oral.cache_paths("valid", l2_only=True)
    l3 = oral.cache_paths("valid", single_level="L3")
    assert l3["root"] not in (l2["root"], oral.cache_paths("valid")["root"])
    assert l3["root"] != oral.cache_paths("test", single_level="L3")["root"]
    with pytest.raises(ValueError, match="either"):
        oral.cache_paths("valid", single_level="L3", l2_only=True)
    with pytest.raises(ValueError, match="Unsupported"):
        oral.cache_paths("valid", single_level="L5")
    with pytest.raises(ValueError, match="independently"):
        oral.load_top_ranked_records({}, subset="valid", levels=("L2", "L3"))
    with pytest.raises(ValueError):
        oral.score("valid", "L3", device=0, num_shards=2, shard_index=2)


def test_copied_query_context_cannot_borrow_reference_molecule_name():
    fixture = json.loads(FIXTURE.read_text())
    fixture["record"]["source_fields"]["molecule_name"] = "REFERENCE_IDENTITY_SENTINEL"
    known, query = V25OralPromptRenderer().render(fixture["record"], fixture["query_smiles"]).split(
        "Experiment B (query measurement hidden)")
    assert "REFERENCE_IDENTITY_SENTINEL" in known
    assert "REFERENCE_IDENTITY_SENTINEL" not in query


def test_prepare_excludes_parent_when_another_tautomer_has_query_scaffold(tmp_path, monkeypatch):
    query = "CC(=O)OCC(CCn1cnc2cnc(N)nc21)COC(C)=O"
    forms = ["Nc1nc2c(ncn2COC(CO)CO)c(=O)[nH]1", "Nc1nc(O)c2ncn(COC(CO)CO)c2n1", "CCO"]
    identity = oral.normalize_molecule_identity(query)
    rows = []
    for index, smiles in enumerate(forms):
        parent = oral.normalize_molecule_identity(smiles)
        rows.append(dict(record_id=str(index), parent_id=parent.parent_inchi_key,
                         parent_smiles=parent.parent_smiles, level="L2", level_family="near_direct",
                         source_id="hf_bioavailability", measurement_kind="continuous",
                         source_fields={"smiles": smiles, "measurement_text": "10%"}))
    assert rows[0]["parent_id"] == rows[1]["parent_id"]
    assert not oral.decide_candidate(identity, {"canonical_smiles": forms[0]}, "scaffold_disjoint").excluded
    assert oral.decide_candidate(identity, {"canonical_smiles": forms[1]}, "scaffold_disjoint").excluded
    paths = {"root": tmp_path, "cache": tmp_path / "scores.sqlite3", "version": tmp_path / "VERSION.json",
             "journals": tmp_path / ".scores", "queries": tmp_path / "queries.jsonl"}
    monkeypatch.setattr(oral, "cache_paths", lambda *args, **kwargs: paths)
    monkeypatch.setattr(oral, "POOL_SIZE", 1)
    monkeypatch.setattr(oral, "_load_records", lambda levels: (rows, {}, {}))
    monkeypatch.setattr(oral, "_read_jsonl", lambda path: [{"drug": query, "molecule_identity": identity.to_dict()}])
    monkeypatch.setattr(oral, "file_sha256", lambda path: oral.TRAINING_TEMPLATE_SHA256)
    result = oral.prepare("valid", pool_size=1, l2_only=True)
    assert result["n_score_assignments"] == 1
    with sqlite3.connect(paths["cache"]) as connection:
        assert connection.execute("SELECT external_record_id FROM records").fetchall() == [("2",)]
