from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from optimization import dili_score
from optimization.selection import record_scores as cache
from predict.utils.json import sha256_file, write_json_atomic


@pytest.mark.parametrize("tamper", [None, "version", "code"])
def test_score_cache_keeps_source_identity_after_relocation(tmp_path, tamper):
    release = tmp_path / "release.json"
    release.write_text('{"task_id":"dili"}\n')
    records = tmp_path / "records.parquet"
    records.write_bytes(b"frozen source identity")
    scores = tmp_path / "scores.parquet"
    pq.write_table(
        pa.table({"source_row_uid": ["one", "two"], "tool_score": [0.25, 1.0]}), scores
    )
    package = Path(dili_score.__file__).parent
    manifest = {
        "schema_version": cache.SCHEMA,
        "status": "complete",
        "scorer_commit": cache.SCORER_COMMIT,
        "scorer_version": dili_score.DILI_SIMPLE_VERSION,
        "scorer_files_sha256": {
            name: sha256_file(package / name) for name in cache.SCORER_FILES
        },
        "release_index": str(release),
        "release_index_sha256": sha256_file(release),
        "canonical_records": str(records),
        "canonical_records_sha256": sha256_file(records),
        "scores": {"path": scores.name, "sha256": sha256_file(scores), "row_count": 2},
    }
    if tamper == "version":
        manifest["scorer_version"] = "different_rules"
    if tamper == "code":
        manifest["scorer_files_sha256"]["scoring.py"] = "0" * 64
    path = tmp_path / "manifest.json"
    write_json_atomic(path, manifest)
    if tamper:
        message = (
            "different scientific rules"
            if tamper == "version"
            else "scoring code differs"
        )
        with pytest.raises(ValueError, match=message):
            cache.load_scores(path)
    else:
        values, audit = cache.load_scores(path)
        assert values == {"one": 0.25, "two": 1.0}
        assert audit["scorer_commit"] == cache.SCORER_COMMIT


def oral_inputs(root, rows=None, task="bioavailability_ma"):
    import sqlite3
    root.mkdir()
    records = root / "records.parquet"
    if rows is None:
        rows = [dict(source_row_uid="one", source_id="fa", endpoint="intestinal permeability",
                     biological_system="Caco-2", reported_result="12", support_text="Source"),
                dict(source_row_uid="two", source_id="oral_exposure", endpoint="oral bioavailability",
                     biological_system="human", reported_result="0", support_text="Source")]
    pq.write_table(pa.Table.from_pylist(rows), records)
    evidence = root / "evidence.json"
    write_json_atomic(evidence, dict(status="complete", task_id=task,
                                    inputs=dict(records=sha256_file(records))))
    levels = {}
    for level in ("L2", "L6"):
        database = root / f"{level}.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE rankings (item_id TEXT)")
            connection.executemany("INSERT INTO rankings VALUES (?)", [("one",), ("two",), ("one",)])
        manifest = root / f"{level}.json"
        write_json_atomic(manifest, dict(database=database.name))
        levels[level] = dict(manifest=manifest.name, manifest_sha256=sha256_file(manifest))
    release = root / "release.json"
    write_json_atomic(release, dict(task_id=task, status="complete",
        evidence=dict(manifest=evidence.name, manifest_sha256=sha256_file(evidence)),
        splits={split: dict(levels=levels) for split in ("valid", "test")}))
    return release, records, root / "scores"


def test_oral_cache_builds_available_levels_and_roundtrips(tmp_path):
    inputs = oral_inputs(tmp_path / "inputs")
    assert cache.candidate_uids(inputs[0]) == {"one", "two"}
    manifest = cache.build_cache(*inputs)
    scores, audit = cache.load_scores(manifest)
    assert scores == {"one": .75, "two": 1.0}
    saved = json.loads(manifest.read_text())
    assert saved["schema_version"] == cache.BIOAVAILABILITY_SCHEMA
    assert saved["task_id"] == "bioavailability_ma"
    assert saved["scorer_commit"] == ""
    assert "dili_score/scoring.py" in saved["scorer_files_sha256"]
    assert audit["scorer_version"] == "bioavailability_relevance_completeness.v2"
    assert pq.read_table(manifest.parent / "scores.parquet")["rule_ids"].to_pylist()
    with pytest.raises(FileExistsError):
        cache.build_cache(*inputs)


@pytest.mark.parametrize("tamper", ("missing", "duplicate", "evidence", "manifest"))
def test_oral_cache_rejects_incompatible_candidates_before_publication(tmp_path, tamper):
    import json
    release, records, output = oral_inputs(tmp_path / "inputs")
    if tamper in ("missing", "duplicate"):
        rows = pq.read_table(records).to_pylist()[:1]
        if tamper == "duplicate":
            rows *= 2
        pq.write_table(pa.Table.from_pylist(rows), records)
        # Keep source pins consistent to exercise UID validation itself.
        evidence = release.parent / "evidence.json"
        document = json.loads(evidence.read_text())
        document["inputs"]["records"] = sha256_file(records)
        write_json_atomic(evidence, document)
        document = json.loads(release.read_text())
        document["evidence"]["manifest_sha256"] = sha256_file(evidence)
        write_json_atomic(release, document)
    elif tamper == "evidence":
        pq.write_table(pa.table({"source_row_uid": ["unrelated"]}), records)
    else:
        (release.parent / "L6.json").write_text('{}')
    with pytest.raises(ValueError):
        cache.build_cache(release, records, output)
    assert not output.exists()


@pytest.mark.parametrize("field", ("task_id", "scorer_version", "scorer_files_sha256"))
def test_oral_cache_rejects_changed_scientific_contract(tmp_path, field):
    import json
    path = cache.build_cache(*oral_inputs(tmp_path / "inputs"))
    manifest = json.loads(path.read_text())
    manifest[field] = {} if field == "scorer_files_sha256" else "different"
    write_json_atomic(path, manifest)
    with pytest.raises(ValueError):
        cache.load_scores(path)


@pytest.mark.parametrize("task,endpoint", [("bbb_martins", "BBB penetration"),
                                          ("ames", "Ames"), ("carcinogens", "carcinogenicity")])
def test_new_task_caches_roundtrip_and_bind_all_scorer_dependencies(tmp_path, task, endpoint):
    rows = [dict(source_row_uid=uid, endpoint=endpoint, biological_system="rat",
                 reported_result=value, support_text="Source") for uid, value in (("one", "positive"), ("two", "negative"))]
    inputs = oral_inputs(tmp_path / "inputs", rows, task)
    manifest = cache.build_cache(*inputs)
    scores, audit = cache.load_scores(manifest)
    assert scores == {"one": 1, "two": 1}
    saved = json.loads(manifest.read_text())
    assert saved["schema_version"] == cache.TASK_SCHEMAS[task]
    assert saved["task_id"] == task
    assert saved["scorer_commit"] == ""
    assert {"record_scoring.py", "bioavailability_score/scoring.py", "dili_score/scoring.py"} <= saved["scorer_files_sha256"].keys()
    assert audit["scorer_version"] == f"{task}_relevance_completeness.v2"
    assert pq.read_table(manifest.parent / "scores.parquet")["rule_ids"].to_pylist()
    with pytest.raises(FileExistsError):
        cache.build_cache(*inputs)
    saved["task_id"] = "dili"
    write_json_atomic(manifest, saved)
    with pytest.raises(ValueError, match="task differs"):
        cache.load_scores(manifest)
