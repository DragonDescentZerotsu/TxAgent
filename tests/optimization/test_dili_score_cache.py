from pathlib import Path
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
