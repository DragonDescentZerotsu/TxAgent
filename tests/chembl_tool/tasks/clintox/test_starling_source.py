import json
import csv
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.tasks.clintox.starling_source import (
    ARCHIVE_SHA256,
    DEFAULT_DATA_ROOT,
    DIRECT_SOURCE_ID,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    SOURCE_COLUMNS,
    SOURCE_RELEASE,
    SOURCE_SPECS,
)


def test_send_v2_contract_is_complete_and_stable():
    assert SOURCE_RELEASE == "clintox_send_v2"
    assert len(SOURCE_SPECS) == 7
    assert sum(EXPECTED_SOURCE_ROWS.values()) == 4_846_914
    assert DIRECT_SOURCE_ID == "human_clinical_toxicity"
    assert set(EXPECTED_SOURCE_ROWS) == set(EXPECTED_SOURCE_SHA256) == set(SOURCE_COLUMNS)
    assert all("global_identifier" not in columns for columns in SOURCE_COLUMNS.values())
    assert len(ARCHIVE_SHA256) == 64


def test_tracked_source_manifest_matches_release_contract():
    manifest_path = DEFAULT_DATA_ROOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_release"] == SOURCE_RELEASE
    assert manifest["archive"]["sha256"] == ARCHIVE_SHA256
    by_source = {item["source_id"]: item for item in manifest["sources"]}
    files = {
        item["source_id"]: item
        for item in manifest["files"]
        if item["archive_member"].endswith("/extractions.parquet")
    }
    assert set(by_source) == set(EXPECTED_SOURCE_ROWS)
    for source_id, expected_rows in EXPECTED_SOURCE_ROWS.items():
        source = by_source[source_id]
        assert expected_rows == source["rows"]
        assert tuple(source["columns"]) == SOURCE_COLUMNS[source_id]
        assert source["smiles_nonempty_rows"] == expected_rows
        assert files[source_id]["sha256"] == EXPECTED_SOURCE_SHA256[source_id]


@pytest.mark.skipif(
    not all(
        (DEFAULT_DATA_ROOT / source_id / "extractions.parquet").is_file()
        for source_id in EXPECTED_SOURCE_ROWS
    ),
    reason="full send_v2 source is restored only in data-build workspaces",
)
def test_local_source_parquets_match_release_contract():
    for source_id, expected_rows in EXPECTED_SOURCE_ROWS.items():
        path = DEFAULT_DATA_ROOT / source_id / "extractions.parquet"
        parquet = pq.ParquetFile(path)
        assert parquet.metadata.num_rows == expected_rows
        assert tuple(parquet.schema_arrow.names) == SOURCE_COLUMNS[source_id]
        assert file_sha256(path) == EXPECTED_SOURCE_SHA256[source_id]


def test_send_v2_release_delta_manual_audit_is_complete():
    root = (
        Path(__file__).resolve().parents[4]
        / "tools/chembl_tool/tasks/clintox/data_processing/source_delta_v2"
    )
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    audit_path = root / "manual_audit.tsv"
    with audit_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 140
    assert {row["manual_review_status"] for row in rows} == {"pass"}
    assert summary["status"] == "complete"
    assert summary["manual_review_passes"] == 140
    assert summary["manual_review_failures"] == 0
    assert summary["manual_review_tsv_sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
