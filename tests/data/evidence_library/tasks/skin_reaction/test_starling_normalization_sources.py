"""The four Skin_Reaction source profiles must match the pinned manifest."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_SHA256,
    source_profiles,
    validate_source_digest,
)


DATA_DIR = Path("data/raw/starling/skin_reaction")
MANIFEST = DATA_DIR / "SOURCE_MANIFEST.json"


def _manifest_entries():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {entry["mechanism_family"]: entry for entry in payload["mapping"]}


def test_expected_rows_and_digests_match_the_source_manifest():
    entries = _manifest_entries()
    assert set(entries) == set(EXPECTED_SOURCE_ROWS) == set(EXPECTED_SOURCE_SHA256)
    for source_id, entry in entries.items():
        assert EXPECTED_SOURCE_ROWS[source_id] == entry["parquet_rows"]
        assert EXPECTED_SOURCE_SHA256[source_id] == entry["parquet_sha256"]


def test_every_profile_declares_columns_that_exist_in_its_parquet():
    for profile in source_profiles(DATA_DIR):
        path = Path(profile.source_path)
        if not path.exists():
            pytest.skip(f"source parquet not staged: {path}")
        columns = set(pq.ParquetFile(path).schema_arrow.names)
        declared = {
            profile.endpoint_field,
            profile.measurement_field,
            profile.smiles_field,
            profile.record_id_field,
            profile.support_text_field,
            *profile.context_fields,
            *profile.name_fields,
        }
        if profile.unit_field:
            declared.add(profile.unit_field)
        missing = sorted(field for field in declared if field and field not in columns)
        assert not missing, f"{profile.source_id} declares absent columns: {missing}"


def test_structure_is_read_from_the_uppercase_smiles_column():
    # Every skin parquet carries its own structures, unlike Bioavailability_Ma
    # which maps global_identifier through a shared SMILES mapping file.
    for profile in source_profiles(DATA_DIR):
        assert profile.structure_mode == "direct"
        assert profile.smiles_field == "SMILES"


def test_row_counts_match_the_staged_parquets():
    for profile in source_profiles(DATA_DIR):
        path = Path(profile.source_path)
        if not path.exists():
            pytest.skip(f"source parquet not staged: {path}")
        rows = pq.ParquetFile(path).metadata.num_rows
        assert rows == EXPECTED_SOURCE_ROWS[profile.source_id]


def test_digest_validation_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="no pinned digest"):
        validate_source_digest("not_a_source", MANIFEST)
