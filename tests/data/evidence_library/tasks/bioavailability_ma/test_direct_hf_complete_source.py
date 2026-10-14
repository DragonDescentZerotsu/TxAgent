import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v7.tasks.bioavailability_ma.starling_normalization_sources import (
    DEFAULT_HF_BIOAVAILABILITY_PARQUET,
    EXPECTED_RAW_HF_BIOAVAILABILITY_ROWS,
    HF_BIOAVAILABILITY_SOURCE_COLUMNS,
)


def test_complete_hf_bioavailability_source_is_small_complete_and_unpartitioned():
    path = Path(DEFAULT_HF_BIOAVAILABILITY_PARQUET)
    manifest = json.loads((path.parent / "source_manifest.json").read_text())
    metadata = pq.read_metadata(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert path.stat().st_size < 100_000_000
    assert metadata.num_rows == EXPECTED_RAW_HF_BIOAVAILABILITY_ROWS
    assert tuple(metadata.schema.names) == HF_BIOAVAILABILITY_SOURCE_COLUMNS
    assert digest == manifest["records_parquet"]["sha256"]
    assert manifest["semantics"]["historical_kept_dropped_partition"] is False
    assert manifest["semantics"]["historical_preparation_reason_included"] is False
    assert not any("drop" in column or "preparation" in column for column in metadata.schema.names)
