"""Paths for the staged repair-aware Oral canonical-source v3 release."""

from data.processing.paths import ARTIFACTS_ROOT


CANONICAL_VERSION = "bioavailability_canonical_direct.v3"
CANONICAL_SOURCE_DIR = (
    ARTIFACTS_ROOT
    / "starling/bioavailability_ma/canonical_sources/canonical_direct_v3"
)
HF_SNAPSHOT_PATH = CANONICAL_SOURCE_DIR / "hf_oral_bioavailability_snapshot.parquet"
HF_NONDIRECT_RECORDS_PATH = CANONICAL_SOURCE_DIR / "hf_nondirect_records.parquet"
DIRECT_SOURCE_ROWS_PATH = CANONICAL_SOURCE_DIR / "direct_source_rows.parquet"
DIRECT_CLAIMS_PATH = CANONICAL_SOURCE_DIR / "direct_claims.parquet"
DIRECT_REJECTED_ROWS_PATH = CANONICAL_SOURCE_DIR / "direct_rejected_rows.parquet"
DEDUP_AUDIT_PATH = CANONICAL_SOURCE_DIR / "cross_source_dedup_audit.parquet"
LOCAL_PARTITION_AUDIT_PATH = CANONICAL_SOURCE_DIR / "local_partition_audit.parquet"
MANIFEST_PATH = CANONICAL_SOURCE_DIR / "merge_manifest.json"

PRIOR_DIRECT_CLAIMS_PATH = (
    ARTIFACTS_ROOT
    / "starling/bioavailability_ma/canonical_sources/canonical_direct_v2/direct_claims.parquet"
)

RESIDUAL_SOURCE_DIR = (
    ARTIFACTS_ROOT
    / "starling/bioavailability_ma/canonical_sources/oral_exposure_residual_v3"
)
RESIDUAL_RECORDS_PATH = RESIDUAL_SOURCE_DIR / "exposure_records.parquet"
RESIDUAL_MANIFEST_PATH = RESIDUAL_SOURCE_DIR / "partition_manifest.json"


__all__ = [
    "CANONICAL_SOURCE_DIR",
    "CANONICAL_VERSION",
    "DEDUP_AUDIT_PATH",
    "DIRECT_CLAIMS_PATH",
    "DIRECT_REJECTED_ROWS_PATH",
    "DIRECT_SOURCE_ROWS_PATH",
    "HF_NONDIRECT_RECORDS_PATH",
    "HF_SNAPSHOT_PATH",
    "LOCAL_PARTITION_AUDIT_PATH",
    "MANIFEST_PATH",
    "PRIOR_DIRECT_CLAIMS_PATH",
    "RESIDUAL_MANIFEST_PATH",
    "RESIDUAL_RECORDS_PATH",
    "RESIDUAL_SOURCE_DIR",
]
