"""Build the single Bioavailability pair-bucket assay-transfer policy.

The builder is shared (``common/starling/build_pair_bucket_transfer_policy``);
this entry point binds the Bioavailability_Ma contract so the historical
command line and call signature keep working unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.build_pair_bucket_transfer_policy import (
    POLICY_FILENAME,
    TransferPolicyBuildSpec,
)
from tools.chembl_tool.common.starling.build_pair_bucket_transfer_policy import (
    build_pair_bucket_transfer_policy as _build_pair_bucket_transfer_policy,
)
from tools.chembl_tool.common.starling.pair_bucket_transfer_policy import (
    MIN_ASSAY_TRANSFER_SAMPLES,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    DEFAULT_NORMALIZED_DIR,
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
)
from tools.chembl_tool.tasks.bioavailability_ma.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_bucket_transfer_policy import (
    TRANSFER_POLICY_PROFILE,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)


DEFAULT_PAIR_BUCKET_DIR = DEFAULT_NORMALIZED_DIR / "04_pair_buckets"
DEFAULT_OUTPUT_DIR = DEFAULT_NORMALIZED_DIR / "05_assay_transfer_policy"
DEFAULT_AUXILIARY_MANIFEST = (
    DEFAULT_NORMALIZED_DIR / "02_normalized/auxiliary_mapping_manifest.json"
)

BUILD_SPEC = TransferPolicyBuildSpec(
    profile=TRANSFER_POLICY_PROFILE,
    pair_bucket_version=BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    source_pair_fields=SOURCE_PAIR_FIELDS,
    auxiliary_mapping_version=MAPPING_VERSION,
    auxiliary_attachment_version=AUXILIARY_ATTACHMENT_VERSION,
)


def build_pair_bucket_transfer_policy(
    *,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    auxiliary_manifest_path: str | Path,
    out_dir: str | Path,
    minimum_samples: int = MIN_ASSAY_TRANSFER_SAMPLES,
) -> dict[str, Any]:
    return _build_pair_bucket_transfer_policy(
        spec=BUILD_SPEC,
        records_path=records_path,
        pair_bucket_records_path=pair_bucket_records_path,
        pair_bucket_metadata_path=pair_bucket_metadata_path,
        auxiliary_manifest_path=auxiliary_manifest_path,
        out_dir=out_dir,
        minimum_samples=minimum_samples,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=str(DEFAULT_NORMALIZED_DIR / "03_records/records.parquet"),
    )
    parser.add_argument(
        "--pair-bucket-records",
        default=str(DEFAULT_PAIR_BUCKET_DIR / PAIR_BUCKET_RECORDS_FILENAME),
    )
    parser.add_argument(
        "--pair-bucket-metadata",
        default=str(DEFAULT_PAIR_BUCKET_DIR / PAIR_BUCKET_METADATA_FILENAME),
    )
    parser.add_argument("--auxiliary-manifest", default=str(DEFAULT_AUXILIARY_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_pair_bucket_transfer_policy(
        records_path=args.records,
        pair_bucket_records_path=args.pair_bucket_records,
        pair_bucket_metadata_path=args.pair_bucket_metadata,
        auxiliary_manifest_path=args.auxiliary_manifest,
        out_dir=args.out_dir,
    )
    summary = payload["summary"]
    print(
        "[build_starling_pair_bucket_transfer_policy] "
        f"buckets={summary['pair_buckets']:,} "
        f"supported={summary['minimum_support_buckets']:,} "
        f"variance_flagged={summary['variance_gate_flagged_buckets']:,} "
        f"eligible={summary['assay_transfer_eligible_buckets']:,} "
        f"out={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
