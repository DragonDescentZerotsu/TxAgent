"""Build Bioavailability v7 calibration or the frozen v6 transfer policy.

The historical command and function name remain compatibility entry points.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.build_pair_bucket_transfer_policy import (
    POLICY_FILENAME,
    TransferPolicyBuildSpec,
)
from tools.chembl_tool.common.starling.build_pair_bucket_distance_calibration import (
    build_pair_bucket_distance_calibration as _build_distance_calibration,
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
    BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION,
    ENDPOINT_FIELD_BY_SOURCE,
    SOURCE_PAIR_FIELDS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import RECORD_CONTRACT


DEFAULT_PAIR_BUCKET_DIR = DEFAULT_NORMALIZED_DIR / "03_pair_buckets"
DEFAULT_LEGACY_OUTPUT_DIR = DEFAULT_NORMALIZED_DIR / "05_assay_transfer_policy"
DEFAULT_V7_OUTPUT_DIR = DEFAULT_NORMALIZED_DIR / "03_pair_buckets"
DEFAULT_OUTPUT_DIR = DEFAULT_LEGACY_OUTPUT_DIR
DEFAULT_AUXILIARY_MANIFEST = (
    DEFAULT_NORMALIZED_DIR / "02_canonicalized/auxiliary_mapping_manifest.json"
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
    minimum_samples: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    metadata = json.loads(Path(pair_bucket_metadata_path).read_text(encoding="utf-8"))
    spec = BUILD_SPEC
    if metadata.get("contract_version") == BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION:
        profile = replace(
            TRANSFER_POLICY_PROFILE,
            source_candidate_fields={
                source: item.variance_candidates
                for source, item in RECORD_CONTRACT.pair_buckets.items()
            },
        )
        spec = replace(
            BUILD_SPEC,
            profile=profile,
            pair_bucket_version=BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION,
            source_pair_fields={
                source: item.additional_dimensions
                for source, item in RECORD_CONTRACT.pair_buckets.items()
            },
            endpoint_field_by_source=ENDPOINT_FIELD_BY_SOURCE,
        )
        return _build_distance_calibration(
            spec=spec,
            record_contract=RECORD_CONTRACT,
            records_path=records_path,
            pair_bucket_records_path=pair_bucket_records_path,
            pair_bucket_metadata_path=pair_bucket_metadata_path,
            auxiliary_manifest_path=auxiliary_manifest_path,
            out_dir=out_dir,
            minimum_samples=minimum_samples,
            workers=workers,
        )
    return _build_pair_bucket_transfer_policy(
        spec=spec,
        records_path=records_path,
        pair_bucket_records_path=pair_bucket_records_path,
        pair_bucket_metadata_path=pair_bucket_metadata_path,
        auxiliary_manifest_path=auxiliary_manifest_path,
        out_dir=out_dir,
        minimum_samples=(
            MIN_ASSAY_TRANSFER_SAMPLES
            if minimum_samples is None
            else minimum_samples
        ),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=str(DEFAULT_NORMALIZED_DIR / "03_pair_buckets/records.parquet"),
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
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. By default, v7 metadata publishes beside the input as "
            "07_distance_calibration and frozen v6 metadata uses 05_assay_transfer_policy."
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir = str(
        resolve_output_dir(args.pair_bucket_metadata, explicit=args.out_dir)
    )
    payload = build_pair_bucket_transfer_policy(
        records_path=args.records,
        pair_bucket_records_path=args.pair_bucket_records,
        pair_bucket_metadata_path=args.pair_bucket_metadata,
        auxiliary_manifest_path=args.auxiliary_manifest,
        out_dir=args.out_dir,
        workers=args.workers,
    )
    summary = payload["summary"]
    if "calibration_valid_buckets" in summary:
        print(
            "[build_starling_pair_bucket_distance_calibration] "
            f"buckets={summary['pair_buckets']:,} "
            f"supported={summary['minimum_support_buckets']:,} "
            f"calibrated={summary['calibration_valid_buckets']:,} "
            f"out={args.out_dir}",
            flush=True,
        )
        return 0
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


def resolve_output_dir(
    pair_bucket_metadata_path: str | Path,
    *,
    explicit: str | Path | None,
) -> Path:
    if explicit:
        return Path(explicit)
    metadata_path = Path(pair_bucket_metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract_version") == BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION:
        return metadata_path.parent
    return metadata_path.parent.parent / "05_assay_transfer_policy"


if __name__ == "__main__":
    raise SystemExit(main())
