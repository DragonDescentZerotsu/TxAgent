"""Build BBB v7 distance calibration or the frozen v6 transfer policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.build_pair_bucket_transfer_policy import (
    POLICY_FILENAME,
    TransferPolicyBuildSpec,
    build_pair_bucket_transfer_policy as _build,
)
from data.processing.evidence_library.versions.v9.build_pair_bucket_distance_calibration import (
    build_pair_bucket_distance_calibration as _build_distance_calibration,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.build_starling_pair_bucket_sidecar import (
    DEFAULT_NORMALIZED_DIR,
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_pair_bucket_transfer_policy import (
    TRANSFER_POLICY_PROFILE,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_pair_buckets import (
    BBB_MARTINS_PAIR_BUCKET_VERSION,
    BBB_MARTINS_V7_PAIR_BUCKET_VERSION,
    ENDPOINT_FIELD_BY_SOURCE,
    SOURCE_PAIR_FIELDS,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_schema import RECORD_CONTRACT


BUILD_SPEC = TransferPolicyBuildSpec(
    profile=TRANSFER_POLICY_PROFILE,
    pair_bucket_version=BBB_MARTINS_PAIR_BUCKET_VERSION,
    source_pair_fields=SOURCE_PAIR_FIELDS,
    auxiliary_mapping_version=MAPPING_VERSION,
    auxiliary_attachment_version=AUXILIARY_ATTACHMENT_VERSION,
    include_soft_transfer_contract=True,
)


def build_pair_bucket_transfer_policy(
    *,
    records_path: str | Path,
    pair_bucket_records_path: str | Path,
    pair_bucket_metadata_path: str | Path,
    auxiliary_manifest_path: str | Path,
    out_dir: str | Path,
    workers: int = 1,
) -> dict[str, Any]:
    metadata = json.loads(Path(pair_bucket_metadata_path).read_text(encoding="utf-8"))
    spec = BUILD_SPEC
    if metadata.get("contract_version") == BBB_MARTINS_V7_PAIR_BUCKET_VERSION:
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
            pair_bucket_version=BBB_MARTINS_V7_PAIR_BUCKET_VERSION,
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
            workers=workers,
        )
    return _build(
        spec=spec,
        records_path=records_path,
        pair_bucket_records_path=pair_bucket_records_path,
        pair_bucket_metadata_path=pair_bucket_metadata_path,
        auxiliary_manifest_path=auxiliary_manifest_path,
        out_dir=out_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default=str(DEFAULT_NORMALIZED_DIR / "03_pair_buckets/records.parquet"),
    )
    parser.add_argument(
        "--pair-bucket-records",
        default=str(DEFAULT_NORMALIZED_DIR / "03_pair_buckets" / PAIR_BUCKET_RECORDS_FILENAME),
    )
    parser.add_argument(
        "--pair-bucket-metadata",
        default=str(DEFAULT_NORMALIZED_DIR / "03_pair_buckets" / PAIR_BUCKET_METADATA_FILENAME),
    )
    parser.add_argument(
        "--auxiliary-manifest",
        default=str(DEFAULT_NORMALIZED_DIR / "02_canonicalized/auxiliary_mapping_manifest.json"),
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_NORMALIZED_DIR / "03_pair_buckets")
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    payload = build_pair_bucket_transfer_policy(
        records_path=args.records,
        pair_bucket_records_path=args.pair_bucket_records,
        pair_bucket_metadata_path=args.pair_bucket_metadata,
        auxiliary_manifest_path=args.auxiliary_manifest,
        out_dir=args.out_dir,
        workers=args.workers,
    )
    print(payload["summary"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "BUILD_SPEC",
    "HELDOUT_LABEL_PATHS",
    "POLICY_FILENAME",
    "build_pair_bucket_transfer_policy",
]
