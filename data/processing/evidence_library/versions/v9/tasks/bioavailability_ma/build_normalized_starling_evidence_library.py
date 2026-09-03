"""Build the Bioavailability v7 evidence library through canonical pair buckets."""

from __future__ import annotations

from copy import copy

from data.processing.evidence_library.shared.v2.build_runtime import (
    starling_build_session,
)
from data.processing.evidence_library.versions.v9.build_normalized_evidence_library import (
    CLEANED_FILENAME,
    MANIFEST_FILENAME,
    NORMALIZATION_STAGE_VERSION,
    RECORD_DEPENDENT_DIRECTORIES,
    STAGE_OUTPUT_FILENAMES,
    STAGES,
    parse_args,
    run_with_args,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.build_starling_downstream_artifacts import (
    CORE_PAIR_BUCKET_STAGE,
    build_canonical_artifacts,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.starling_policy import (
    DEFAULT_OUT_DIR,
    POLICY,
)

ARTIFACT_STAGES = (
    "00_source",
    "01_cleaned",
    "02_canonicalized",
    CORE_PAIR_BUCKET_STAGE,
)
NORMALIZED_RECORDS_FILENAME = "02_canonicalized/records.parquet"
RECORDS_FILENAME = "03_pair_buckets/records.parquet"
AUXILIARY_MAPPING_MANIFEST_FILENAME = "02_canonicalized/auxiliary_mapping_manifest.json"
SOURCE_COLUMN_CONTRACT_FILENAME = "02_canonicalized/source_contract.json"
VALIDITY_POLICY_FILENAME = "02_canonicalized/record_validity_policy.json"
__all__ = [
    "ARTIFACT_STAGES",
    "AUXILIARY_MAPPING_MANIFEST_FILENAME",
    "CLEANED_FILENAME",
    "DEFAULT_OUT_DIR",
    "MANIFEST_FILENAME",
    "NORMALIZATION_STAGE_VERSION",
    "NORMALIZED_RECORDS_FILENAME",
    "POLICY",
    "RECORDS_FILENAME",
    "RECORD_DEPENDENT_DIRECTORIES",
    "SOURCE_COLUMN_CONTRACT_FILENAME",
    "STAGES",
    "STAGE_OUTPUT_FILENAMES",
    "VALIDITY_POLICY_FILENAME",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(POLICY, argv, "pair-buckets", True)
    with starling_build_session(args.out_dir):
        if args.through_stage != "pair-buckets":
            return run_with_args(POLICY, args)
        if args.from_stage != "pair-buckets":
            record_args = copy(args)
            record_args.through_stage = "normalize"
            if result := run_with_args(POLICY, record_args):
                return result
        build_canonical_artifacts(
            normalized_root=args.out_dir,
            workers=args.workers,
            rebuild_request={
                "from_stage": args.from_stage,
                "through_stage": CORE_PAIR_BUCKET_STAGE,
            },
            validation_level=args.validation_level,
            cache_mode=args.cache_mode,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
