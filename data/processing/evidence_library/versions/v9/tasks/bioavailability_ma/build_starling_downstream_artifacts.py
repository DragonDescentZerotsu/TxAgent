"""bioavailability_ma binding for canonical Stage-3 pair-bucket construction."""

from __future__ import annotations

from typing import Any

from data.processing.evidence_library.versions.v9 import pair_bucket_build
from data.processing.evidence_library.versions.v9.pair_bucket_build import (
    CORE_PAIR_BUCKET_STAGE,
    PairBucketBuildSpec,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    build_sidecar,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.build_starling_pair_bucket_transfer_policy import (
    build_pair_bucket_transfer_policy,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.direct_record_mapping import (
    build_direct_record_mapping,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.starling_policy import (
    POLICY,
)


def get_spec() -> PairBucketBuildSpec:
    return PairBucketBuildSpec(
        task_id="bioavailability_ma",
        policy=POLICY,
        pair_bucket_version=BIOAVAILABILITY_V7_PAIR_BUCKET_VERSION,
        build_sidecar=build_sidecar,
        build_transfer_policy=build_pair_bucket_transfer_policy,
        direct_mapping_builder=build_direct_record_mapping,
    )


def build_canonical_artifacts(**kwargs: Any) -> dict[str, Any]:
    return pair_bucket_build.build_canonical_artifacts(get_spec(), **kwargs)


__all__ = ["CORE_PAIR_BUCKET_STAGE", "build_canonical_artifacts", "get_spec"]
