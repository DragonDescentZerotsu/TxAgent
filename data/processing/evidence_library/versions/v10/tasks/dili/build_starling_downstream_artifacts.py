"""DILI binding for canonical Stage-3 construction."""

from __future__ import annotations

from typing import Any

from data.processing.evidence_library.versions.v10 import pair_bucket_build
from data.processing.evidence_library.versions.v10.standard_pair_bucket_build import (
    make_standard_pair_bucket_spec,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_policy import (
    POLICY,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_schema import (
    PAIR_DIMENSION_INPUTS,
    PAIR_MAPPING_VERSION,
)


def get_spec():
    return make_standard_pair_bucket_spec(
        task_id="dili",
        policy=POLICY,
        pair_bucket_version="dili_pair_buckets.v10",
        auxiliary_mapping_version=PAIR_MAPPING_VERSION,
        auxiliary_output_fields_by_source={
            source: ("canonical_endpoint_concept", *fields)
            for source, fields in PAIR_DIMENSION_INPUTS.items()
        },
        endpoint_field_by_source={
            source: "canonical_endpoint_concept" for source in PAIR_DIMENSION_INPUTS
        },
    )


def build_canonical_artifacts(**kwargs: Any) -> dict[str, Any]:
    return pair_bucket_build.build_canonical_artifacts(get_spec(), **kwargs)


__all__ = ["build_canonical_artifacts", "get_spec"]
