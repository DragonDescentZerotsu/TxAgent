"""BBB-only normalized evidence-library V8 policy."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from data.processing.paths import evidence_library_root
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.measurement_resolution_rules import (
    RULE_POLICY_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_measurement_resolution import (
    DEFAULT_MAPPING_PATH as MEASUREMENT_MAPPING,
    SOURCE_MEASUREMENT_FIELDS,
    validate_mapping_provenance,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.starling_endpoint_normalization import (
    DEFAULT_MISSING_ENDPOINT_MAPPING,
    MISSING_ENDPOINT_MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins._v7_base_policy import (
    POLICY as V7_POLICY,
    _clean_source_values as clean_v7_source_values,
    manifest_versions as v7_manifest_versions,
)
from data.processing.evidence_library.versions.v9.unit_vocabulary import (
    DEFAULT_VOCABULARY_PATH,
    VOCABULARY_VERSION,
)


DEFAULT_OUT_DIR = str(evidence_library_root("bbb_martins", "v9"))
BASE_EXACT_UNIT_MAPPING = (
    Path(__file__).resolve().parent
    / "data_processing/canonicalization_v8/bbb_unit_reconciliation.v1.json"
)
MANUAL_ASSAY_TRANSFER_INELIGIBILITY_PATH = (
    Path(__file__).resolve().parent
    / "data_processing/assay_transfer_manual_ineligibility.v1.json"
)


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    V7_POLICY.add_cli_arguments(parser)
    parser.set_defaults(
        measurement_resolution_mapping=(
            str(MEASUREMENT_MAPPING) if MEASUREMENT_MAPPING.is_file() else ""
        )
    )


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    V7_POLICY.validate_arguments(parser, args)
    if args.through_stage not in {"source", "clean"}:
        try:
            validate_mapping_provenance(args.measurement_resolution_mapping)
        except ValueError as error:
            parser.error(str(error))


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    return {
        **v7_manifest_versions(complete=complete),
        "evidence_library_version": "bbb_normalized_v8",
        "measurement_routing_rule_policy_version": RULE_POLICY_VERSION,
        "observed_unit_vocabulary_version": VOCABULARY_VERSION,
        "measurement_value_transform": "identity",
        "measurement_scale_representation": "distinct_unit_identity",
        "assay_transfer_log_transform": "disabled",
        "missing_endpoint_mapping_version": MISSING_ENDPOINT_MAPPING_VERSION,
        "maximum_extracted_measurements_per_source_row": 1,
        "source_measurement_fields": {
            source_id: {
                "measurement_field": fields[0],
                "unit_field": fields[1] or None,
            }
            for source_id, fields in SOURCE_MEASUREMENT_FIELDS.items()
        },
    }


POLICY = replace(
    V7_POLICY,
    default_out_dir=DEFAULT_OUT_DIR,
    # The copied base cleaner calls the V8 router once after endpoint
    # normalization.  Do not add a second task-local routing pass here.
    source_value_cleaner=clean_v7_source_values,
    add_cli_arguments=add_cli_arguments,
    validate_arguments=validate_arguments,
    manifest_versions=manifest_versions,
    scientific_assets=tuple(
        path
        for path in V7_POLICY.scientific_assets
        if "assay_transfer_measurements_v2/policy.json" not in str(path)
    ) + (DEFAULT_VOCABULARY_PATH, DEFAULT_MISSING_ENDPOINT_MAPPING),
    assay_transfer_measurement_policy=None,
    exact_unit_mapping_path=BASE_EXACT_UNIT_MAPPING,
)


__all__ = ["BASE_EXACT_UNIT_MAPPING", "DEFAULT_OUT_DIR", "POLICY"]
