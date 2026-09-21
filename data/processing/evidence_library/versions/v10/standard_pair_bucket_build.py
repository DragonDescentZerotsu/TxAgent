"""Shared Stage-3 binding for canonical-v7 task contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
)
from data.processing.evidence_library.shared.v2.build_pair_bucket_transfer_policy import (
    TransferPolicyBuildSpec,
)
from data.processing.evidence_library.shared.v2.normalization.audit import write_parquet
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.pair_bucket_transfer_policy import (
    TransferPolicyProfile,
)
from data.processing.evidence_library.shared.v2.pair_buckets import (
    materialize_pair_buckets,
    read_pair_bucket_input,
)
from data.processing.evidence_library.versions.v10.build_pair_bucket_distance_calibration import (
    build_pair_bucket_distance_calibration,
)
from data.processing.evidence_library.versions.v10.pair_bucket_build import (
    PAIR_BUCKET_METADATA_FILENAME,
    PAIR_BUCKET_RECORDS_FILENAME,
    PairBucketBuildSpec,
)


def make_standard_pair_bucket_spec(
    *,
    task_id: str,
    policy: Any,
    pair_bucket_version: str,
    auxiliary_mapping_version: str,
    auxiliary_output_fields_by_source: Mapping[str, tuple[str, ...]],
    endpoint_field_by_source: Mapping[str, str] | None = None,
) -> PairBucketBuildSpec:
    """Bind one canonical-v7 record contract to the shared Stage-3 builders."""
    contract = policy.record_contract
    if contract is None:
        raise ValueError("standard Stage 3 requires a canonical-v7 record contract")
    sources = set(contract.sources)
    if set(auxiliary_output_fields_by_source) != sources:
        raise ValueError("auxiliary output/source inventory differs from record contract")
    pair_fields = {
        source: spec.additional_dimensions
        for source, spec in contract.pair_buckets.items()
    }
    endpoint_fields = dict(
        endpoint_field_by_source
        or {source: "canonical_endpoint_name" for source in sources}
    )
    if set(endpoint_fields) != sources:
        raise ValueError("endpoint field/source inventory differs from record contract")
    required_outputs = tuple(
        dict.fromkeys(
            field
            for source in contract.sources
            for field in auxiliary_output_fields_by_source[source]
        )
    )
    transfer_spec = TransferPolicyBuildSpec(
        profile=TransferPolicyProfile(
            version=f"{task_id}_pair_bucket_transfer_policy.v1",
            source_candidate_fields={
                source: spec.variance_candidates
                for source, spec in contract.pair_buckets.items()
            },
        ),
        pair_bucket_version=pair_bucket_version,
        source_pair_fields=pair_fields,
        auxiliary_mapping_version=auxiliary_mapping_version,
        auxiliary_attachment_version=AUXILIARY_ATTACHMENT_VERSION,
        required_auxiliary_output_fields=required_outputs,
        endpoint_field_by_source=endpoint_fields,
        include_soft_transfer_contract=True,
    )

    def build_sidecar(
        *,
        records_path: str | Path,
        out_dir: str | Path,
        assay_transfer_record_ineligibility: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        records_path = Path(records_path)
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        records, v7, source_fields = read_pair_bucket_input(
            records_path,
            v7_source_fields=pair_fields,
            legacy_source_fields=pair_fields,
            v7_endpoint_field_by_source=endpoint_fields,
        )
        if not v7:
            raise ValueError("standard Stage 3 requires canonical-v7 records")
        source_uids = [str(record.get("source_row_uid") or "") for record in records]
        if not all(source_uids) or len(source_uids) != len(set(source_uids)):
            raise ValueError(
                "standard Stage 3 requires unique nonempty source_row_uid values"
            )
        rows, metadata = materialize_pair_buckets(
            records,
            source_required_fields=source_fields,
            contract_version=pair_bucket_version,
            endpoint_field_by_source=endpoint_fields,
            assay_transfer_record_ineligibility=assay_transfer_record_ineligibility,
            canonical_record_contract=True,
        )
        if not all(metadata["validations"].values()):
            raise ValueError(f"pair-bucket validation failed: {metadata['validations']}")
        output = target / PAIR_BUCKET_RECORDS_FILENAME
        write_parquet(output, rows)
        metadata.update(
            {
                "input": {
                    "path": str(records_path),
                    "sha256": file_sha256(records_path),
                    "records": len(records),
                },
                "output": {"path": str(output), "sha256": file_sha256(output)},
                "scope": {
                    "pair_enumeration": False,
                    "pair_labels": False,
                    "comparison_thresholds": False,
                    "modeling_dataset": False,
                },
            }
        )
        (target / PAIR_BUCKET_METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return metadata

    def build_transfer_policy(
        *,
        records_path: str | Path,
        pair_bucket_records_path: str | Path,
        pair_bucket_metadata_path: str | Path,
        auxiliary_manifest_path: str | Path,
        out_dir: str | Path,
        workers: int = 1,
    ) -> dict[str, Any]:
        return build_pair_bucket_distance_calibration(
            spec=transfer_spec,
            record_contract=contract,
            records_path=records_path,
            pair_bucket_records_path=pair_bucket_records_path,
            pair_bucket_metadata_path=pair_bucket_metadata_path,
            auxiliary_manifest_path=auxiliary_manifest_path,
            out_dir=out_dir,
            workers=workers,
        )

    return PairBucketBuildSpec(
        task_id=task_id,
        policy=policy,
        pair_bucket_version=pair_bucket_version,
        build_sidecar=build_sidecar,
        build_transfer_policy=build_transfer_policy,
        protected_source_row_uids_path=policy.protected_voter_membership,
        reviewed_level_mapping_path=policy.source_universe_mapping,
    )


__all__ = ["make_standard_pair_bucket_spec"]
