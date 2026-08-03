import json

import pandas as pd

from tools.chembl_tool.tasks.bioavailability_ma.audit_starling_v5_v6_migration import (
    audit_migration,
)


def test_migration_audit_tracks_schema_and_attachment_coverage(tmp_path):
    v5_path = tmp_path / "v5.parquet"
    v6_path = tmp_path / "v6.parquet"
    manifest_path = tmp_path / "auxiliary.json"
    pd.DataFrame(
        [
            {
                "normalized_record_id": "a",
                "source_id": "fa",
                "canonical_assay_system": "caco_2",
            }
        ]
    ).to_parquet(v5_path, index=False)
    pd.DataFrame(
        [
            {
                "normalized_record_id": "a",
                "source_id": "fa",
                "global_context": "caco_2",
                "global_species_context": "human",
                "auxiliary_mapping_status": "mapped",
                "auxiliary_attachment_version": "starling_auxiliary_attachment.v1",
            },
            {
                "normalized_record_id": "b",
                "source_id": "oral_exposure",
                "global_context": None,
                "global_species_context": None,
                "auxiliary_mapping_status": "not_applicable",
                "auxiliary_attachment_version": "starling_auxiliary_attachment.v1",
            },
        ]
    ).to_parquet(v6_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "mapping_sha256": "abc",
                "mapping_version": "starling_auxiliary.globally_reconciled.v1",
                "coverage": {"validations": {"all_applicable_records_mapped": True}},
            }
        ),
        encoding="utf-8",
    )

    result = audit_migration(
        v5_records=v5_path,
        v6_records=v6_path,
        auxiliary_manifest=manifest_path,
    )
    assert result["records"]["common_normalized_record_ids"] == 1
    assert result["records"]["v6_only_normalized_record_ids"] == 1
    assert result["schema"]["heuristic_fields_present_in_v6"] == []
    assert result["schema"]["auxiliary_status_by_source"]["fa"] == {"mapped": 1}
    assert all(result["validations"].values())
