from __future__ import annotations

import hashlib
import json

import pandas as pd

from tools.chembl_tool.tasks.bbb_martins.data_processing.build_direct_endpoint_errata import (
    build_errata_proposal,
)


def test_errata_proposal_is_complete_deterministic_and_human_blocked(tmp_path):
    base = tmp_path / "v1.json"
    base.write_text(
        json.dumps(
            {
                "mapping_version": "bbb_martins_direct_endpoint.globally_reconciled.v1",
                "approval": {
                    "human_approved": True,
                    "approved_by": "reviewer",
                    "approved_at": "2026-08-03T00:00:00Z",
                },
                "source": {
                    "sha256": "source",
                    "inventory_count_including_missing": 2,
                    "inventory_sha256": "inventory",
                },
                "mapping": {
                    "BBB": "brain_to_blood_ratio",
                    "Log BB": "logbb",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    original_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    records = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "source_id": "direct_bbb",
                "endpoint_name": "BBB",
                "canonical_endpoint_name": "brain_to_blood_ratio",
                "canonical_unit_text": "ratio",
                "finite_scalar_value": -1.63,
                "measurement_unit_status": "cleaned_pair",
                "canonicalization_status": "unreviewed_endpoint_semantics",
            },
            {
                "source_id": "direct_bbb",
                "endpoint_name": "Log BB",
                "canonical_endpoint_name": "logbb",
                "canonical_unit_text": "dimensionless",
                "finite_scalar_value": -0.4,
                "measurement_unit_status": "cleaned_pair",
                "canonicalization_status": "valid",
            },
            {
                "source_id": "direct_bbb",
                "endpoint_name": None,
                "canonical_endpoint_name": "missing_endpoint",
                "canonical_unit_text": None,
                "finite_scalar_value": None,
                "measurement_unit_status": "missing_measurement",
                "canonicalization_status": "missing_canonical_endpoint",
            },
        ]
    ).to_parquet(records, index=False)
    root = tmp_path / "v2"
    manifest = build_errata_proposal(
        base_mapping_path=base,
        normalized_records_path=records,
        output_root=root,
    )
    proposal = json.loads(
        (root / "proposal/proposed_endpoint_mapping.json").read_text(encoding="utf-8")
    )
    assert proposal["approval"]["human_approved"] is False
    assert proposal["mapping"]["BBB"] == "bbb_ambiguous_measurement"
    assert set(proposal["mapping"]) == {"BBB", "Log BB"}
    assert manifest["counts"]["proposed_mapping_changes"] == 1
    assert hashlib.sha256(base.read_bytes()).hexdigest() == original_sha
