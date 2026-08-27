from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.tasks.clintox.clinical_trial_failure_benchmark import (
    build_canonical_frames,
    build_starling_coverage,
)


def test_source_role_overlap_is_positive_and_invalid_structure_is_audited():
    aacttox = pd.DataFrame([{"smiles": "CCO", "CT_TOX": 1}])
    comparator = pd.DataFrame(
        [
            {"smiles": "CCO", "FDA_APPROVED": 1},
            {"smiles": "CCC", "FDA_APPROVED": 1},
            {"smiles": "not-a-smiles", "FDA_APPROVED": 1},
        ]
    )
    reference = pd.DataFrame(
        [
            {"smiles": "CCO", "FDA_APPROVED": 1, "CT_TOX": 1},
            {"smiles": "CCC", "FDA_APPROVED": 1, "CT_TOX": 0},
        ]
    )

    result = build_canonical_frames(aacttox, comparator, reference)
    parents = result["parent_labels"].set_index("drug")

    assert len(parents) == 2
    assert int(parents.loc["CCO", "Y"]) == 1
    assert bool(parents.loc["CCO", "source_role_overlap"]) is True
    assert parents.loc["CCO", "label_decision"] == (
        "positive_aacttox_event_overrides_comparator"
    )
    assert int(parents.loc["CCC", "Y"]) == 0
    assert result["stats"]["source_identity_status_counts"] == {
        "accepted": 3,
        "invalid_or_unresolved_smiles": 1,
    }
    assert result["stats"]["n_common_reference_label_mismatches"] == 0


def test_joined_reference_conflict_uses_existential_positive_for_audit_only():
    aacttox = pd.DataFrame([{"smiles": "CCO", "CT_TOX": 1}])
    comparator = pd.DataFrame([{"smiles": "CCO", "FDA_APPROVED": 1}])
    reference = pd.DataFrame(
        [
            {"smiles": "CCO", "FDA_APPROVED": 1, "CT_TOX": 0},
            {"smiles": "CCO", "FDA_APPROVED": 0, "CT_TOX": 1},
        ]
    )

    result = build_canonical_frames(aacttox, comparator, reference)

    assert result["stats"]["n_reference_parent_raw_label_conflicts"] == 1
    row = result["reference_reconciliation"].iloc[0]
    assert bool(row["reference_raw_label_conflict"]) is True
    assert row["reconciliation_status"] == "label_match"


def test_starling_coverage_never_changes_labels():
    aacttox = pd.DataFrame([{"smiles": "CCO", "CT_TOX": 1}])
    comparator = pd.DataFrame([{"smiles": "CCC", "FDA_APPROVED": 1}])
    reference = pd.DataFrame(
        [
            {"smiles": "CCO", "FDA_APPROVED": 0, "CT_TOX": 1},
            {"smiles": "CCC", "FDA_APPROVED": 1, "CT_TOX": 0},
        ]
    )
    canonical = build_canonical_frames(aacttox, comparator, reference)

    coverage = build_starling_coverage(
        canonical["parent_labels"], ["CCO", "CCO", "CCN"]
    )

    assert coverage["stats"]["starling_rows_used_to_create_labels"] == 0
    covered = coverage["coverage"].set_index("drug")
    assert int(covered.loc["CCO", "Y"]) == 1
    assert int(covered.loc["CCC", "Y"]) == 0
    assert int(
        covered.loc["CCO", "human_clinical_toxicity_source_row_count"]
    ) == 2
    assert bool(
        covered.loc["CCC", "has_human_clinical_toxicity_evidence"]
    ) is False


def test_stage1_vote_and_residual_artifacts_reconcile():
    canonical = Path(
        "data/starling_data/clintox/canonical_clinical_trial_failure_v1"
    )
    split = Path(
        "data/processed_clintox_clinical_trial_failure_v1/ClinTox/scaffold"
    )
    inventory = json.loads((canonical / "source_inventory.json").read_text())
    voters = [
        json.loads(line)
        for line in (split / "voting_records.jsonl").read_text().splitlines()
    ]
    residuals = pq.read_table(canonical / "direct_residual_candidates.parquet")

    assert inventory["counts"] == {
        "direct_residual_candidate_rows": 338,
        "direct_vote_rows": 1505,
        "excluded_direct_source_rows": 4,
        "indirect_rows": 4_846_914,
        "source_datasets": 9,
        "source_rows": 4_848_423,
    }
    assert len(voters) == 1505
    assert sum(not row["vote_matches_final_parent_label"] for row in voters) == 65
    assert residuals.num_rows == 338
    assert set(residuals["source_role"].to_pylist()) == {"indirect"}
    assert set(residuals["retrieval_eligible"].to_pylist()) == {False}
