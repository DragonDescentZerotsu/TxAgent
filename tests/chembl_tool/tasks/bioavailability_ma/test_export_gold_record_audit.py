from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data.processing.gold_labels.benchmark_dataset import (
    LabeledSourceRecord,
    accepted,
)
from tools.chembl_tool.tasks.bioavailability_ma.export_gold_record_audit import (
    _source_audit_by_id,
    annotate_claim_decisions,
    annotate_direct_source_rows,
)


def test_claim_decisions_are_selected_and_sorted_by_parent_identity():
    claims = pd.DataFrame(
        [
            {
                "parent_identity_key": "parent-key",
                "canonical_claim_id": "claim-1",
                "source_record_id": "hf:0",
                "source_record_ids": ["hf:0"],
                "n_source_records": 1,
                "cross_source_deduplicated": False,
            }
        ]
    )
    parents = {
        "parent-key": {
            "gold_label": 1,
            "split": "test",
            "gold_label_decision": "unanimous",
        }
    }
    decision = accepted(
        LabeledSourceRecord(
            smiles="CCO",
            label=1,
            source_id="source",
            source_record_id="claim-1",
        )
    )

    with patch(
        "tools.chembl_tool.tasks.bioavailability_ma.export_gold_record_audit.load_label_decisions",
        return_value=(iter([decision]), {}),
    ):
        annotated = annotate_claim_decisions(
            claims,
            parents,
            manifest_path=Path("unused-manifest.json"),
            source_path=Path("unused-claims.parquet"),
        )

    assert annotated.loc[0, "audit_claim_decision"] == "used_as_gold_vote"
    assert annotated.loc[0, "audit_vote_direction"] == "supports_final_gold_label"
    assert annotated.loc[0, "audit_gold_split"] == "test"


def test_source_rows_share_one_canonical_vote_after_cross_source_dedup():
    claims = pd.DataFrame(
        [
            {
                "canonical_claim_id": "claim-1",
                "source_record_id": "hf:0",
                "source_record_ids": ["hf:0", "local:0:x"],
                "n_source_records": 2,
                "cross_source_deduplicated": True,
                "audit_canonical_claim_id": "claim-1",
                "audit_claim_decision": "used_as_gold_vote",
                "audit_filter_reason": "",
                "audit_candidate_label": 1,
                "audit_vote_direction": "supports_final_gold_label",
                "audit_used_in_gold_vote": True,
            }
        ]
    )
    source_audit = _source_audit_by_id(claims)
    parents = {
        "parent-key": {
            "gold_label": 1,
            "split": "train",
            "gold_label_decision": "unanimous",
        }
    }
    source_rows = pd.DataFrame(
        [
            {"parent_identity_key": "parent-key", "source_record_id": "hf:0"},
            {"parent_identity_key": "parent-key", "source_record_id": "local:0:x"},
        ]
    )

    annotated = annotate_direct_source_rows(source_rows, parents, source_audit)

    assert annotated["audit_used_in_gold_vote"].tolist() == [True, True]
    assert set(annotated["audit_canonical_claim_id"]) == {"claim-1"}
    assert annotated["audit_source_record_is_claim_representative"].tolist() == [True, False]
    assert annotated["audit_claim_n_source_records"].tolist() == [2, 2]


def test_filtered_claim_propagates_exact_rule_reason_to_source_record():
    claims = pd.DataFrame(
        [
            {
                "canonical_claim_id": "claim-2",
                "source_record_id": "hf:1",
                "source_record_ids": ["hf:1"],
                "n_source_records": 1,
                "cross_source_deduplicated": False,
                "audit_canonical_claim_id": "claim-2",
                "audit_claim_decision": "filtered_by_gold_rule",
                "audit_filter_reason": "nonhuman_or_unresolved_population",
                "audit_candidate_label": None,
                "audit_vote_direction": "not_a_vote",
                "audit_used_in_gold_vote": False,
            }
        ]
    )
    parents = {
        "parent-key": {
            "gold_label": 1,
            "split": "valid",
            "gold_label_decision": "accepted_record_majority",
        }
    }

    annotated = annotate_direct_source_rows(
        pd.DataFrame(
            [{"parent_identity_key": "parent-key", "source_record_id": "hf:1"}]
        ),
        parents,
        _source_audit_by_id(claims),
    )

    assert annotated.loc[0, "audit_pipeline_status"] == "filtered_by_gold_rule_via_canonical_claim"
    assert annotated.loc[0, "audit_filter_reason"] == "nonhuman_or_unresolved_population"
    assert not bool(annotated.loc[0, "audit_used_in_gold_vote"])
