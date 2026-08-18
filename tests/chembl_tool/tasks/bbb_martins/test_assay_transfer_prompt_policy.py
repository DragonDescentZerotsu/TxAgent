from __future__ import annotations

from tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline import (
    V11_DEFAULT_INDEX,
    _group_prompt_payload,
    _parse_args,
)


def test_bbb_v11_cli_selects_scaffold_index_and_record_bundle():
    args = _parse_args(
        [
            "--retrieval-strategy",
            "assay_transfer_tool",
            "--group-prompt-format",
            "assay_transfer_tool",
            "--assay-transfer-selection-unit",
            "unique_molecule",
            "--assay-transfer-records-per-molecule",
            "5",
        ]
    )

    assert args.index == V11_DEFAULT_INDEX
    assert args.assay_transfer_selection_unit == "unique_molecule"
    assert args.assay_transfer_records_per_molecule == 5
    assert args.assay_transfer_initial_morgan_filter == 50


def test_bbb_group_prompt_exposes_endpoint_distinct_record_bundle():
    source_record = {
        "source_contract": {
            "contract_version": "source_column_contract.v2",
            "source_or_simply_cleaned": {"endpoint_name": True},
        },
        "source_fields": {"endpoint_name": "BBB permeability"},
    }
    group = {
        "group_id": "direct_bbb",
        "tier": "Tier 1",
        "endpoint_group": "direct_bbb",
        "transfer_neighbor_selection": {"selection_score_is_llm_visible": True},
        "neighbors": [
            {
                "rank": 1,
                "molecule_chembl_id": "neighbor_1",
                "canonical_smiles": "CCO",
                "similarity": 0.5,
                "similarity_bucket": "moderate_analog",
                "transfer_selection_score": 0.9,
                "transfer_selected_records": [
                    {
                        "record_rank": 1,
                        "transfer_selection_score": 0.9,
                        "transfer_winning_record": source_record,
                    },
                    {
                        "record_rank": 2,
                        "transfer_selection_score": 0.8,
                        "transfer_winning_record": source_record,
                    },
                ],
                "prefetched_comparisons": [],
                "evidence_rows": [],
                "shared_assay_context": {},
            }
        ],
    }

    payload = _group_prompt_payload({"identity_hidden": True}, group)
    records = payload["neighbors"][0]["assay_transfer_records"]
    assert [record["assay_transfer_score"] for record in records] == [0.9, 0.8]
    assert "BBB probabilities" in " ".join(payload["instructions"])


def test_bbb_group_prompt_distinguishes_molecule_mean_from_representative_scores():
    source_record = {
        "source_contract": {
            "contract_version": "source_column_contract.v2",
            "source_or_simply_cleaned": {"endpoint_name": True},
        },
        "source_fields": {"endpoint_name": "brain exposure"},
    }
    group = {
        "group_id": "direct_bbb",
        "tier": "Tier 1",
        "endpoint_group": "direct_bbb",
        "transfer_neighbor_selection": {"selection_score_is_llm_visible": True},
        "neighbors": [
            {
                "rank": 1,
                "molecule_chembl_id": "neighbor_1",
                "canonical_smiles": "CCO",
                "similarity": 0.5,
                "similarity_bucket": "moderate_analog",
                "transfer_selection_score": 0.6,
                "transfer_molecule_mean_score": 0.6,
                "transfer_selected_records": [
                    {
                        "record_rank": 1,
                        "transfer_selection_score": 0.9,
                        "transfer_winning_record": source_record,
                    }
                ],
                "prefetched_comparisons": [],
                "evidence_rows": [],
                "shared_assay_context": {},
            }
        ],
    }
    payload = _group_prompt_payload({"identity_hidden": True}, group)
    family = payload["neighbors"][0]["assay_transfer_family_evidence"][0]
    assert family["molecule_mean_assay_transfer_score"] == 0.6
    assert family["representative_records"][0]["assay_transfer_score"] == 0.9
    assert "frozen Stage 07 Morgan representatives" in " ".join(payload["instructions"])
