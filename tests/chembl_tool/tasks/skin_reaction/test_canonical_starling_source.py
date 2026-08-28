from __future__ import annotations

import pandas as pd

from tools.chembl_tool.common.starling.normalization.organization import (
    organize_normalized_records,
)
from tools.chembl_tool.tasks.skin_reaction.build_canonical_starling_source import (
    build_canonical_frames,
)
from tools.chembl_tool.tasks.skin_reaction.canonical_starling_source import (
    AOP_PARTITION,
    DIRECT_PARTITION,
    PartitionDecision,
    REJECT_PARTITION,
    classify_aop_source_record,
    classify_direct_source_record,
    direct_outcome_reason,
)
from tools.chembl_tool.tasks.skin_reaction import direct_record_mapping
from tools.chembl_tool.tasks.skin_reaction.starling_spacing_and_spelling import (
    family_assignment,
)


def test_direct_source_routes_final_outcome_aop_and_photo_records():
    direct = {
        "reaction_type": "sensitization",
        "assay_or_test": "LLNA",
        "support_text": "A positive local lymph node assay was reported.",
    }
    aop = {
        "reaction_type": "sensitization",
        "assay_or_test": "DPRA",
        "support_text": "Cysteine depletion was positive.",
    }
    unicode_aop = {
        "reaction_type": "sensitization",
        "assay_or_test": "U‑SENS™",
        "support_text": "CD86 expression increased.",
    }
    photo = {
        "reaction_type": "allergic_contact_dermatitis_contact_allergy",
        "assay_or_test": "clinical case report",
        "support_text": "Berloque dermatitis followed UVA exposure.",
    }

    assert classify_direct_source_record(direct).partition == DIRECT_PARTITION
    assert classify_direct_source_record(aop).partition == AOP_PARTITION
    assert classify_direct_source_record(aop).aop_event == "MIE_protein_binding"
    assert classify_direct_source_record(unicode_aop).partition == AOP_PARTITION
    assert classify_direct_source_record(unicode_aop).aop_event == "KE3_dendritic_cell_activation"
    assert classify_direct_source_record(photo).partition == REJECT_PARTITION


def test_direct_source_checks_population_context_for_photo_and_in_silico_rows():
    photo = {
        "reaction_type": "sensitization",
        "assay_or_test": "patch test",
        "support_text": "A positive reaction was reported.",
        "species_or_population": "patients with photoallergic contact dermatitis",
    }
    prediction = {
        "reaction_type": "sensitization",
        "assay_or_test": "clinical observation",
        "support_text": "A positive result was reported.",
        "species_or_population": "in silico model",
    }

    assert classify_direct_source_record(photo).reason == "out_of_scope_photo_hazard"
    assert classify_direct_source_record(prediction).reason == "prediction_only"


def test_direct_source_rejects_defined_approach_but_not_subject_count_wording():
    defined_approach = {
        "reaction_type": "sensitization",
        "assay_or_test": "skin sensitization assessment",
        "support_text": "The conclusion used an OECD defined approach.",
    }
    subject_count = {
        "reaction_type": "sensitization",
        "assay_or_test": "patch test",
        "support_text": "Two out of three subjects had a positive patch test.",
    }

    assert classify_direct_source_record(defined_approach).partition == REJECT_PARTITION
    assert classify_direct_source_record(subject_count).partition == DIRECT_PARTITION


def test_aop_source_moves_adverse_outcome_and_keeps_only_key_events():
    outcome = {
        "aop_event": "adverse_outcome_skin_sensitization",
        "assay_type": "GPMT",
        "result_label": "positive",
        "support_text": "The guinea pig maximization test was positive.",
    }
    key_event = {
        "aop_event": "KE3_dendritic_cell_activation",
        "assay_type": "h-CLAT",
        "result_label": "positive",
        "support_text": "CD86 expression increased.",
    }
    prediction = {
        "aop_event": "integrated_or_unspecified",
        "assay_type": "GARDskin prediction",
        "result_label": "positive",
        "support_text": "An in-silico model prediction was positive.",
    }
    condition_prediction = {
        "aop_event": "MIE_protein_binding",
        "assay_type": "OECD protein binding alert",
        "result_label": "negative",
        "support_text": "No alert was found.",
        "experimental_conditions": "In silico prediction using OECD QSAR Toolbox",
    }
    integrated = {
        "aop_event": "adverse_outcome_skin_sensitization",
        "assay_type": "2o3 DA (DPRA, LuSens, h-CLAT)",
        "result_label": "positive",
        "support_text": "The 2 out of 3 defined approach predicted a sensitizer.",
    }

    assert classify_aop_source_record(outcome).partition == DIRECT_PARTITION
    assert classify_aop_source_record(key_event).partition == AOP_PARTITION
    assert classify_aop_source_record(prediction).partition == REJECT_PARTITION
    assert classify_aop_source_record(condition_prediction).partition == REJECT_PARTITION
    assert classify_aop_source_record(integrated).partition == REJECT_PARTITION


def test_direct_assay_result_is_detected_for_retrieval_purity_overlay():
    llna_mislabeled_as_ke4 = {
        "aop_event": "KE4_T_cell_activation",
        "assay_type": "LLNA",
        "result_label": "negative",
        "support_text": "The local lymph node assay was negative.",
    }
    hclat = {
        "aop_event": "KE3_dendritic_cell_activation",
        "assay_type": "h-CLAT",
        "result_label": "positive",
        "support_text": "CD86 expression increased.",
    }

    assert (
        direct_outcome_reason(llna_mislabeled_as_ke4)
        == "validated_direct_assay_outcome_overrides_aop_tag"
    )
    assert classify_aop_source_record(llna_mislabeled_as_ke4).partition == AOP_PARTITION
    assert direct_outcome_reason(hclat) == ""
    assert classify_aop_source_record(hclat).partition == AOP_PARTITION


def test_canonical_build_is_mutually_exclusive_and_deduplicates_cross_source():
    common = {
        "pmid": "1",
        "extraction_id": "ext_1",
        "confidence": 0.9,
        "paragraph_idx": 0,
        "support_text": "The LLNA was positive.",
        "SMILES": "CCO",
    }
    direct = pd.DataFrame(
        [
            {
                **common,
                "outcome_label": "positive",
                "reaction_type": "sensitization",
                "assay_or_test": "LLNA",
                "species_or_population": "mouse",
                "dose_or_concentration": "",
                "positive_count": "",
                "total_tested": "",
                "effect_metric": "",
                "extra_details": "",
            },
            {
                **common,
                "pmid": "2",
                "support_text": "DPRA cysteine depletion was positive.",
                "outcome_label": "positive",
                "reaction_type": "sensitization",
                "assay_or_test": "DPRA",
                "species_or_population": "in chemico",
                "dose_or_concentration": "",
                "positive_count": "",
                "total_tested": "",
                "effect_metric": "",
                "extra_details": "",
            },
        ]
    )
    aop = pd.DataFrame(
        [
            {
                **common,
                "global_identifier": "",
                "assay_type": "LLNA",
                "aop_event": "adverse_outcome_skin_sensitization",
                "endpoint_or_target": "",
                "result_label": "positive",
                "result_value": "",
                "result_unit": "",
                "experimental_conditions": "mouse",
                "qualifying_conditions": "",
                "extra_details": "",
                "needs_more_context": False,
            },
            {
                **common,
                "pmid": "3",
                "support_text": "CD86 expression increased.",
                "global_identifier": "",
                "assay_type": "h-CLAT",
                "aop_event": "KE3_dendritic_cell_activation",
                "endpoint_or_target": "CD86",
                "result_label": "positive",
                "result_value": "",
                "result_unit": "",
                "experimental_conditions": "THP-1",
                "qualifying_conditions": "",
                "extra_details": "",
                "needs_more_context": False,
            },
        ]
    )

    result = build_canonical_frames(direct, aop)

    assert len(result["direct_records"]) == 1
    assert len(result["aop_records"]) == 2
    assert set(result["aop_records"]["aop_event"]) == {
        "MIE_protein_binding",
        "KE3_dendritic_cell_activation",
    }
    assert len(result["dedup_audit"]) == 1
    assert result["stats"]["partition_reconciles"] is True
    assert result["stats"]["direct_aop_source_record_overlap"] == 0


def test_normalized_routing_obeys_canonical_partition_across_both_raw_sources():
    direct = family_assignment(
        "sensitization_aop",
        "adverse_outcome_skin_sensitization",
        {"canonical_sensitization_partition": DIRECT_PARTITION},
    )
    aop = family_assignment(
        "direct_skin_reaction",
        "sensitization",
        {"canonical_sensitization_partition": AOP_PARTITION},
    )
    rejected = family_assignment(
        "direct_skin_reaction",
        "sensitization",
        {"canonical_sensitization_partition": REJECT_PARTITION},
    )

    assert direct.group_id == "Direct.skin_reaction"
    assert aop.group_id == "Mechanism.sensitization_aop"
    assert rejected is None


def test_direct_mapping_counts_aop_source_final_outcomes_only(monkeypatch):
    decisions = {
        "sensitization_aop:0": PartitionDecision(
            DIRECT_PARTITION, "validated_direct_outcome"
        ),
        "direct_skin_reaction:0": PartitionDecision(
            AOP_PARTITION, "mechanistic_assay_in_direct_source", "MIE_protein_binding"
        ),
        "direct_skin_reaction:1": PartitionDecision(
            REJECT_PARTITION, "out_of_scope_irritation"
        ),
    }
    monkeypatch.setattr(direct_record_mapping, "load_partition_audit", lambda: decisions)
    monkeypatch.setattr(direct_record_mapping, "read_jsonl", lambda _path: [])
    records = [
        {
            "canonical_record_id": "aop-direct",
            "source_id": "sensitization_aop",
            "source_row_number": 1,
            "result_label": "positive",
            "qualifying_conditions": None,
        },
        {
            "canonical_record_id": "direct-aop",
            "source_id": "direct_skin_reaction",
            "source_row_number": 1,
            "qualifying_conditions": None,
        },
        {
            "canonical_record_id": "direct-reject",
            "source_id": "direct_skin_reaction",
            "source_row_number": 2,
            "qualifying_conditions": None,
        },
    ]

    mapped = direct_record_mapping.build_direct_record_mapping(records)

    assert len(mapped) == 1
    assert mapped[0]["canonical_record_id"] == "aop-direct"
    assert mapped[0]["direct_vote_status"] == "counted"
    assert mapped[0]["direct_vote_label"] == 1
    assert mapped[0]["retrieval_source_id"] == "direct_vote"
    assert mapped[0]["direct_group_id"] == "Direct.skin_reaction"


def test_partition_reject_reason_excludes_an_otherwise_retrievable_row():
    records, _, exclusions, _ = organize_normalized_records(
        [
            {
                "normalized_record_id": "rejected-row",
                "source_record_id": "direct_skin_reaction:1",
                "source_id": "direct_skin_reaction",
                "source_row_number": 2,
                "canonical_smiles": "CCO",
                "group_id": "Direct.skin_reaction",
                "canonical_endpoint_name": "sensitization",
                "retrieval_exclusion_reason": (
                    "canonical_skin_partition_reject:out_of_scope_irritation"
                ),
            }
        ]
    )

    assert records[0]["retrieval_eligible"] is False
    assert records[0]["organization_status"] == (
        "canonical_skin_partition_reject:out_of_scope_irritation"
    )
    assert exclusions[0]["retrieval_exclusion_reason"] == (
        "canonical_skin_partition_reject:out_of_scope_irritation"
    )
