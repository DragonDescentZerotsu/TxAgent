from __future__ import annotations

import pytest

from tools.chembl_tool.common.starling.categorical_response import (
    BINARY_OUTCOME_UNIT,
    ORDINAL_OUTCOME_UNIT,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_categorical_response import (
    POLICY,
    canonical_fg_target_id,
    classify_direct_qualitative_text,
    encoding_policy_manifest,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_policy import _enrich_record
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMNS,
)


class _NoAuxiliary:
    def attach(self, record):
        return {
            "global_context": None,
            "global_species_context": None,
            "auxiliary_mapping_status": "not_applicable",
        }


class _NoReference:
    def attach(self, record):
        return {"canonical_reference_scope": "not_applicable"}


def _apply(source_id: str, **fields):
    return POLICY.apply(
        {"source_id": source_id, "finite_scalar_value": None, **fields}
    )


@pytest.mark.parametrize("value", ["high", "good", "excellent", "nearly complete"])
def test_direct_positive_wording_uses_the_high_category(value):
    encoded = _apply(
        "hf_bioavailability",
        measurement_text=value,
        bioavailability_report_type="absolute",
    )
    assert encoded["finite_scalar_value"] == 1.0
    assert encoded["canonical_unit"] == ORDINAL_OUTCOME_UNIT
    assert encoded["categorical_encoder_id"] == (
        "direct_oral_bioavailability_ordinal.v1"
    )


@pytest.mark.parametrize("value", ["very low", "poor", "negligible", "minimal"])
def test_direct_negative_wording_uses_the_low_category(value):
    encoded = _apply(
        "hf_bioavailability",
        measurement_text=value,
        bioavailability_report_type="absolute",
    )
    assert encoded["finite_scalar_value"] == -1.0


@pytest.mark.parametrize(
    "value",
    [
        "moderate",
        "intermediate",
        "moderate BA",
        "intermediate oral bioavailability",
        "moderate absolute oral bioavailability",
    ],
)
def test_direct_middle_wording_uses_the_middle_category(value):
    encoded = _apply(
        "hf_bioavailability",
        measurement_text=value,
        bioavailability_report_type="absolute",
    )
    assert encoded["finite_scalar_value"] == 0.0
    assert encoded["canonical_unit"] == ORDINAL_OUTCOME_UNIT
    assert encoded["categorical_encoder_id"] == (
        "direct_oral_bioavailability_ordinal.v1"
    )


@pytest.mark.parametrize(
    "value",
    [
        "low to moderate",
        "moderate to high",
        "moderate/high",
        "orally bioavailable",
        "2-fold higher",
        "higher than reference",
        "low to high",
        "high (40%)",
        "not high",
        "not good",
        "not complete",
        "did not appear to have low",
        "not reported",
        "",
    ],
)
def test_direct_ambiguous_relative_numeric_and_conflicting_wording_abstains(value):
    assert not _apply(
        "hf_bioavailability",
        measurement_text=value,
        bioavailability_report_type="absolute",
    )


def test_nondirect_hf_wording_never_uses_the_direct_encoder():
    assert not _apply(
        "hf_bioavailability",
        measurement_text="high",
        bioavailability_report_type="relative_comparison",
    )


def test_direct_qualitative_classifier_keeps_evidence_reason_vocabulary():
    assert classify_direct_qualitative_text("high") == (
        "high",
        "explicit_qualitative_high",
    )
    assert classify_direct_qualitative_text("poor") == (
        "low",
        "explicit_qualitative_low",
    )
    assert classify_direct_qualitative_text("moderate") == (
        "middle",
        "explicit_qualitative_middle",
    )
    assert classify_direct_qualitative_text("not high") == (
        None,
        "unmapped_or_ambiguous_qualitative_value",
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [("substrate", 1.0), ("not_substrate", -1.0), ("not substrate", -1.0)],
)
def test_fg_status_is_binary_and_target_scoped(status, expected):
    encoded = _apply(
        "fg",
        substrate_status=status,
        transporter_or_enzyme="P-gp/ABCB1",
    )
    assert encoded["finite_scalar_value"] == expected
    assert encoded["categorical_encoder_id"] == "fg_substrate_status_binary.v1"
    assert encoded["canonical_unit"] == BINARY_OUTCOME_UNIT


@pytest.mark.parametrize("status", ["inconclusive", "inhibited", "unknown", ""])
def test_fg_uninformative_statuses_abstain(status):
    assert not _apply(
        "fg",
        substrate_status=status,
        transporter_or_enzyme="ABCB1",
    )


def test_fg_missing_target_abstains():
    assert not _apply(
        "fg",
        substrate_status="substrate",
        transporter_or_enzyme=None,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("P-gp/ABCB1", "ABCB1"),
        ("P‐gp/ABCB1", "ABCB1"),
        ("MDR1", "ABCB1"),
        ("BCRP", "ABCG2"),
        ("MRP-2/ABCC2", "ABCC2"),
        ("PEPT1 (SLC15A1)", "SLC15A1"),
        ("SGLT-1", "SLC5A1"),
        ("P-gp/ABCB1 and BCRP/ABCG2", "ABCB1+ABCG2"),
        ("CYP3A", "CYP3A"),
        ("CYP3A4", "CYP3A4"),
        ("new transporter", "source_token:new_transporter"),
    ],
)
def test_fg_target_aliases_are_reviewed_and_fail_conservatively(raw, expected):
    assert canonical_fg_target_id(raw) == expected


def test_real_scalar_always_precedes_a_categorical_anchor():
    assert not POLICY.apply(
        {
            "source_id": "fg",
            "finite_scalar_value": 0.37,
            "substrate_status": "substrate",
            "transporter_or_enzyme": "ABCB1",
        }
    )


def _normalized_record(source_id: str, **updates):
    record = {
        **{field: None for field in SOURCE_COLUMNS[source_id]},
        "source_id": source_id,
        "source_row_number": 1,
        "source_record_id": "record-1",
        "measurement_text": None,
        "unit_text": None,
        "canonical_endpoint": "oral_bioavailability" if source_id == "hf_bioavailability" else "intestinal_efflux",
        "canonical_measurement": None,
        "canonical_unit": None,
        "finite_scalar_value": None,
        "measurement_unit_status": "missing_measurement",
        "unit_notation_status": "none",
        "unit_notation_factor": None,
        "structure_status": "resolved",
        "canonical_smiles": "CCO",
        "smiles": "CCO",
        "bioavailability_report_type": "absolute" if source_id == "hf_bioavailability" else None,
    }
    record.update(updates)
    return record


def test_enrichment_assigns_semantic_direct_endpoint_and_producer_ids():
    enriched = _enrich_record(
        _normalized_record(
            "hf_bioavailability",
            measurement_text="high",
            oral_bioavailability_value="high",
        ),
        _NoAuxiliary(),
        _NoReference(),
    )
    assert enriched["canonical_endpoint"] == "oral_bioavailability_outcome"
    assert enriched["categorical_encoder_id"] == (
        "direct_oral_bioavailability_ordinal.v1"
    )
    assert enriched["canonical_endpoint_producer_id"] == (
        "direct_oral_bioavailability_ordinal.v1"
    )
    assert enriched["canonical_pair_producer_id"] == (
        "direct_oral_bioavailability_ordinal.v1"
    )
    assert enriched["normalization_validity_status"] == "valid"


def test_enrichment_assigns_target_specific_fg_endpoint():
    enriched = _enrich_record(
        _normalized_record(
            "fg",
            substrate_status="substrate",
            transporter_or_enzyme="P-gp/ABCB1",
        ),
        _NoAuxiliary(),
        _NoReference(),
    )
    assert enriched["canonical_endpoint"] == "fg_substrate_outcome:ABCB1"
    assert enriched["normalization_validity_status"] == "valid"


def test_parseable_unresolved_numeric_measurement_is_not_overwritten():
    enriched = _enrich_record(
        _normalized_record(
            "fg",
            canonical_measurement="2.5",
            measurement_text="2.5",
            substrate_status="substrate",
            transporter_or_enzyme="ABCB1",
        ),
        _NoAuxiliary(),
        _NoReference(),
    )
    assert "categorical_encoder_id" not in enriched


def test_manifest_freezes_separate_domains_and_target_policy():
    manifest = encoding_policy_manifest()
    scales = {
        item["scale_id"]: item for item in manifest["controlled_measurements"]
    }
    assert [item["category_id"] for item in scales[
        "direct_oral_bioavailability_ordinal.v1"
    ]["categories"]] == ["low", "middle", "high"]
    assert [item["category_id"] for item in scales[
        "fg_substrate_status_binary.v1"
    ]["categories"]] == ["not_substrate", "substrate"]
    assert manifest["fg_target_alias_policy"]["family_isoform_collapse"] is False
