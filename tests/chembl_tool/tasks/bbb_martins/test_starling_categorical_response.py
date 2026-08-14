from tools.chembl_tool.common.starling.categorical_response import BINARY_OUTCOME_UNIT
from tools.chembl_tool.tasks.bbb_martins.starling_categorical_response import (
    POLICY,
    encoding_policy_manifest,
)
from tools.chembl_tool.tasks.bbb_martins.starling_policy import (
    _endpoint_registry,
    _enrich_record,
)


class _NoAuxiliary:
    def attach(self, record):
        return {}


class _NoReference:
    def attach(self, record):
        return {
            "canonical_reference_scope": "not_applicable",
            "canonical_reference_basis": "none",
        }


def _direct_normalized_record(**updates):
    record = {
        "source_id": "direct_bbb",
        "finite_scalar_value": None,
        "canonical_measurement": None,
        "canonical_unit": None,
        "canonical_endpoint": "missing_endpoint",
        "measurement_unit_status": "missing_measurement",
        "spacing_and_spelling_status": "unchanged",
        "spacing_and_spelling_reason": "direct_missing_endpoint",
        "structure_status": "resolved",
        "canonical_smiles": "CCO",
        "source_index": 1,
        "pmid": "1",
        "support_text": "Permeable.",
        "bbb_permeability_label": "permeable",
        "bbb_transport_label": None,
        "quant_metric": None,
        "quant_value": None,
        "quant_units": None,
        "assay_model": None,
        "species": None,
        "qualifying_conditions": None,
        "extra_details": None,
        "smiles": "CCO",
    }
    record.update(updates)
    return record


def _apply(source_id: str, **fields):
    return POLICY.apply({"source_id": source_id, "finite_scalar_value": None, **fields})


def test_reviewed_binary_categories_have_frozen_signed_anchors():
    cases = (
        ("direct_bbb", {"bbb_permeability_label": "permeable"}, 1.0, "bbb_permeability_binary.v1"),
        ("direct_bbb", {"bbb_permeability_label": "good permeability"}, 1.0, "bbb_permeability_binary.v1"),
        ("direct_bbb", {"bbb_permeability_label": "poor penetration"}, -1.0, "bbb_permeability_binary.v1"),
        ("passive_permeability", {"passive_bbb_interpretation": "permeable_or_high"}, 1.0, "passive_bbb_interpretation_binary.v1"),
        ("passive_permeability", {"passive_bbb_interpretation": "impermeable_or_low"}, -1.0, "passive_bbb_interpretation_binary.v1"),
        ("efflux_transport", {"interaction_conclusion": "substrate"}, 1.0, "efflux_substrate_binary.v1"),
        ("efflux_transport", {"interaction_conclusion": "non-inhibitor"}, -1.0, "efflux_inhibitor_binary.v1"),
    )
    for source, fields, value, encoder in cases:
        encoded = _apply(source, **fields)
        assert encoded["finite_scalar_value"] == value
        assert encoded["canonical_unit"] == BINARY_OUTCOME_UNIT
        assert encoded["categorical_encoder_id"] == encoder


def test_unreviewed_transport_and_influx_prose_abstain():
    assert not _apply("direct_bbb", bbb_permeability_label="increased_permeability")
    assert not _apply("direct_bbb", bbb_transport_label="efflux substrate")
    assert not _apply("efflux_transport", interaction_conclusion="possible substrate")
    assert not _apply("influx_transport", reported_result="increased brain uptake")


def test_real_scalar_always_precedes_categorical_anchor():
    assert not POLICY.apply(
        {
            "source_id": "direct_bbb",
            "finite_scalar_value": 0.37,
            "bbb_permeability_label": "permeable",
        }
    )


def test_manifest_freezes_semantic_endpoints_and_encoder_key_boundary():
    manifest = encoding_policy_manifest()
    assert manifest["encoder_id_is_part_of_the_pair_bucket_key"] is True
    assert manifest["parseable_unresolved_unit_measurement_precedence"] is True
    assert manifest["semantic_endpoint_by_encoder"]["efflux_substrate_binary.v1"] == "efflux_substrate_outcome"


def test_parseable_numeric_with_unresolved_unit_is_not_overwritten_by_binary_anchor():
    enriched = _enrich_record(
        _direct_normalized_record(
            canonical_measurement="2.5",
            measurement_unit_status="cleaned_only_unrecognized_unit",
            quant_value="2.5",
        ),
        _NoAuxiliary(),
        _NoReference(),
    )
    assert "categorical_encoder_id" not in enriched
    assert enriched["canonical_measurement_source"] == "source_scalar"
    assert enriched["canonical_unit_resolution_source"] == "unresolved"


def test_categorical_endpoint_and_provenance_are_semantic():
    enriched = _enrich_record(
        _direct_normalized_record(), _NoAuxiliary(), _NoReference()
    )
    assert enriched["canonical_endpoint"] == "bbb_permeability_outcome"
    assert enriched["canonical_measurement"] == "1"
    assert enriched["canonical_unit"] == BINARY_OUTCOME_UNIT
    assert enriched["canonical_endpoint_source_field"] == "bbb_permeability_label"
    assert enriched["canonical_endpoint_policy_status"] == "categorical_encoded"
    assert enriched["canonical_endpoint_rule_id"] == "bbb_permeability_binary.v1"


def test_endpoint_registry_separates_scalar_and_categorical_variants():
    base = {
        "source_id": "passive_permeability",
        "endpoint_name": "Pe",
        "spacing_and_spelling_endpoint": "effective_permeability",
        "spacing_and_spelling_status": "reviewed",
        "spacing_and_spelling_reason": "passive_effective_permeability",
        "spacing_and_spelling_version": "test",
    }
    registry = _endpoint_registry(
        [
            {**base, "canonical_endpoint": "effective_permeability"},
            {
                **base,
                "canonical_endpoint": "passive_bbb_permeability_outcome",
                "categorical_encoder_id": "passive_bbb_interpretation_binary.v1",
            },
        ]
    )
    assert len(registry["passive_permeability"]) == 2
    assert {row["decision_variant"] for row in registry["passive_permeability"]} == {
        "source_measurement",
        "passive_bbb_interpretation_binary.v1",
    }
