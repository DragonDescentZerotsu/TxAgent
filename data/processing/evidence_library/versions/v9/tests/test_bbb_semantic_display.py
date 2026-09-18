import copy

import pytest

from data.processing.evidence_library.versions.v9.tasks.bbb_martins.semantic_display import (
    semantic_display_fields,
    semantic_prompt_payload,
)
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.transporter_identifiers import (
    canonical_transporter_identifier,
)


def test_binary_display_is_semantic_without_changing_record():
    record = {
        "canonical_measurement_text": "-1",
        "canonical_unit_text": "binary_outcome_class",
        "canonical_measurement_scale_id": "efflux_substrate_binary.v1",
        "canonical_category_id": "negative",
    }

    assert semantic_display_fields(record) == {
        "measurement_text": "non-substrate",
        "unit_text": "efflux-transporter substrate status",
        "transporter_identifier": None,
    }
    assert record["canonical_measurement_text"] == "-1"


def test_database_identifiers_resolve_inside_composites():
    raw = "UniProt:P08183 / NCBIGene:9429"

    assert canonical_transporter_identifier(raw) == (
        "ATP-dependent translocase ABCB1 [human] / "
        "ABCG2 - ATP binding cassette subfamily G member 2 (JR blood group) "
        "[human]"
    )


def test_canonical_transporter_omits_database_accessions():
    assert canonical_transporter_identifier("UniProt:P06795") == (
        "Abcb1b - ATP-dependent translocase ABCB1 [mouse]"
    )
    assert canonical_transporter_identifier("UMLS:C0069906") == "P-glycoproteins"


def test_unresolved_source_identifier_is_not_used_as_canonical():
    assert canonical_transporter_identifier("SMILES:8244578") is None
    assert semantic_display_fields({
        "canonical_transporter_identifier": None,
        "transporter_identifier": "SMILES:8244578",
    })["transporter_identifier"] is None


@pytest.mark.parametrize("scale,field", [
    ("bbb_permeability_binary.v1", "bbb_permeability_label"),
    ("passive_bbb_interpretation_binary.v1", "passive_bbb_interpretation"),
    ("efflux_substrate_binary.v1", "interaction_conclusion"),
    ("efflux_inhibitor_binary.v1", "interaction_conclusion"),
])
@pytest.mark.parametrize("category,value", [("negative", -1), ("positive", 1)])
def test_projection_consumes_encoder_input_without_mutating_provenance(scale, field, category, value):
    record = {"canonical_measurement_scale_id": scale, "canonical_category_id": category,
              "finite_scalar_value": value, "canonical_measurement_text": str(value),
              "canonical_unit_text": "binary_outcome_class", "measurement_kind": "binary"}
    payload = {field: "original", "bbb_transport_label": "transporter_mediated", "support_text": "evidence"}
    original = copy.deepcopy((record, payload))
    projected = semantic_prompt_payload(record, payload)
    assert field not in projected
    assert projected["bbb_transport_label"] == payload["bbb_transport_label"]
    assert projected["measurement_text"] != str(value)
    assert projected["unit_text"] != record["canonical_unit_text"]
    assert (record, payload) == original


def test_invalid_binary_metadata_fails_instead_of_guessing():
    with pytest.raises(ValueError, match="unsupported BBB display scale"):
        semantic_prompt_payload({"measurement_kind": "binary"}, {})
    with pytest.raises(ValueError, match="invalid BBB category"):
        semantic_display_fields({"canonical_measurement_scale_id": "bbb_permeability_binary.v1",
                                 "finite_scalar_value": 0})
