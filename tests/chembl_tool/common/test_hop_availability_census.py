from tools.chembl_tool.common.hop_availability_census import (
    CandidateSpec,
    classify_row,
)


def _spec(mode: str) -> CandidateSpec:
    return CandidateSpec(
        family_id="candidate",
        display_name="candidate",
        declared_level="H1",
        parent_c_family_id="C",
        measured_node="node",
        admissible_path=("node", "C"),
        selector_kind="target",
        selector_values=("CHEMBL1",),
        assay_mode=mode,
        citations=(),
    )


def _row(**updates):
    row = {
        "canonical_smiles": "CCO",
        "data_validity_comment": "",
        "potential_duplicate": 0,
        "confidence_score": 9,
        "relationship_type": "D",
        "description": "Inhibition of human target activity",
        "target_pref_name": "Target",
        "standard_type": "IC50",
    }
    row.update(updates)
    return row


def test_functional_target_accepts_direct_activity():
    assert classify_row(_row(), _spec("functional_target")) == "accepted"


def test_functional_target_rejects_binding_only_description():
    assert (
        classify_row(
            _row(description="Binding affinity by KINOMEscan", standard_type="Ki"),
            _spec("functional_target"),
        )
        == "binding_only_assay"
    )


def test_direct_ppi_requires_interaction_semantics():
    assert (
        classify_row(
            _row(
                target_pref_name="Keap1/Nrf2",
                description="Inhibition of KEAP1-NRF2 protein-protein interaction",
            ),
            _spec("direct_ppi"),
        )
        == "accepted"
    )
    assert classify_row(_row(), _spec("direct_ppi")) == "description_not_direct_ppi"


def test_property_measurement_keeps_valid_measurement_without_target_gate():
    row = _row(confidence_score=0, relationship_type="")
    assert classify_row(row, _spec("property_measurement")) == "accepted"
