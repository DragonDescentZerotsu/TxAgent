"""Tests for the in-distribution (starling hf_cleaned) evidence/index builder."""

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.reranking.build_starling_in_distribution_library import (
    build_in_distribution_evidence_rows,  # noqa: F401  (imported for symmetry)
    concept_from_endpoint_key,
    evidence_row_from_record,
    in_distribution_catalog_record,
    molecule_id_for,
)


@pytest.mark.parametrize(
    "key,concept",
    [
        ("q2.intestinal_absorption.fraction_absorbed.percent", "Fa"),
        ("q3.intestinal_transport.efflux_ratio.dimensionless_ratio.secretory_over_absorptive", "Fg"),
        ("q4.hepatic.intrinsic_clearance.ml_min_kg", "Fh"),
        ("q1.oral_bioavailability.f.percent", "oral_bioavailability"),
        ("q1.oral_exposure.cmax.ng_ml", "oral_exposure"),
        ("index.Fg.efflux.not_reported", None),  # TxAgent's synthetic key is rejected
        ("", None),
    ],
)
def test_concept_mapping(key, concept):
    assert concept_from_endpoint_key(key) == concept


def _record(**over):
    base = {
        "smiles": "c1ccccc1",
        "canonical_endpoint_key": "q3.intestinal_transport.efflux_ratio.dimensionless_ratio.secretory_over_absorptive",
        "endpoint_subtype": "efflux_ratio",
        "unit_basis": "dimensionless",
        "metric_type": "dimensionless_ratio",
        "scalar_value": 1.5,
        "support_text": "B>A transport observed",
        "extra_details": "60 ± 3 -> 44 ± 3",
        "transporter_or_enzyme": "BCRP/ABCG2",
        "substrate_status": "substrate",
        "record_id": "abc123",
        "source_row_number": 7,
    }
    base.update(over)
    return base


def test_evidence_row_preserves_normalized_fields():
    row = evidence_row_from_record(_record())
    assert row["group_id"] == "Fg.gut_wall_efflux_intestinal_metabolism"
    assert row["assay_tier"] == "Fg"
    # normalized value/endpoint flow into the minimal_evidence contract fields
    assert row["standard_type"].startswith("q3.")
    assert row["standard_value"] == 1.5
    assert row["standard_units"] == "dimensionless"
    # the full normalized record is carried for the catalog/template/presentation
    rec = row["starling_record"]
    assert rec["canonical_endpoint_key"].startswith("q3.")
    assert rec["scalar_value"] == 1.5
    assert rec["assay_concept"] == "Fg"
    assert rec["support_text"] == "B>A transport observed"
    assert rec["transporter_or_enzyme"] == "BCRP/ABCG2"
    # no synthetic index.* key anywhere
    assert "index." not in row["standard_type"]


def test_rows_out_of_scope_or_no_smiles_are_dropped():
    assert evidence_row_from_record(_record(canonical_endpoint_key="index.Fg.x.y")) is None
    assert evidence_row_from_record(_record(smiles="")) is None


def test_molecule_id_is_stable_per_smiles():
    a = molecule_id_for("c1ccccc1")
    b = molecule_id_for("c1ccccc1")
    c = molecule_id_for("CCO")
    assert a == b and a != c and a.startswith("STARLING_ID_")


def test_catalog_record_renders_in_distribution_v6_5():
    from tools.chembl_tool.tasks.bioavailability_ma.reranking.build_starling_in_distribution_library import (
        in_distribution_catalog_record,
    )
    from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
        AssayTransferPromptRenderer,
        V6_5_TEMPLATE_PROFILE,
    )

    cat = in_distribution_catalog_record(_record())
    assert cat["assay_concept"] == "Fg"
    assert cat["canonical_endpoint_key"].startswith("q3.")
    assert cat["record_id"] == "abc123"  # uses the starling record_id
    prompt = AssayTransferPromptRenderer(profile=V6_5_TEMPLATE_PROFILE).render(cat, _record()["smiles"])
    assert "endpoint: q3." in prompt
    assert "known value: 1.5 dimensionless" in prompt
    assert "substrate status: substrate" in prompt
    assert "index." not in prompt  # never the synthetic reconstructed key


def test_categorical_or_nonscalar_record_is_unscoreable():
    # substrate-status style record with no finite scalar -> no catalog record
    assert in_distribution_catalog_record(_record(scalar_value=None)) is None
    assert in_distribution_catalog_record(_record(scalar_value="substrate")) is None


def test_support_text_joined_by_child_id_for_presentation():
    # eligible-style record: no raw support_text column, narrative comes from the join.
    rec = _record(child_id="CID1", support_text=None, context_extra_details="short details")
    # with a join match -> full narrative
    cat = in_distribution_catalog_record(rec, support_by_id={"CID1": "full narrative sentence"})
    assert cat["support_text"] == "full narrative sentence"
    # without a match -> falls back to context_extra_details
    cat2 = in_distribution_catalog_record(rec, support_by_id={"OTHER": "x"})
    assert cat2["support_text"] == "short details"
