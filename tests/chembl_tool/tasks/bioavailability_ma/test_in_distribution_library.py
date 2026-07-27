"""Tests for the in-distribution (starling hf_cleaned) evidence/index builder."""

import json

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


def test_prebuilt_manifest_keeps_every_survivor_from_raw_top_100(tmp_path, monkeypatch):
    from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
        CATALOG_SCHEMA_VERSION,
        V6_5_TEMPLATE_PROFILE,
        template_bundle_hash,
    )
    from tools.chembl_tool.tasks.bioavailability_ma.reranking.build_assay_transfer_rerank_catalog import (
        build_manifest_for_prebuilt_catalog,
    )

    candidates = []
    catalog_records = []
    for index in range(100):
        record = in_distribution_catalog_record(
            _record(smiles=f"C{'C' * (index + 1)}", child_id=f"r{index}")
        )
        catalog_records.append(record)
        candidates.append(
            {
                "molecule_chembl_id": f"m{index}",
                "canonical_smiles": record["canonical_smiles"],
                "similarity": 1.0 - index / 1000,
                "structural_rank": index + 1,
                "evidence_rows": [{"catalog_record": record}],
            }
        )

    def fake_retrieve(query_smiles, index, *, reranker, **kwargs):
        assert kwargs["assay_transfer_initial_morgan_filter"] == 100
        reranker.rerank_records(
            query_smiles=query_smiles,
            group_id="Fg.gut_wall_efflux_intestinal_metabolism",
            candidates=candidates,
        )
        return {}

    monkeypatch.setattr(
        "tools.chembl_tool.common.experiment_retrieval.retrieve_experiment_view",
        fake_retrieve,
    )
    catalog = tmp_path / "catalog.jsonl"
    metadata = {
        "record_type": "catalog_metadata",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": "in-distribution-test",
        "template_hash": template_bundle_hash(profile=V6_5_TEMPLATE_PROFILE),
        "template_profile": V6_5_TEMPLATE_PROFILE,
    }
    catalog.write_text(
        "\n".join(json.dumps(row) for row in [metadata, *catalog_records]) + "\n"
    )
    manifest = tmp_path / "manifest.jsonl"
    summary = build_manifest_for_prebuilt_catalog(
        records=[{"drug": "CCO"}],
        indices=[0],
        smiles_field="drug",
        index={"version": "test"},
        catalog_path=catalog,
        manifest_output=manifest,
        experiment_mode="full_mechanism",
        top_k_per_group=10,
        min_similarity=0.0,
        neighbor_identity_policy="parent_disjoint",
        initial_morgan_filter=100,
    )

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert summary["n_candidates"] == 100
    assert len(rows[1]["candidates"]) == 100
    assert rows[1]["candidates"][-1]["structural_rank"] == 100
