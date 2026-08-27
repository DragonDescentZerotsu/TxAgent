import json

import pandas as pd

from tools.chembl_tool.paper_experiments.build_assay_family_catalog import build_catalog


def test_normalized_catalog_preserves_record_families_with_assay_ordering(tmp_path):
    path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "group_id": "Tier 1.starling_direct_bbb_evidence",
                "canonical_assay_context": "shared assay",
                "canonical_endpoint_name": "brain exposure",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Mechanism.efflux_transport",
                "canonical_assay_context": "shared assay",
                "canonical_endpoint_name": "efflux",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Mechanism.passive_permeability",
                "canonical_assay_context": "passive assay",
                "canonical_endpoint_name": "Papp",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Mechanism.influx_transport",
                "canonical_assay_context": "influx assay",
                "canonical_endpoint_name": "uptake",
                "retrieval_eligible": True,
            },
        ]
    ).to_parquet(path, index=False)

    rows, manifest = build_catalog("bbb_martins", path)

    by_context = {row["assay_context"]: row for row in rows}
    assert by_context["shared assay"]["first_level"] == 1
    assert by_context["shared assay"]["family_levels"] == [1, 3]
    assert by_context["shared assay"]["first_endpoint_group"] == (
        "direct_brain_exposure"
    )
    assert by_context["shared assay"]["family_endpoint_groups"] == [
        "direct_brain_exposure",
        "efflux_transport",
    ]
    assert {
        row["source_group_id"]: row["endpoint_group"]
        for row in by_context["shared assay"]["source_families"]
    } == {
        "Mechanism.efflux_transport": "efflux_transport",
        "Tier 1.starling_direct_bbb_evidence": "direct_brain_exposure",
    }
    assert by_context["passive assay"]["first_level"] == 2
    assert manifest["n_physical_assays"] == 3
    assert manifest["n_multi_family_assays"] == 1
    assert manifest["family_assignment_unit"] == "source_record"
    assert "first_level is only" in manifest["progressive_overlap_policy"]


def test_bioavailability_catalog_includes_nondirect_outcome_family(tmp_path):
    path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "group_id": "Observed.direct_oral_bioavailability",
                "canonical_assay_context": "direct study",
                "canonical_endpoint_name": "oral bioavailability",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Observed.nondirect_oral_bioavailability",
                "canonical_assay_context": "relative human tablet study",
                "canonical_endpoint_name": "oral bioavailability",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Observed.oral_auc_cmax_exposure",
                "canonical_assay_context": "oral exposure study",
                "canonical_endpoint_name": "AUC",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Fa.absorption_solubility_permeability",
                "canonical_assay_context": "Caco-2",
                "canonical_endpoint_name": "Papp",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Fg.gut_wall_efflux_intestinal_metabolism",
                "canonical_assay_context": "efflux assay",
                "canonical_endpoint_name": "efflux ratio",
                "retrieval_eligible": True,
            },
            {
                "group_id": "Fh.hepatic_clearance_metabolic_stability",
                "canonical_assay_context": "microsomes",
                "canonical_endpoint_name": "clearance",
                "retrieval_eligible": True,
            },
        ]
    ).to_parquet(path, index=False)

    rows, manifest = build_catalog("bioavailability_ma", path)

    assert [level["family_id"] for level in manifest["levels"]] == [
        "Observed.direct_oral_bioavailability",
        "Observed.nondirect_oral_bioavailability",
        "Observed.oral_auc_cmax_exposure",
        "Fa.absorption_solubility_permeability",
        "Fg.gut_wall_efflux_intestinal_metabolism",
        "Fh.hepatic_clearance_metabolic_stability",
    ]
    assert [row["first_level"] for row in rows] == [1, 2, 3, 4, 5, 6]


def test_bbb_source_purity_catalog_adds_parallel_near_direct_level(tmp_path):
    path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "group_id": group,
                "canonical_assay_context": context,
                "canonical_endpoint_name": endpoint,
                "retrieval_eligible": True,
            }
            for group, context, endpoint in (
                ("Tier 1.starling_direct_bbb_evidence", "brain PK", "brain concentration"),
                ("Proxy.central_functional_access", "central PD", "target engagement"),
                ("Mechanism.passive_permeability", "PAMPA-BBB", "Papp"),
                ("Mechanism.efflux_transport", "MDCK-MDR1", "efflux ratio"),
                ("Mechanism.influx_transport", "OATP uptake", "uptake"),
            )
        ]
    ).to_parquet(path, index=False)

    rows, manifest = build_catalog(
        "bbb_martins",
        path,
        config_name="STARLING_SOURCE_PURITY",
    )

    assert [level["endpoint_group"] for level in manifest["levels"]] == [
        "direct_brain_exposure",
        "central_functional_access_proxy",
        "passive_permeability",
        "efflux_transport",
        "influx_transport",
    ]
    assert [row["first_level"] for row in rows] == [1, 2, 3, 4, 5]


def test_clintox_catalog_adds_one_explicit_clinical_context_unit(tmp_path):
    path = tmp_path / "catalog.jsonl"
    rows = [
        {
            "source_id": "clinical_trial_failure",
            "assay_id": "direct",
            "assay_context": "direct assay",
            "record_count": 2,
        },
        {
            "source_id": "general_cytotoxicity",
            "assay_id": "cytotox",
            "assay_context": "cytotox assay",
            "record_count": 3,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    catalog, manifest = build_catalog("clintox", path)

    assert [row["first_level"] for row in catalog] == [1, 2, 7]
    assert catalog[1]["assay_id"] == "STARLING_CLINTOX_CLINICAL_CONTEXT"
    assert manifest["n_physical_assays"] == 3
    assert manifest["n_multi_family_assays"] == 0
