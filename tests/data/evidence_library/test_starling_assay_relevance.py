from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.assay_catalog import (
    assay_unit,
    build_catalog,
)


def test_context_is_assay_and_endpoint_is_descriptive(tmp_path):
    records = pd.DataFrame(
        [
            {
                "canonical_record_id": "r1",
                "molecule_id": "m1",
                "canonical_endpoint_name": "apparent_permeability",
                "canonical_assay_context": "caco_2",
                "canonical_species_context": "human",
                "measurement_kind": "continuous",
                "assay_system": "Caco-2 monolayer",
            },
            {
                "canonical_record_id": "r2",
                "molecule_id": "m2",
                "canonical_endpoint_name": "efflux_ratio",
                "canonical_assay_context": "caco_2",
                "canonical_species_context": "human",
                "measurement_kind": "continuous",
                "assay_system": "Caco-2 monolayer",
            },
            {
                "canonical_record_id": "r3",
                "molecule_id": "m1",
                "canonical_endpoint_name": "oral_bioavailability",
                "canonical_assay_context": None,
                "canonical_species_context": "rat",
                "measurement_kind": "continuous",
                "assay_system": None,
            },
            {
                "canonical_record_id": "r4",
                "molecule_id": "m3",
                "canonical_endpoint_name": "oral_bioavailability",
                "canonical_assay_context": None,
                "canonical_species_context": "rat",
                "measurement_kind": "continuous",
                "assay_system": None,
            },
            {
                "canonical_record_id": "r5",
                "molecule_id": "m4",
                "canonical_endpoint_name": "skin_irritation",
                "canonical_assay_context": "patch test",
                "canonical_species_context": "human",
                "measurement_kind": "categorical",
                "assay_system": "patch test",
            },
        ]
    )
    membership = pd.DataFrame({"canonical_record_id": ["r1", "r2", "r3", "r4", "r5"]})
    records_path = tmp_path / "records.parquet"
    membership_path = tmp_path / "membership.parquet"
    pq.write_table(pa.Table.from_pandas(records), records_path)
    pq.write_table(pa.Table.from_pandas(membership), membership_path)

    catalog, manifest = build_catalog(
        task="bioavailability_ma",
        task_definition="Predict oral bioavailability.",
        records_path=records_path,
        membership_path=membership_path,
        min_molecules=2,
    )

    assert {row["assay_context"] for row in catalog} == {
        "caco_2",
        "endpoint_fallback::oral_bioavailability",
    }
    caco = next(row for row in catalog if row["assay_context"] == "caco_2")
    assert {item["value"] for item in caco["endpoints"]} == {
        "apparent_permeability",
        "efflux_ratio",
    }
    assert caco["unique_molecule_count"] == 2
    assert manifest["n_all_assay_units"] == 3
    assert manifest["n_eligible_assay_units"] == 2
    assert manifest["quality_gate"] == "none"
    assert "endpoints do not split assay units" in manifest["endpoint_role"]


def test_assay_unit_uses_endpoint_only_as_missing_context_fallback():
    assert assay_unit("pampa-bbb", "papp") == (
        "pampa-bbb",
        "canonical_assay_context",
    )
    assert assay_unit(None, "brain_to_plasma_ratio") == (
        "endpoint_fallback::brain_to_plasma_ratio",
        "endpoint_fallback",
    )
