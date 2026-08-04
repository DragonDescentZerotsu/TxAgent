"""Pinned source contracts for the BBB Martins normalized-v6 library."""

from __future__ import annotations

from pathlib import Path

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.contracts import NormalizedSourceProfile
from tools.chembl_tool.tasks.bbb_martins.starling_benchmark import SOURCE_REVISION


EXPECTED_SOURCE_ROWS = {
    "direct_bbb": 304_845,
    "passive_permeability": 10_119,
    "efflux_transport": 191_262,
    "influx_transport": 75_720,
}

EXPECTED_SOURCE_SHA256 = {
    "direct_bbb": "14c01314377b6c31f18e6ec9c1ee72a8605a06800d74385fc8a7273f2334df57",
    "passive_permeability": "1c1b602fe640666c4fb2e006c9712673ecba7ae097bbd7f3e027094762e7c5a2",
    "efflux_transport": "cc24fe33857a2032143c7b6f54d18e7796204b33fa459c2ec94e9af46a0f3db0",
    "influx_transport": "846acb38e8c9adad2b18835aa1ff7516ca043adfd8f402413b2170ebd1387ab4",
}


def source_profiles(data_dir: Path) -> list[NormalizedSourceProfile]:
    return [
        NormalizedSourceProfile(
            source_id="direct_bbb",
            source_name="starling-labs/BBB",
            source_path=str(data_dir / "Direct_BBB" / "records.parquet"),
            source_revision=SOURCE_REVISION,
            endpoint_field="quant_metric",
            measurement_field="quant_value",
            unit_field="quant_units",
            smiles_field="smiles",
            structure_mode="direct",
            record_id_field="source_index",
            name_fields=(),
            context_fields=(
                "pmid",
                "bbb_permeability_label",
                "bbb_transport_label",
                "assay_model",
                "species",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        NormalizedSourceProfile(
            source_id="passive_permeability",
            source_name="Starling/BBB/passive_permeability",
            source_path=str(data_dir / "passive_permeability" / "extractions.parquet"),
            endpoint_field="metric_name",
            measurement_field="metric_value",
            unit_field="metric_units",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("global_identifier",),
            context_fields=(
                "assay_type",
                "biological_system",
                "metric_uncertainty",
                "passive_bbb_interpretation",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
        ),
        NormalizedSourceProfile(
            source_id="efflux_transport",
            source_name="Starling/BBB/efflux_transport",
            source_path=str(data_dir / "efflux_transport" / "extractions.parquet"),
            endpoint_field="quantitative_metric",
            measurement_field="quantitative_value",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("global_identifier",),
            context_fields=(
                "transporter_identifier",
                "evidence_type",
                "interaction_conclusion",
                "assay_system",
                "perturbation",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
        ),
        NormalizedSourceProfile(
            source_id="influx_transport",
            source_name="Starling/BBB/influx_transport",
            source_path=str(data_dir / "influx_transport" / "extractions.parquet"),
            endpoint_field="transport_endpoint",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("global_identifier",),
            context_fields=(
                "mediator_name",
                "mediator_identifier",
                "transport_mechanism",
                "evidence_basis",
                "assay_model",
                "reported_result",
                "qualifying_conditions",
                "extra_details",
                "needs_more_context",
                "pmid",
            ),
        ),
    ]


def validate_source_digest(source_id: str, source_parquet: Path) -> str:
    expected = EXPECTED_SOURCE_SHA256.get(source_id)
    if expected is None:
        raise ValueError(f"no pinned digest for source {source_id!r}")
    actual = file_sha256(source_parquet)
    if actual != expected:
        raise ValueError(
            f"source digest drift for {source_id}: expected {expected}, found {actual}"
        )
    return actual


__all__ = [
    "EXPECTED_SOURCE_ROWS",
    "EXPECTED_SOURCE_SHA256",
    "source_profiles",
    "validate_source_digest",
]
