"""Pinned source contracts for the BBB Martins normalized-v6 library."""

from __future__ import annotations

from pathlib import Path

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.normalization.contracts import NormalizedSourceProfile
from data.processing.evidence_library.versions.v10.tasks.bbb_martins.starling_benchmark import SOURCE_REVISION


EXPECTED_SOURCE_ROWS = {
    "direct_bbb": 304_845,
    "passive_permeability": 10_119,
    "efflux_transport": 191_262,
    "influx_transport": 75_720,
}

EXPECTED_SOURCE_SHA256 = {
    "direct_bbb": "7fcb0628df2ea67c3e050e2151248ebc059cfa678f8174af196ac24835ce7db4",
    "passive_permeability": "7efb1118dd6a8e573fa451ee501d9fb5e7b702c7ee8670cab035ee4ec09aacb3",
    "efflux_transport": "9adcfa03a13aee25ddca3be610fa006620962a3237fb00b8848c828e54aa25dd",
    "influx_transport": "b57c1ccdf99c1f86ccf0c154f68a139442354962590bf56f70395c71c7572199",
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
            # Influx reports its quantity in ``reported_result`` and has no unit
            # column, so the value carries its own unit inline.  Before this was
            # declared, the source had no measurement role at all and every
            # influx row resolved to a non-scalar.
            measurement_field="reported_result",
            embedded_unit=True,
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("global_identifier",),
            context_fields=(
                "mediator_name",
                "mediator_identifier",
                "transport_mechanism",
                "evidence_basis",
                "assay_model",
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
