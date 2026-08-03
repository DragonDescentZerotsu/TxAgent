"""Pinned source contracts for the layered Bioavailability normalizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tools.chembl_tool.common.starling.normalization.contracts import NormalizedSourceProfile
from tools.chembl_tool.common.starling.oral_bioavailability import (
    ORAL_BIOAVAILABILITY_DATASET,
    ORAL_BIOAVAILABILITY_REVISION,
)


EXPECTED_SOURCE_ROWS = {
    "oral_exposure": 119_192,
    "fa": 85_061,
    "fg": 27_713,
    "fh": 67_943,
    "direct_hf": 163_815,
}

DEFAULT_DIRECT_HF_PARQUET = Path(
    "data/starling_data/bioavailability_ma/Direct_HF/records.parquet"
)
DIRECT_HF_SOURCE_COLUMNS = (
    "source_index",
    "pmid",
    "molecule_name",
    "smiles",
    "support_text",
    "oral_bioavailability_value",
    "bioavailability_report_type",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
)


def source_profiles(data_dir: Path) -> list[NormalizedSourceProfile]:
    return [
        NormalizedSourceProfile(
            source_id="oral_exposure",
            source_name="starling-labs/bioavailability_ma/Oral_AUC-Cmax-Exposure",
            source_path=str(data_dir / "Oral_AUC-Cmax_Exposure" / "extractions.parquet"),
            endpoint_field="exposure_measure",
            measurement_field="parameter_value",
            unit_field="parameter_units",
            context_fields=(
                "statistic_type",
                "oral_dose",
                "study_context",
                "comparator_exposure",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        NormalizedSourceProfile(
            source_id="fa",
            source_name="starling-labs/bioavailability_ma/Fa",
            source_path=str(data_dir / "Fa" / "extractions.parquet"),
            endpoint_field="endpoint_category",
            measurement_field="reported_value",
            unit_field="reported_units",
            context_fields=(
                "assay_system",
                "condition_medium",
                "biological_context",
                "formulation_or_solid_form",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        NormalizedSourceProfile(
            source_id="fg",
            source_name="starling-labs/bioavailability_ma/Fg",
            source_path=str(data_dir / "Fg" / "extractions.parquet"),
            endpoint_field="gut_wall_process",
            measurement_field="measured_value",
            embedded_unit=True,
            name_fields=("molecule_name", "global_identifier"),
            context_fields=(
                "transporter_or_enzyme",
                "substrate_status",
                "assay_system",
                "intestinal_site",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        NormalizedSourceProfile(
            source_id="fh",
            source_name="starling-labs/bioavailability_ma/Fh",
            source_path=str(data_dir / "Fh" / "extractions.parquet"),
            endpoint_field="metric_type",
            measurement_field="reported_value",
            unit_field="reported_units",
            context_fields=(
                "assay_system",
                "species",
                "molecular_form",
                "enzyme_or_pathway",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
    ]


def direct_hf_profile(records_path: Path) -> NormalizedSourceProfile:
    return NormalizedSourceProfile(
        source_id="direct_hf",
        source_name=ORAL_BIOAVAILABILITY_DATASET,
        source_revision=ORAL_BIOAVAILABILITY_REVISION,
        source_path=str(records_path),
        endpoint_constant="oral_bioavailability",
        measurement_field="oral_bioavailability_value",
        unit_constant="%",
        structure_mode="direct",
        record_id_field="source_index",
        confidence_field="confidence",
        name_fields=("molecule_name",),
        context_fields=(
            "bioavailability_report_type",
            "species_or_population",
            "dose",
            "oral_exposure_mode",
            "qualifying_conditions",
            "comparator",
            "extra_details",
        ),
    )


def load_direct_hf_rows(
    records_parquet: Path,
    *,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    """Load the complete pinned HF source without historical filter partitions."""
    frame = pd.read_parquet(records_parquet)
    if tuple(frame.columns) != DIRECT_HF_SOURCE_COLUMNS:
        raise ValueError(
            "Direct HF source-column contract mismatch: "
            f"expected={DIRECT_HF_SOURCE_COLUMNS!r}, found={tuple(frame.columns)!r}"
        )
    if frame["source_index"].isna().any() or not frame["source_index"].is_unique:
        raise ValueError("Direct HF source_index must be complete and unique")
    frame = frame.sort_values("source_index", kind="stable")
    if not max_rows:
        expected = list(range(EXPECTED_SOURCE_ROWS["direct_hf"]))
        if frame["source_index"].astype(int).tolist() != expected:
            raise ValueError("Direct HF source_index coverage must be exactly 0..163814")
    elif max_rows:
        frame = frame.head(max_rows)
    frame = frame.astype(object).where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


__all__ = [
    "DEFAULT_DIRECT_HF_PARQUET",
    "DIRECT_HF_SOURCE_COLUMNS",
    "EXPECTED_SOURCE_ROWS",
    "direct_hf_profile",
    "load_direct_hf_rows",
    "source_profiles",
]
