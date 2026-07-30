"""Pinned source contracts for the layered Bioavailability normalizer."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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


def direct_hf_profile(
    records_path: Path,
    dropped_path: Path,
) -> NormalizedSourceProfile:
    return NormalizedSourceProfile(
        source_id="direct_hf",
        source_name=ORAL_BIOAVAILABILITY_DATASET,
        source_revision=ORAL_BIOAVAILABILITY_REVISION,
        source_path=f"{records_path};{dropped_path}",
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
    records_jsonl: Path,
    dropped_jsonl: Path,
    *,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    """Reconstruct all pinned HF rows from its kept+dropped prepared partition."""
    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for payload in _read_jsonl(records_jsonl):
        source_index = int(payload["source_index"])
        raw = dict(payload.get("metadata") or {})
        raw["source_index"] = source_index
        raw.setdefault("smiles", payload.get("smiles"))
        raw.setdefault("molecule_name", payload.get("molecule_name"))
        _append_direct_row(rows, seen_indices, raw)
        if max_rows and len(rows) >= max_rows:
            return rows
    for payload in _read_jsonl(dropped_jsonl):
        source_index = int(payload["source_index"])
        raw = dict(payload.get("raw_row") or {})
        raw["source_index"] = source_index
        raw["_prepared_drop_reason"] = payload.get("drop_reason")
        _append_direct_row(rows, seen_indices, raw)
        if max_rows and len(rows) >= max_rows:
            return rows
    return rows


def _append_direct_row(
    rows: list[dict[str, Any]],
    seen_indices: set[int],
    row: dict[str, Any],
) -> None:
    source_index = int(row["source_index"])
    if source_index in seen_indices:
        raise ValueError(f"duplicate direct HF source_index: {source_index}")
    seen_indices.add(source_index)
    rows.append(row)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


__all__ = [
    "EXPECTED_SOURCE_ROWS",
    "direct_hf_profile",
    "load_direct_hf_rows",
    "source_profiles",
]
