"""Markdown report generation for Bioavailability_Ma assay screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.assay_report import AssayReportConfig, write_report as write_common_report


CONFIG = AssayReportConfig(
    title="ChEMBL Oral Bioavailability Assay Screening Report",
    sections=(
        ("Direct absolute oral bioavailability", "Tier 1"),
        ("In vivo oral exposure and absorption", "Tier 2"),
        ("In vitro intestinal permeability and efflux", "Tier 3"),
        ("Solubility, dissolution and GI stability", "Tier 4"),
        ("Metabolism, first-pass and clearance", "Tier 5"),
        ("Formulation, food-effect and relative bioavailability context", "Tier 6"),
    ),
)


def write_report(
    path: Path,
    *,
    total_assays: int,
    candidates: list[dict[str, Any]],
    min_score: int,
    chembl_sqlite: str,
) -> None:
    write_common_report(
        CONFIG,
        path,
        total_assays=total_assays,
        candidates=candidates,
        min_score=min_score,
        chembl_sqlite=chembl_sqlite,
    )
