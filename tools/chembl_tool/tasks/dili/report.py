"""Markdown report generation for DILI assay screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.assay_report import AssayReportConfig, write_report as write_common_report


CONFIG = AssayReportConfig(
    title="ChEMBL DILI Assay Screening Report",
    sections=(
        ("Direct human or clinical DILI anchors", "Tier 1"),
        ("In vivo liver injury phenotype and clinical pathology", "Tier 2"),
        ("Cholestasis and hepatobiliary transporter liability", "Tier 3"),
        ("Mitochondrial, oxidative and organelle stress", "Tier 4"),
        ("Reactive metabolite, bioactivation and immune/idiosyncratic liability", "Tier 5"),
        ("Hepatic cell injury models and exposure/property modifiers", "Tier 6"),
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
