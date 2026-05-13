"""Markdown report generation for BBB Martins assay screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.assay_report import AssayReportConfig, write_report as write_common_report


CONFIG = AssayReportConfig(
    title="ChEMBL BBB Assay Screening Report",
    sections=(
        ("Direct BBB / brain exposure", "Tier 1"),
        ("Passive permeability / barrier model", "Tier 2"),
        ("Efflux transporter", "Tier 3"),
        ("Influx transporter", "Tier 4"),
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
