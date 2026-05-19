"""Markdown report generation for Skin_Reaction assay screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.assay_report import AssayReportConfig, write_report as write_common_report


CONFIG = AssayReportConfig(
    title="ChEMBL Skin_Reaction Assay Screening Report",
    sections=(
        ("Direct skin reaction anchors", "Tier 1"),
        ("Skin sensitisation AOP key-event assays", "Tier 2"),
        ("Phototoxicity, irritation, corrosion and local skin damage", "Tier 3"),
        ("Skin exposure and barrier penetration modifiers", "Tier 4"),
        ("Weak or context-dependent skin background", "Tier 5"),
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
