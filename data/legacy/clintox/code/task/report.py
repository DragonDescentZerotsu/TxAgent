"""Markdown report generation for ClinTox assay screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.assay_report import AssayReportConfig, write_report as write_common_report


CONFIG = AssayReportConfig(
    title="ChEMBL ClinTox Assay Screening Report",
    sections=(
        ("Clinical or human safety evidence", "Tier 1"),
        ("In vivo animal toxicology", "Tier 2"),
        ("Organ toxicity and safety pharmacology", "Tier 3"),
        ("Genotoxicity, mutagenicity and carcinogenicity", "Tier 4"),
        ("Tox21/ToxCast and cellular stress pathways", "Tier 5"),
        ("General cytotoxicity and viability", "Tier 6"),
        ("Safety-relevant off-target, DDI and exposure liability", "Tier 7"),
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

