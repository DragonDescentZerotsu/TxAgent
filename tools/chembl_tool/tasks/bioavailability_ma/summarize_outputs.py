"""Summarize Bioavailability_Ma screening outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.summarize_outputs import SummaryConfig, main as run_summary
from tools.chembl_tool.common.task_workflows.summarize_outputs import summarize_rows as summarize_common_rows


CONFIG = SummaryConfig(
    description=__doc__ or "",
    default_out_dir="outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw",
    candidates_filename="bioavailability_assay_candidates.csv",
    default_report_filename="bioavailability_health_check.md",
    title="Oral Bioavailability Assay Screening Health Check",
    tiers=("Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5", "Tier 6", "none"),
    missing_tier_warnings=(
        ("Tier 1", "no Tier 1 direct oral bioavailability assays retained."),
        ("Tier 5", "no Tier 5 metabolism/clearance assays retained."),
    ),
    negative_flags_note="Some retained candidates have negative flags; inspect whether strong oral bioavailability context justifies keeping them.",
)


def summarize_rows(rows: list[dict[str, Any]], *, candidates_path: Path) -> list[str]:
    return summarize_common_rows(CONFIG, rows, candidates_path=candidates_path)


def main(argv: list[str] | None = None) -> int:
    return run_summary(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
