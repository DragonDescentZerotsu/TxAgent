"""Re-score existing Bioavailability Ma output files after rule changes."""

from __future__ import annotations

from tools.chembl_tool.tasks.bioavailability_ma.report import write_report
from tools.chembl_tool.tasks.bioavailability_ma.scoring import scored_row
from tools.chembl_tool.tasks.bioavailability_ma.screen_assays import OUTPUT_FIELDS
from tools.chembl_tool.tasks.bioavailability_ma.summarize_outputs import summarize_rows
from tools.chembl_tool.common.task_workflows.rescore_outputs import RescoreConfig, main as run_rescore


CONFIG = RescoreConfig(
    description=__doc__ or "",
    default_in_dir="outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw",
    default_out_dir="outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v1",
    candidates_filename="bioavailability_assay_candidates.csv",
    candidates_jsonl_filename="bioavailability_assay_candidates.jsonl",
    report_filename="bioavailability_assay_report.md",
    health_check_filename="bioavailability_health_check.md",
    activity_evidence_filename="bioavailability_activity_evidence.csv",
    keep_field="keep_for_bioavailability_reasoning",
    output_fields=tuple(OUTPUT_FIELDS),
    scored_row=scored_row,
    write_report=write_report,
    summarize_rows=summarize_rows,
)


def main(argv: list[str] | None = None) -> int:
    return run_rescore(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
