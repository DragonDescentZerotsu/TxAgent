"""Re-score existing DILI output files after rule changes."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.rescore_outputs import RescoreConfig, main as run_rescore
from tools.chembl_tool.tasks.dili.report import write_report
from tools.chembl_tool.tasks.dili.scoring import scored_row
from tools.chembl_tool.tasks.dili.screen_assays import OUTPUT_FIELDS
from tools.chembl_tool.tasks.dili.summarize_outputs import summarize_rows


CONFIG = RescoreConfig(
    description=__doc__ or "",
    default_in_dir="outputs/chembl_tool/tasks/dili/assay_screening/raw",
    default_out_dir="outputs/chembl_tool/tasks/dili/assay_screening/v1",
    candidates_filename="dili_assay_candidates.csv",
    candidates_jsonl_filename="dili_assay_candidates.jsonl",
    report_filename="dili_assay_report.md",
    health_check_filename="dili_health_check.md",
    activity_evidence_filename="dili_activity_evidence.csv",
    keep_field="keep_for_dili_reasoning",
    output_fields=tuple(OUTPUT_FIELDS),
    scored_row=scored_row,
    write_report=write_report,
    summarize_rows=summarize_rows,
)


def main(argv: list[str] | None = None) -> int:
    return run_rescore(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
