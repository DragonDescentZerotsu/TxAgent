"""CLI for screening ChEMBL assays relevant to DILI reasoning."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.screen_assays import (
    BASE_OUTPUT_FIELDS,
    ScreenAssaysConfig,
    main as run_screening,
)
from tools.chembl_tool.tasks.dili.report import write_report
from tools.chembl_tool.tasks.dili.scoring import scored_row


OUTPUT_FIELDS = [*BASE_OUTPUT_FIELDS, "keep_for_dili_reasoning"]

CONFIG = ScreenAssaysConfig(
    description=__doc__ or "",
    default_out_dir="outputs/chembl_tool/tasks/dili/assay_screening/raw",
    assay_candidates_csv="dili_assay_candidates.csv",
    assay_candidates_jsonl="dili_assay_candidates.jsonl",
    assay_report_md="dili_assay_report.md",
    activity_evidence_csv="dili_activity_evidence.csv",
    keep_field="keep_for_dili_reasoning",
    scan_stage_message="Scanning assays and applying DILI rules",
    finished_message="Finished DILI assay screening",
    scored_row=scored_row,
    write_report=write_report,
    output_fields=tuple(OUTPUT_FIELDS),
)


def main(argv: list[str] | None = None) -> int:
    return run_screening(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
