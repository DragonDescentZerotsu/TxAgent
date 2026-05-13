"""CLI for screening ChEMBL assays relevant to BBB reasoning."""

from __future__ import annotations

from tools.chembl_tool.tasks.bbb_martins.report import write_report
from tools.chembl_tool.tasks.bbb_martins.scoring import scored_row
from tools.chembl_tool.common.task_workflows.screen_assays import (
    BASE_OUTPUT_FIELDS,
    ScreenAssaysConfig,
    export_activity_evidence,
    main as run_screening,
)


OUTPUT_FIELDS = [*BASE_OUTPUT_FIELDS, "keep_for_bbb_reasoning"]

CONFIG = ScreenAssaysConfig(
    description=__doc__ or "",
    default_out_dir="outputs/chembl_tool/tasks/bbb_martins/assay_screening/raw",
    assay_candidates_csv="bbb_assay_candidates.csv",
    assay_candidates_jsonl="bbb_assay_candidates.jsonl",
    assay_report_md="bbb_assay_report.md",
    activity_evidence_csv="bbb_activity_evidence.csv",
    keep_field="keep_for_bbb_reasoning",
    scan_stage_message="Scanning assays and applying BBB rules",
    finished_message="Finished BBB assay screening",
    scored_row=scored_row,
    write_report=write_report,
    output_fields=tuple(OUTPUT_FIELDS),
)


def main(argv: list[str] | None = None) -> int:
    return run_screening(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
