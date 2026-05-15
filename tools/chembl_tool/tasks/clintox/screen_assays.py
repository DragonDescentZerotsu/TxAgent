"""CLI for screening ChEMBL assays relevant to ClinTox reasoning."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.screen_assays import (
    BASE_OUTPUT_FIELDS,
    ScreenAssaysConfig,
    main as run_screening,
)
from tools.chembl_tool.tasks.clintox.report import write_report
from tools.chembl_tool.tasks.clintox.scoring import scored_row


OUTPUT_FIELDS = [*BASE_OUTPUT_FIELDS, "keep_for_clintox_reasoning"]

CONFIG = ScreenAssaysConfig(
    description=__doc__ or "",
    default_out_dir="outputs/chembl_tool/tasks/clintox/assay_screening/raw",
    assay_candidates_csv="clintox_assay_candidates.csv",
    assay_candidates_jsonl="clintox_assay_candidates.jsonl",
    assay_report_md="clintox_assay_report.md",
    activity_evidence_csv="clintox_activity_evidence.csv",
    keep_field="keep_for_clintox_reasoning",
    scan_stage_message="Scanning assays and applying ClinTox rules",
    finished_message="Finished ClinTox assay screening",
    scored_row=scored_row,
    write_report=write_report,
    output_fields=tuple(OUTPUT_FIELDS),
)


def main(argv: list[str] | None = None) -> int:
    return run_screening(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())

