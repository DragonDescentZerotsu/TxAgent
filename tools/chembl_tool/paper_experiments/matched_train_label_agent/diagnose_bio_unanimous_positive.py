"""Run the deterministic Bio unanimous-positive-neighbor trace diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bio_trace_diagnosis import (
    DEFAULT_GOLD_AUDIT,
    build_case_rows,
    summarize_case_rows,
)
from .bio_trace_report import write_outputs
from .contract import DEFAULT_OUTPUT_ROOT


def diagnose(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    gold_audit_path: str | Path = DEFAULT_GOLD_AUDIT,
) -> dict[str, Any]:
    all_rows, source_paths = build_case_rows(
        output_root=output_root, gold_audit_path=gold_audit_path
    )
    summary = summarize_case_rows(all_rows, source_paths)
    cohort_rows = [row for row in all_rows if row["all_three_neighbors_positive"]]
    output = Path(output_root) / "analysis" / "bio_unanimous_positive_diagnosis"
    write_outputs(output, summary, cohort_rows)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gold-audit", type=Path, default=DEFAULT_GOLD_AUDIT)
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(
        output_root=args.output_root, gold_audit_path=args.gold_audit
    ), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
