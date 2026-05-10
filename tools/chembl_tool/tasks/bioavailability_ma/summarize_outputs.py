"""Summarize Bioavailability_Ma screening outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    candidates_path = out_dir / "bioavailability_assay_candidates.csv"
    report_path = Path(args.report_path) if args.report_path else out_dir / "bioavailability_health_check.md"
    rows = _read_rows(candidates_path)
    lines = summarize_rows(rows, candidates_path=candidates_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def summarize_rows(rows: list[dict[str, Any]], *, candidates_path: Path) -> list[str]:
    tier_counts = Counter(row.get("tier", "") for row in rows)
    keyword_counts = Counter()
    negative_counts = Counter()
    endpoint_counts = Counter()
    for row in rows:
        keyword_counts.update(_split_multi(row.get("matched_keywords", "")))
        endpoint_counts.update(_split_multi(row.get("matched_endpoints", "")))
        negative_counts.update(_split_multi(row.get("negative_flags", "")))

    total = len(rows)
    lines = [
        "# Oral Bioavailability Assay Screening Health Check",
        "",
        f"- Candidates file: `{candidates_path}`",
        f"- Retained candidate assays: `{total:,}`",
        "",
        "## Tier Distribution",
        "",
    ]
    if total:
        for tier in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5", "Tier 6", "none"]:
            count = tier_counts.get(tier, 0)
            if count:
                lines.append(f"- {tier}: {count:,} ({100.0 * count / total:.2f}%)")
    else:
        lines.append("- No candidates retained.")

    lines.extend(["", "## Top Matched Keywords", ""])
    lines.extend(_counter_lines(keyword_counts, total=20))
    lines.extend(["", "## Top Matched Endpoints", ""])
    lines.extend(_counter_lines(endpoint_counts, total=20))
    lines.extend(["", "## Negative Flags", ""])
    lines.extend(_counter_lines(negative_counts, total=20))

    for tier in ["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5", "Tier 6"]:
        lines.extend(["", f"## Top Examples: {tier}", ""])
        tier_rows = [row for row in rows if row.get("tier") == tier]
        tier_rows.sort(key=lambda row: _as_int(row.get("score")), reverse=True)
        if not tier_rows:
            lines.append("No examples.")
            continue
        lines.append("| score | assay | target | n mols | matched | description |")
        lines.append("| ---: | --- | --- | ---: | --- | --- |")
        for row in tier_rows[:10]:
            matched = row.get("matched_endpoints") or row.get("matched_keywords") or row.get("matched_targets") or ""
            lines.append(
                "| {score} | {assay} | {target} | {n_mols} | {matched} | {description} |".format(
                    score=_escape(row.get("score", "")),
                    assay=_escape(row.get("assay_chembl_id", "")),
                    target=_escape(row.get("target_pref_name", "")),
                    n_mols=_escape(row.get("n_unique_molecules", "")),
                    matched=_escape(matched),
                    description=_escape(str(row.get("description", ""))[:180]),
                )
            )

    lines.extend(["", "## Health Notes", ""])
    retained_rate_note = "Review retained rate against total assays from the run log."
    lines.append(f"- {retained_rate_note}")
    if tier_counts.get("Tier 1", 0) == 0:
        lines.append("- Warning: no Tier 1 direct oral bioavailability assays retained.")
    if tier_counts.get("Tier 5", 0) == 0:
        lines.append("- Warning: no Tier 5 metabolism/clearance assays retained.")
    if negative_counts:
        lines.append("- Some retained candidates have negative flags; inspect whether strong oral bioavailability context justifies keeping them.")
    return lines


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_multi(value: object) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _counter_lines(counter: Counter[str], total: int) -> list[str]:
    if not counter:
        return ["- None."]
    return [f"- {key}: {value:,}" for key, value in counter.most_common(total)]


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _escape(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/raw")
    parser.add_argument("--report-path", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
