"""Shared Markdown report generation for ChEMBL assay screening tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssayReportConfig:
    title: str
    sections: tuple[tuple[str, str], ...]


def write_report(
    config: AssayReportConfig,
    path: Path,
    *,
    total_assays: int,
    candidates: list[dict[str, Any]],
    min_score: int,
    chembl_sqlite: str,
) -> None:
    tier_counts = Counter(row["tier"] for row in candidates)
    lines: list[str] = [
        f"# {config.title}",
        "",
        f"- ChEMBL SQLite: `{chembl_sqlite}`",
        f"- Minimum score: `{min_score}`",
        f"- Total assays scanned: `{total_assays:,}`",
        f"- Candidate assays retained: `{len(candidates):,}`",
        "",
        "## Tier Counts",
        "",
    ]
    for tier, count in sorted(tier_counts.items()):
        lines.append(f"- {tier}: {count}")

    for title, tier in config.sections:
        lines.extend(["", f"## Top 50 {title}", ""])
        tier_rows = [row for row in candidates if row.get("tier") == tier]
        tier_rows.sort(key=lambda row: int(row.get("score") or 0), reverse=True)
        if not tier_rows:
            lines.append("No candidates retained.")
            continue
        lines.append("| score | assay | target | n mols | reason |")
        lines.append("| ---: | --- | --- | ---: | --- |")
        for row in tier_rows[:50]:
            lines.append(
                "| {score} | {assay} | {target} | {n_mols} | {reason} |".format(
                    score=_escape(row.get("score", "")),
                    assay=_escape(row.get("assay_chembl_id", "")),
                    target=_escape(row.get("target_pref_name", "")),
                    n_mols=_escape(row.get("n_unique_molecules", "")),
                    reason=_escape(row.get("reason", "")),
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
