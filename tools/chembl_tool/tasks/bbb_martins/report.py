"""Markdown report generation for BBB Martins assay screening."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def write_report(
    path: str | Path,
    *,
    total_assays: int,
    candidates: list[dict[str, Any]],
    min_score: int,
    chembl_sqlite: str,
) -> None:
    tier_counts = Counter(row["tier"] for row in candidates)
    lines: list[str] = [
        "# ChEMBL BBB Assay Screening Report",
        "",
        f"- ChEMBL SQLite: `{chembl_sqlite}`",
        f"- Minimum score: `{min_score}`",
        f"- Total assays scanned: `{total_assays}`",
        f"- Candidate assays retained: `{len(candidates)}`",
        "",
        "## Tier Counts",
        "",
    ]
    for tier, count in sorted(tier_counts.items()):
        lines.append(f"- {tier}: {count}")

    sections = [
        ("Direct BBB / brain exposure", "Tier 1"),
        ("Passive permeability / barrier model", "Tier 2"),
        ("Efflux transporter", "Tier 3"),
        ("Influx transporter", "Tier 4"),
    ]
    for title, tier in sections:
        lines.extend(["", f"## Top 50 {title}", ""])
        rows = [row for row in candidates if row["tier"] == tier]
        rows.sort(key=lambda row: int(row["score"]), reverse=True)
        if not rows:
            lines.append("No candidates retained.")
            continue
        lines.append("| score | assay | target | n mols | reason |")
        lines.append("| ---: | --- | --- | ---: | --- |")
        for row in rows[:50]:
            lines.append(
                "| {score} | {assay} | {target} | {n_mols} | {reason} |".format(
                    score=row.get("score", ""),
                    assay=_escape(row.get("assay_chembl_id", "")),
                    target=_escape(row.get("target_pref_name", "")),
                    n_mols=row.get("n_unique_molecules", 0),
                    reason=_escape(row.get("reason", "")),
                )
            )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")
