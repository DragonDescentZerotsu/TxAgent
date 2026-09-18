"""Count literal ``Molecule N`` mentions in progressive L1 model traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ARTIFACT_VERSION = "progressive_molecule_mention_coverage.v1"
MOLECULE_PATTERN = re.compile(r"\bmolecule\s+(10|[1-9])\b", re.IGNORECASE)
STREAM_FIELDS = {"reasoning": "reasoning_content", "final": "raw_content"}
RETRIEVALS = ("assay-transfer", "morgan")
GROUPS = ["version", "task", "retrieval", "query_prior"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_source(value: str) -> tuple[str, Path]:
    version, separator, raw_path = value.partition("=")
    root = Path(raw_path).resolve()
    if not separator or not version or not (root / "matrix.json").is_file():
        raise ValueError(f"Expected VERSION=PATH to a progressive matrix root: {value}")
    return version, root


def _parse_cell(value: str) -> tuple[str, str]:
    task, separator, prior = value.partition(":")
    if not separator or not task or prior not in {"none", "cached"}:
        raise ValueError(f"Expected TASK:none or TASK:cached: {value}")
    return task, prior


def _trace_key(path: Path, root: Path) -> tuple[str, str, str, int]:
    parts = path.relative_to(root).parts
    if len(parts) != 8 or parts[0] != "conditions" or parts[-1] != "prepared.json":
        raise ValueError(f"Unexpected progressive trace path: {path}")
    condition = parts[1]
    retrieval = next(
        (mode for mode in RETRIEVALS if condition.startswith(f"{mode}_")), None
    )
    if retrieval is None:
        raise ValueError(f"Unsupported retrieval condition: {condition}")
    if condition.endswith("_no_query_prior"):
        prior = "none"
    elif condition.endswith("_query_prior"):
        prior = "cached"
    else:
        raise ValueError(f"Unsupported query-prior condition: {condition}")
    return parts[2], retrieval, prior, int(parts[4].removeprefix("query_idx"))


def _collect(
    sources: list[tuple[str, Path]], cells: set[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    mentions: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int]] = set()

    for version, root in sources:
        source_hash = hashlib.sha256()
        source_counts = Counter()
        unavailable: list[dict[str, Any]] = []
        for prepared_path in sorted(
            root.glob("conditions/*/*/queries/query_idx*/levels/level_1/prepared.json")
        ):
            task, retrieval, prior, query_index = _trace_key(prepared_path, root)
            if (task, prior) not in cells:
                continue
            logical_key = (version, task, retrieval, prior, query_index)
            if logical_key in seen:
                raise ValueError(f"Duplicate logical trace: {logical_key}")
            seen.add(logical_key)

            prepared_bytes = prepared_path.read_bytes()
            if int(json.loads(prepared_bytes).get("n_active_molecules") or 0) != 10:
                raise ValueError(f"Expected 10 visible L1 molecules: {prepared_path}")
            relative = prepared_path.relative_to(root).as_posix()
            source_hash.update(f"{relative}\0{_sha256(prepared_bytes)}\n".encode())

            output_path = prepared_path.with_name("output.json")
            status = "missing"
            output: dict[str, Any] = {}
            if output_path.is_file():
                output_bytes = output_path.read_bytes()
                output = json.loads(output_bytes)
                status = "available" if output.get("status") == "ok" else "non_ok"
                output_relative = output_path.relative_to(root).as_posix()
                source_hash.update(f"{output_relative}\0{_sha256(output_bytes)}\n".encode())
            else:
                source_hash.update(f"{relative[:-13]}output.json\0MISSING\n".encode())

            base = dict(
                version=version,
                task=task,
                retrieval=retrieval,
                query_prior=prior,
                query_index=query_index,
            )
            inventory.append({**base, "status": status})
            source_counts[status] += 1
            if status != "available":
                unavailable.append({**base, "status": status})
                continue

            llm = output.get("llm") or {}
            for stream, field in STREAM_FIELDS.items():
                text = llm.get(field) or ""
                if not isinstance(text, str):
                    raise ValueError(f"llm.{field} must be text or null: {output_path}")
                counts = Counter(int(match) for match in MOLECULE_PATTERN.findall(text))
                for position in range(1, 11):
                    mentions.append(
                        {
                            **base,
                            "stream": stream,
                            "molecule_position": position,
                            "mention_count": counts[position],
                            "blank_stream": not text.strip(),
                        }
                    )

        matrix_path = root / "matrix.json"
        matrix_bytes = matrix_path.read_bytes()
        matrix = json.loads(matrix_bytes)
        source_manifests.append(
            {
                "version": version,
                "root": str(root),
                "matrix_sha256": _sha256(matrix_bytes),
                "prompt": (matrix.get("settings") or {}).get("prompt"),
                "trace_inventory_sha256": source_hash.hexdigest(),
                "n_expected": sum(source_counts.values()),
                "n_available": source_counts["available"],
                "n_missing": source_counts["missing"],
                "n_non_ok": source_counts["non_ok"],
                "unavailable_traces": unavailable,
            }
        )

    if not mentions or not inventory:
        raise ValueError("No matching canonical traces found")
    return pd.DataFrame(mentions), pd.DataFrame(inventory), source_manifests


def _summaries(
    mentions: pd.DataFrame, inventory: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability = (
        inventory.groupby(GROUPS + ["status"]).size().unstack(fill_value=0).reset_index()
    )
    availability["n_expected_traces"] = availability.get("available", 0) + availability.get(
        "missing", 0
    ) + availability.get("non_ok", 0)
    availability = availability.rename(
        columns={
            "available": "n_available_traces",
            "missing": "n_missing_outputs",
            "non_ok": "n_non_ok_outputs",
        }
    )
    for column in ("n_available_traces", "n_missing_outputs", "n_non_ok_outputs"):
        if column not in availability:
            availability[column] = 0

    dimensions = GROUPS + ["stream", "molecule_position"]
    positions = (
        mentions.groupby(dimensions)
        .agg(
            n_blank_streams=("blank_stream", "sum"),
            n_mentions=("mention_count", "sum"),
            n_traces_with_mention=("mention_count", lambda values: (values > 0).sum()),
        )
        .reset_index()
        .merge(availability, on=GROUPS, validate="many_to_one")
    )
    positions["trace_coverage_fraction"] = (
        positions.n_traces_with_mention / positions.n_available_traces
    ).round(6)
    positions["mean_mentions_per_available_trace"] = (
        positions.n_mentions / positions.n_available_traces
    ).round(6)
    position_columns = dimensions + [
        "n_expected_traces",
        "n_available_traces",
        "n_missing_outputs",
        "n_non_ok_outputs",
        "n_blank_streams",
        "n_mentions",
        "n_traces_with_mention",
        "trace_coverage_fraction",
        "mean_mentions_per_available_trace",
    ]

    per_trace = (
        mentions.assign(mentioned=mentions.mention_count.gt(0))
        .groupby(GROUPS + ["query_index", "stream"], as_index=False)
        .mentioned.sum()
        .rename(columns={"mentioned": "n_molecule_positions_mentioned"})
    )
    distribution = (
        per_trace.groupby(GROUPS + ["stream", "n_molecule_positions_mentioned"])
        .size()
        .rename("n_traces")
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_frame(
        per_trace[GROUPS + ["stream"]]
        .drop_duplicates()
        .merge(
            pd.DataFrame({"n_molecule_positions_mentioned": range(11)}), how="cross"
        )
    )
    distribution = (
        distribution.set_index(GROUPS + ["stream", "n_molecule_positions_mentioned"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    denominators = per_trace.groupby(GROUPS + ["stream"]).size().rename("available")
    distribution = distribution.join(denominators, on=GROUPS + ["stream"])
    distribution["trace_fraction"] = (
        distribution.n_traces / distribution.available
    ).round(6)
    return positions[position_columns], distribution.drop(columns="available")


def _pooled_summary(
    positions: pd.DataFrame, distribution: pd.DataFrame, inventory: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (version, retrieval, stream), group in positions.groupby(
        ["version", "retrieval", "stream"], sort=True
    ):
        pooled = group.groupby("molecule_position").agg(
            mentions=("n_mentions", "sum"),
            with_mention=("n_traces_with_mention", "sum"),
        )
        available = len(
            inventory.query(
                "version == @version and retrieval == @retrieval and status == 'available'"
            )
        )
        expected = len(inventory.query("version == @version and retrieval == @retrieval"))
        dist = (
            distribution.query(
                "version == @version and retrieval == @retrieval and stream == @stream"
            )
            .groupby("n_molecule_positions_mentioned")
            .n_traces.sum()
        )
        row: dict[str, Any] = {
            "version": version,
            "retrieval": retrieval,
            "stream": stream,
            "n_expected_traces": expected,
            "n_available_traces": available,
        }
        for position in range(1, 11):
            row[f"molecule_{position}_mentions"] = int(pooled.loc[position, "mentions"])
            row[f"molecule_{position}_coverage_fraction"] = round(
                pooled.loc[position, "with_mention"] / available, 6
            )
        row["all_ten_coverage_fraction"] = round(dist.get(10, 0) / available, 6)
        row["zero_position_coverage_fraction"] = round(dist.get(0, 0) / available, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _report(summary: pd.DataFrame, inventory: pd.DataFrame) -> str:
    headers = [f"M{position}" for position in range(1, 11)]
    coverage_lines = [
        "| Version | Retrieval | Stream | Available / expected | "
        + " | ".join(headers)
        + " | All 10 | Zero |",
        "|---|---|---|---:|" + "|".join("---:" for _ in range(12)) + "|",
    ]
    mention_lines = [
        "| Version | Retrieval | Stream | " + " | ".join(headers) + " |",
        "|---|---|---|" + "|".join("---:" for _ in headers) + "|",
    ]
    pooled_stats: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in summary.to_dict("records"):
        version = row["version"]
        retrieval = row["retrieval"]
        stream = row["stream"]
        available = row["n_available_traces"]
        expected = row["n_expected_traces"]
        prefix = f"| {version} | {retrieval} | {stream}"
        pooled_stats[(version, retrieval, stream)] = (
            row["all_ten_coverage_fraction"],
            row["zero_position_coverage_fraction"],
        )
        coverage_lines.append(
            f"{prefix} | {available} / {expected} | "
            + " | ".join(
                f"{100 * row[f'molecule_{i}_coverage_fraction']:.1f}%"
                for i in range(1, 11)
            )
            + f" | {100 * row['all_ten_coverage_fraction']:.1f}% | "
            f"{100 * row['zero_position_coverage_fraction']:.1f}% |"
        )
        mention_lines.append(
            f"{prefix} | "
            + " | ".join(str(row[f"molecule_{i}_mentions"]) for i in range(1, 11))
            + " |"
        )

    counts = [
        f"{version} {(group.status == 'available').sum()}/{len(group)}"
        for version, group in inventory.groupby("version", sort=True)
    ]
    findings = []
    for retrieval in RETRIEVALS:
        v12 = pooled_stats.get(("v12", retrieval, "reasoning"))
        v14 = pooled_stats.get(("v14", retrieval, "reasoning"))
        if v12 is None or v14 is None:
            continue
        v12_all, v12_zero = v12
        v14_all, v14_zero = v14
        findings.append(
            f"- For {retrieval}, v14 increased all-ten reasoning coverage from "
            f"{100 * v12_all:.1f}% to {100 * v14_all:.1f}% and reduced zero-position "
            f"traces from {100 * v12_zero:.1f}% to {100 * v14_zero:.1f}%."
        )
    final_zero = [
        zero
        for (version, retrieval, stream), (_, zero) in pooled_stats.items()
        if stream == "final"
    ]
    findings.append(
        f"- Final JSON almost never used literal molecule numbers: "
        f"{100 * min(final_zero):.1f}%-{100 * max(final_zero):.1f}% of final streams "
        "mentioned none."
    )
    return "\n".join(
        [
            "# V12-v14 literal molecule-mention coverage",
            "",
            "This audit compares the exact later-v12 predecessor cells shared with v14: BBB with and "
            "without cached query prior, and Bioavailability with cached query prior, for both "
            "assay-transfer and Morgan retrieval.",
            "",
            f"Canonical output coverage is {', '.join(counts)}. Missing or non-OK outputs are reported "
            "but excluded from mention-rate denominators.",
            "",
            "## Key findings",
            "",
            *findings,
            "",
            "## Trace coverage by molecule position",
            "",
            "Each cell is the percentage of available canonical traces containing at least one literal "
            "case-insensitive `Molecule N` mention in that model-authored stream.",
            "",
            *coverage_lines,
            "",
            "## Total literal mentions",
            "",
            *mention_lines,
            "",
            "## Counting contract",
            "",
            "- `reasoning` is `llm.reasoning_content`; `final` is `llm.raw_content`. They are separate; "
            "no union metric is calculated.",
            "- `llm.messages` is excluded because it embeds the prompt labels. Parsed `content` and "
            "`state` are excluded because they duplicate the final response.",
            "- Molecule numbers are query-local panel positions, not stable chemical identities.",
            "- The case-insensitive pattern is `\\bmolecule\\s+(10|[1-9])\\b`.",
            "- The headline table is in `molecule_mention_coverage_summary.tsv`; detailed task/prior "
            "rows are in `mention_coverage_by_position.tsv`; per-trace coverage distributions are in "
            "`trace_coverage_distribution.tsv`.",
            "- `average_molecules_mentioned_per_query.tsv` is the requested single-column reasoning "
            "summary. Its row order is recorded in `manifest.json`.",
            "",
        ]
    )


def run_analysis(
    sources: list[tuple[str, Path]], cells: set[tuple[str, str]], output_dir: Path
) -> dict[str, Any]:
    mentions, inventory, source_manifests = _collect(sources, cells)
    positions, distribution = _summaries(mentions, inventory)
    summary = _pooled_summary(positions, distribution, inventory)
    mean_by_group = (
        distribution[distribution.stream == "reasoning"]
        .assign(
            weighted=lambda frame: frame.n_molecule_positions_mentioned
            * frame.n_traces
        )
        .groupby(["version", "retrieval"], sort=True)
        .agg(weighted=("weighted", "sum"), n_traces=("n_traces", "sum"))
        .reset_index()
    )
    mean_by_group["average_number_of_molecules_mentioned_per_query"] = (
        mean_by_group.weighted / mean_by_group.n_traces
    ).round(6)
    mean_by_group["condition"] = (
        mean_by_group.version + "_" + mean_by_group.retrieval
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "average_molecules_mentioned_per_query.tsv": mean_by_group[
            ["condition", "average_number_of_molecules_mentioned_per_query"]
        ],
        "molecule_mention_coverage_summary.tsv": summary,
        "mention_coverage_by_position.tsv": positions,
        "trace_coverage_distribution.tsv": distribution,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, sep="\t", index=False)
    report_path = output_dir / "REPORT.md"
    report_path.write_text(_report(summary, inventory), encoding="utf-8")

    expected = inventory.groupby("version").size().astype(int).to_dict()
    available_series = inventory[inventory.status == "available"].groupby("version").size()
    available = {version: int(available_series.get(version, 0)) for version in expected}
    analyzer_path = Path(__file__).resolve()
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": {
            "level": 1,
            "visible_molecule_positions": list(range(1, 11)),
            "case_insensitive_regex": MOLECULE_PATTERN.pattern,
            "streams": STREAM_FIELDS,
            "streams_combined": False,
            "excluded_fields": ["llm.messages", "llm.content", "state"],
            "canonical_output_status": "ok",
            "recovery_attempts_included": False,
            "coverage_denominator": "available canonical output.json traces",
            "single_column_row_order": mean_by_group[
                ["version", "retrieval"]
            ].to_dict("records"),
            "included_cells": [
                {"task": task, "query_prior": prior} for task, prior in sorted(cells)
            ],
        },
        "sources": source_manifests,
        "n_expected_by_version": expected,
        "n_available_by_version": available,
        "analyzer": {
            "path": str(analyzer_path),
            "sha256": _sha256(analyzer_path.read_bytes()),
        },
        "output_sha256": {
            name: _sha256((output_dir / name).read_bytes())
            for name in [*tables, "REPORT.md"]
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="VERSION=PATH")
    parser.add_argument("--cell", action="append", required=True, metavar="TASK:PRIOR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = run_analysis(
            [_parse_source(value) for value in args.source],
            {_parse_cell(value) for value in args.cell},
            args.output_dir.resolve(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "expected": manifest["n_expected_by_version"],
                "available": manifest["n_available_by_version"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
