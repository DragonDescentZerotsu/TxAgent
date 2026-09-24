"""Measure explicit reasoning references and final-card claims in flat runs.

The inputs are one or more matrix roots, their selected condition manifests,
and each run's saved request and ``final_reasoning_output.json``. This reader
uses the request's exact visible-reference index and the shared
``predict.harnesses.reasoning_references`` matcher; it never reads or publishes
raw reasoning text. It writes per-query and grouped TSVs, gaps, a short report,
and a hash-pinned manifest under ``outputs/analysis/prediction``. Pass
``--allow-partial`` for a refreshable partial snapshot; a complete publication
requires complete matrices and every expected final output and is immutable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from predict.harnesses.reasoning_references import match_references
from predict.utils.json import atomic_output_path, sha256_file, write_json_atomic


SCHEMA_VERSION = "flat_reasoning_diagnostics.v1"
REFERENCE_SCHEMA_VERSION = "progressive_reasoning_references.v1"
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs/analysis/prediction/optimization_reasoning_usage_v1"
)
RUN_INDEX_RE = re.compile(r"_idx([0-9]+)$")
MOLECULE_KINDS = {"molecule", "context_conditioned_molecule"}
UNIT_KINDS = MOLECULE_KINDS | {"record", "evidence_group"}

QUERY_FIELDS = [
    "benchmark", "matrix_path", "task", "evaluation_subset", "selection_method",
    "selection_profile", "reranking", "record_pool", "morgan_primary_parent_width",
    "l1_min_contrast", "prompt_version", "model", "reasoning_effort", "query_prior",
    "run_id", "query_index", "visible_molecule_count", "mentioned_molecule_count",
    "molecule_coverage", "visible_record_count", "mentioned_record_count", "record_coverage",
    "matched_reference_occurrences", "unknown_reference_ids", "ambiguous_reference_ids",
    "ambiguous_reference_occurrences", "reasoning_char_count", "reasoning_word_count",
    "reasoning_tokens", "claim_count", "supportive_claim_count", "contradictory_claim_count",
    "other_claim_count", "request_sha256", "output_sha256",
]
SUMMARY_DIMENSIONS = [
    "benchmark", "matrix_path", "task", "evaluation_subset", "selection_method",
    "selection_profile", "reranking", "record_pool", "morgan_primary_parent_width",
    "l1_min_contrast", "prompt_version", "model", "reasoning_effort", "query_prior",
]
SUMMARY_METRICS = [
    "query_count", "expected_query_count", "visible_molecule_count",
    "mentioned_molecule_count", "molecule_coverage", "mean_molecule_coverage",
    "visible_record_count", "mentioned_record_count", "record_coverage",
    "mean_record_coverage", "matched_reference_occurrences",
    "unknown_reference_id_count", "ambiguous_reference_occurrences",
    "mean_reasoning_char_count", "mean_reasoning_word_count", "n_reasoning_token_values",
    "mean_reasoning_tokens", "claim_count", "supportive_claim_count",
    "contradictory_claim_count", "other_claim_count", "mean_claim_count",
]
GAP_FIELDS = [
    "matrix_path", "task", "evaluation_subset", "query_index", "run_id", "stage", "reason",
]


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _add_input(
    inputs: dict[str, str], path: Path, expected_sha256: str | None = None,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required provenance input is missing: {path}")
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"recorded input hash does not match current file: {path}")
    inputs[str(path.resolve())] = digest
    return digest


def _matrix_file(root: Path) -> Path:
    path = root / "matrix.json" if root.is_dir() else root
    if path.name != "matrix.json" or not path.is_file():
        raise FileNotFoundError(f"matrix root must contain matrix.json: {root}")
    return path.resolve()


def _recorded_path(value: str | Path, relative_to: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else relative_to / path).resolve()


def _selection_method(selection: Mapping[str, Any]) -> str:
    if selection.get("mixed_selection"):
        return "mixed"
    direct = bool(selection.get("preselected_contexts"))
    indirect = bool(selection.get("preselected_uids"))
    if direct and indirect:
        return "direct+indirect"
    if direct:
        return "direct"
    if indirect:
        return "indirect"
    return "matrix_default"


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.6f}" if denominator else ""


def _gap(
    gaps: list[dict[str, Any]], matrix_path: Path, task: str, subset: str,
    query_index: Any, run_id: str, stage: str, reason: str,
) -> None:
    gaps.append({
        "matrix_path": str(matrix_path), "task": task, "evaluation_subset": subset,
        "query_index": "" if query_index is None else query_index, "run_id": run_id,
        "stage": stage, "reason": reason,
    })


def _measure_run(
    *, matrix_path: Path, task: str, subset: str, metadata: Mapping[str, Any],
    run_dir: Path, expected_index: int,
    inputs: dict[str, str], gaps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    run_id = run_dir.name
    request_path = run_dir / "request.json"
    output_path = run_dir / "final_reasoning_output.json"
    if not request_path.is_file():
        _gap(
            gaps, matrix_path, task, subset, expected_index, run_id,
            "request", "missing_request",
        )
        return None
    if not output_path.is_file():
        _gap(
            gaps, matrix_path, task, subset, expected_index, run_id,
            "output", "missing_final_output",
        )
        return None
    try:
        request, request_hash = _read_json(request_path)
        output, output_hash = _read_json(output_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        _gap(
            gaps, matrix_path, task, subset, expected_index, run_id,
            "trace", f"invalid_json:{type(error).__name__}",
        )
        return None
    inputs[str(request_path.resolve())] = request_hash
    inputs[str(output_path.resolve())] = output_hash

    contract = request.get("reasoning_reference_contract")
    index = request.get("reasoning_reference_index")
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != REFERENCE_SCHEMA_VERSION
        or contract.get("matching") != "case_insensitive_exact_visible_id"
        or contract.get("required_mentions") is not False
        or not isinstance(contract.get("layout"), str)
        or not isinstance(index, list)
        or any(
            not isinstance(row, dict)
            or not row.get("visible_id")
            or not row.get("stable_id")
            or row.get("unit_kind") not in UNIT_KINDS
            for row in index
        )
    ):
        _gap(
            gaps, matrix_path, task, subset, expected_index, run_id,
            "request", "invalid_reasoning_reference_contract",
        )
        return None

    if output.get("status") != "ok":
        _gap(
            gaps, matrix_path, task, subset, expected_index, run_id,
            "output", f"status_{output.get('status', 'missing')}",
        )
        return None
    llm = output.get("llm")
    content = llm.get("content") if isinstance(llm, dict) else None
    if (
        not isinstance(llm, dict)
        or "reasoning_content" not in llm
        or not isinstance(llm["reasoning_content"], str)
    ):
        _gap(
            gaps, matrix_path, task, subset, expected_index, run_id,
            "output", "missing_private_reasoning",
        )
        return None
    if not isinstance(content, dict) or not isinstance(content.get("claims"), list):
        _gap(
            gaps, matrix_path, task, subset, expected_index, run_id,
            "output", "missing_final_claims",
        )
        return None

    reasoning = llm["reasoning_content"]
    found, occurrences, unknown, ambiguous, ambiguous_occurrences = match_references(
        reasoning, index
    )
    kinds_by_stable: dict[str, set[str]] = defaultdict(set)
    visible: Counter[str] = Counter()
    for row in index:
        stable_id = str(row["stable_id"])
        kind = str(row["unit_kind"])
        kinds_by_stable[stable_id].add(kind)
        if kind in MOLECULE_KINDS:
            visible["molecule"] += 1
        elif kind == "record":
            visible["record"] += 1
    mentioned: Counter[str] = Counter()
    for stable_id in found:
        kinds = kinds_by_stable.get(stable_id, set())
        if kinds & MOLECULE_KINDS:
            mentioned["molecule"] += 1
        if "record" in kinds:
            mentioned["record"] += 1

    claims = content["claims"]
    roles = [
        str(claim.get("evidence_role") or "").casefold()
        for claim in claims if isinstance(claim, dict)
    ]
    supportive = roles.count("supportive")
    contradictory = roles.count("contradictory")
    usage = llm.get("usage") if isinstance(llm.get("usage"), dict) else {}
    reasoning_tokens = _number(usage.get("reasoning_tokens"))
    return {
        **metadata,
        "run_id": run_id,
        "query_index": expected_index,
        "visible_molecule_count": visible["molecule"],
        "mentioned_molecule_count": mentioned["molecule"],
        "molecule_coverage": _ratio(mentioned["molecule"], visible["molecule"]),
        "visible_record_count": visible["record"],
        "mentioned_record_count": mentioned["record"],
        "record_coverage": _ratio(mentioned["record"], visible["record"]),
        "matched_reference_occurrences": occurrences,
        "unknown_reference_ids": json.dumps(unknown, ensure_ascii=False),
        "ambiguous_reference_ids": json.dumps(ambiguous, ensure_ascii=False),
        "ambiguous_reference_occurrences": ambiguous_occurrences,
        "reasoning_char_count": len(reasoning),
        "reasoning_word_count": len(reasoning.split()),
        "reasoning_tokens": "" if reasoning_tokens is None else reasoning_tokens,
        "claim_count": len(claims), "supportive_claim_count": supportive,
        "contradictory_claim_count": contradictory,
        "other_claim_count": len(claims) - supportive - contradictory,
        "request_sha256": request_hash, "output_sha256": output_hash,
    }


def _summarize(
    rows: list[dict[str, Any]], expected_by_group: Mapping[tuple[Any, ...], int],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in SUMMARY_DIMENSIONS)].append(row)
    summary = []
    all_dimensions = set(groups) | set(expected_by_group)
    for dimensions in sorted(all_dimensions, key=lambda item: tuple(map(str, item))):
        group = groups.get(dimensions, [])
        row = dict(zip(SUMMARY_DIMENSIONS, dimensions))
        def total(field: str) -> int:
            return sum(int(value or 0) for value in (item[field] for item in group))
        def mean(field: str) -> str:
            values = [float(item[field]) for item in group if item[field] not in ("", None)]
            return f"{sum(values) / len(values):.6f}" if values else ""
        molecule_denominator = total("visible_molecule_count")
        record_denominator = total("visible_record_count")
        token_values = sum(item["reasoning_tokens"] != "" for item in group)
        metric_values = {
            "query_count": len(group),
            "expected_query_count": expected_by_group.get(dimensions, len(group)),
            "visible_molecule_count": molecule_denominator,
            "mentioned_molecule_count": total("mentioned_molecule_count"),
            "molecule_coverage": _ratio(total("mentioned_molecule_count"), molecule_denominator),
            "mean_molecule_coverage": mean("molecule_coverage"),
            "visible_record_count": record_denominator,
            "mentioned_record_count": total("mentioned_record_count"),
            "record_coverage": _ratio(total("mentioned_record_count"), record_denominator),
            "mean_record_coverage": mean("record_coverage"),
            "matched_reference_occurrences": total("matched_reference_occurrences"),
            "unknown_reference_id_count": sum(
                len(json.loads(item["unknown_reference_ids"])) for item in group
            ),
            "ambiguous_reference_occurrences": total("ambiguous_reference_occurrences"),
            "mean_reasoning_char_count": mean("reasoning_char_count"),
            "mean_reasoning_word_count": mean("reasoning_word_count"),
            "n_reasoning_token_values": token_values,
            "mean_reasoning_tokens": mean("reasoning_tokens"),
            "claim_count": total("claim_count"),
            "supportive_claim_count": total("supportive_claim_count"),
            "contradictory_claim_count": total("contradictory_claim_count"),
            "other_claim_count": total("other_claim_count"),
            "mean_claim_count": mean("claim_count"),
        }
        summary.append({**row, **metric_values})
    return summary


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _report(
    status: str, rows: list[dict[str, Any]], gaps: list[dict[str, Any]], matrices: list[str],
) -> str:
    benchmarks = sorted({str(row["benchmark"]) for row in rows})
    tasks = sorted({str(row["task"]) for row in rows})
    return "\n".join([
        "# Flat reasoning usage diagnostics",
        "",
        f"Status: **{status}**",
        "",
        f"Measured queries: {len(rows)}; diagnostic gaps: {len(gaps)}.",
        f"Benchmarks: {', '.join(benchmarks) if benchmarks else 'none yet'}.",
        f"Tasks: {', '.join(tasks) if tasks else 'none yet'}.",
        "",
        "Reasoning mentions are exact, case-insensitive visible identifiers from each saved request. ",
        "Molecule and record coverage count unique referenced units; occurrence counts retain repeats. ",
        "Reasoning text is measured in memory and is not copied into these outputs. Empty reasoning ",
        "and zero mentions are valid observations. Missing outputs are gaps, never zero-valued rows.",
        "",
        "## Matrix roots",
        "",
        *[f"- `{path}`" for path in matrices],
        "",
        "Tables: `reasoning_usage.per_query.tsv`, `reasoning_usage.summary.tsv`, and ",
        "`diagnostic_gaps.tsv`. The manifest records hashes for consumed traces, matrices, ",
        "selection provenance, condition manifests, and analyzer code.",
        "",
    ])


def build(
    matrix_roots: Iterable[Path], output_root: Path = DEFAULT_OUTPUT_ROOT, *,
    allow_partial: bool = False,
) -> Path:
    """Analyze saved flat traces and publish one partial or complete bundle."""
    roots = [_matrix_file(Path(root)) for root in matrix_roots]
    if not roots:
        raise ValueError("at least one matrix root is required")
    if len(set(roots)) != len(roots):
        raise ValueError("matrix roots must be unique")
    output_root = output_root.resolve()
    output_dir = output_root / ("partial" if allow_partial else "complete")
    if output_dir.exists():
        prior_manifest = output_dir / "diagnostics_manifest.json"
        if not allow_partial:
            raise FileExistsError(f"refusing to overwrite completed analysis: {output_dir}")
        if not prior_manifest.is_file():
            raise FileExistsError(
                f"refusing to overwrite unrecognized output directory: {output_dir}"
            )
        prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if prior.get("status") != "partial":
            raise FileExistsError(f"refusing to overwrite non-partial analysis: {output_dir}")

    inputs: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    expected_by_group: dict[tuple[Any, ...], int] = {}
    matrix_statuses: list[tuple[Path, str]] = []

    for matrix_path in roots:
        matrix, matrix_hash = _read_json(matrix_path)
        inputs[str(matrix_path)] = matrix_hash
        matrix_status = str(matrix.get("status") or "unknown")
        matrix_statuses.append((matrix_path, matrix_status))
        selections = matrix.get("selections")
        if not isinstance(selections, list) or not selections:
            raise ValueError(f"matrix has no selection records: {matrix_path}")
        if matrix_status != "complete":
            _gap(gaps, matrix_path, "", "", "", "", "matrix", f"matrix_status_{matrix_status}")

        for selection in selections:
            if not isinstance(selection, dict) or not selection.get("path"):
                raise ValueError(f"matrix contains an invalid selection record: {matrix_path}")
            selection_path = _recorded_path(selection["path"], matrix_path.parent)
            cache_manifest, cache_hash = _read_json(selection_path)
            selection_sha = selection.get("sha256")
            if selection_sha and cache_hash != selection_sha:
                raise ValueError(f"recorded input hash does not match current file: {selection_path}")
            inputs[str(selection_path)] = cache_hash
            condition_root = selection_path.parent.parent
            condition_path = condition_root / "manifest.json"
            if not condition_path.is_file():
                raise FileNotFoundError(f"condition manifest is missing: {condition_path}")
            condition_manifest, condition_hash = _read_json(condition_path)
            inputs[str(condition_path.resolve())] = condition_hash
            if cache_manifest.get("task") != selection.get("task"):
                raise ValueError(f"selection path task disagrees with matrix: {selection_path}")
            selection_inputs = [
                ("preselected_contexts", None),
                ("preselected_uids", None),
                ("mixed_selection", None),
            ]
            for path_key, hash_key in selection_inputs:
                value = selection.get(path_key)
                if value:
                    _add_input(
                        inputs, _recorded_path(value, matrix_path.parent),
                        str(selection[hash_key]) if hash_key and selection.get(hash_key) else None,
                    )
            if (
                selection.get("selection_contract_sha256")
                and cache_manifest.get("selection_contract_sha256")
                != selection["selection_contract_sha256"]
            ):
                raise ValueError(f"selection contract hash disagrees with matrix: {selection_path}")

            task = str(selection.get("task") or "")
            subset = str(selection.get("evaluation_subset") or "")
            expected_indices = condition_manifest.get("indices")
            if not isinstance(expected_indices, list):
                expected_indices = list(range(int(condition_manifest.get("n_items") or 0)))
            expected_indices = sorted({int(index) for index in expected_indices})
            run_dirs: dict[int, Path] = {}
            for run_dir in sorted((condition_root / "runs").glob("*/")):
                match = RUN_INDEX_RE.search(run_dir.name)
                query_index = int(match.group(1)) if match else None
                run_manifest_path = run_dir / "manifest.json"
                if run_manifest_path.is_file():
                    try:
                        run_manifest, digest = _read_json(run_manifest_path)
                        inputs[str(run_manifest_path.resolve())] = digest
                        if run_manifest.get("query_index") != query_index:
                            _gap(
                                gaps, matrix_path, task, subset, query_index or "",
                                run_dir.name, "run", "run_manifest_index_mismatch",
                            )
                    except (OSError, json.JSONDecodeError, ValueError):
                        _gap(
                            gaps, matrix_path, task, subset, "", run_dir.name,
                            "run", "invalid_run_manifest",
                        )
                else:
                    _gap(
                        gaps, matrix_path, task, subset, "", run_dir.name,
                        "run", "missing_run_manifest",
                    )
                if query_index is None:
                    _gap(
                        gaps, matrix_path, task, subset, "", run_dir.name,
                        "run", "unknown_query_index",
                    )
                elif query_index in run_dirs:
                    _gap(
                        gaps, matrix_path, task, subset, query_index, run_dir.name,
                        "run", "duplicate_query_index",
                    )
                else:
                    run_dirs[query_index] = run_dir

            metadata = {
                "benchmark": matrix.get("benchmark", ""),
                "matrix_path": str(matrix_path), "task": task, "evaluation_subset": subset,
                "selection_method": _selection_method(selection),
                "selection_profile": selection.get("profile", ""),
                "reranking": selection.get("reranking", ""),
                "record_pool": selection.get("record_pool", ""),
                "morgan_primary_parent_width": selection.get("morgan_primary_parent_width", ""),
                "l1_min_contrast": selection.get("l1_min_contrast", ""),
                "prompt_version": cache_manifest.get(
                    "prompt_version", matrix.get("prompt_version", "")
                ),
                "model": condition_manifest.get("model", matrix.get("model", "")),
                "reasoning_effort": matrix.get("reasoning_effort", ""),
                "query_prior": cache_manifest.get("query_prior", ""),
            }
            group_key = tuple(metadata[field] for field in SUMMARY_DIMENSIONS)
            expected_by_group[group_key] = expected_by_group.get(group_key, 0) + len(expected_indices)
            for query_index in expected_indices:
                run_dir = run_dirs.get(query_index)
                if run_dir is None:
                    _gap(
                        gaps, matrix_path, task, subset, query_index, "",
                        "run", "missing_run_directory",
                    )
                    continue
                row = _measure_run(
                    matrix_path=matrix_path, task=task, subset=subset,
                    metadata=metadata, run_dir=run_dir,
                    expected_index=query_index, inputs=inputs, gaps=gaps,
                )
                if row is not None:
                    rows.append(row)
            for unexpected in sorted(set(run_dirs) - set(expected_indices)):
                _gap(
                    gaps, matrix_path, task, subset, unexpected,
                    run_dirs[unexpected].name, "run", "unexpected_query_index",
                )

    if not allow_partial and gaps:
        raise ValueError(
            f"complete analysis requires complete matrices and outputs; found {len(gaps)} diagnostic gaps"
        )
    status = "partial" if allow_partial else "complete"
    summary = _summarize(rows, expected_by_group)
    if allow_partial:
        output_dir.mkdir(parents=True, exist_ok=True)
        publication_dir = output_dir
        staging_dir = None
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=".complete-", dir=output_root))
        publication_dir = staging_dir
    table_paths = {
        "reasoning_usage.per_query.tsv": (rows, QUERY_FIELDS),
        "reasoning_usage.summary.tsv": (summary, SUMMARY_DIMENSIONS + SUMMARY_METRICS),
        "diagnostic_gaps.tsv": (gaps, GAP_FIELDS),
    }
    try:
        for name, (table_rows, fields) in table_paths.items():
            _write_tsv(publication_dir / name, table_rows, fields)
        report = _report(status, rows, gaps, [str(path) for path in roots])
        with atomic_output_path(publication_dir / "report.md") as temporary:
            temporary.write_text(report, encoding="utf-8")
        inputs[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
        shared_code = (
            Path(__file__).resolve().parents[2]
            / "predict/harnesses/reasoning_references.py"
        )
        inputs[str(shared_code)] = sha256_file(shared_code)
        outputs = {
            name: sha256_file(publication_dir / name) for name in table_paths
        }
        outputs["report.md"] = sha256_file(publication_dir / "report.md")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "partial_snapshot": bool(allow_partial),
            "matrices": [
                {"path": str(path), "status": matrix_status}
                for path, matrix_status in matrix_statuses
            ],
            "expected_queries": sum(expected_by_group.values()),
            "measured_queries": len(rows),
            "n_diagnostic_gaps": len(gaps),
            "inputs": dict(sorted(inputs.items())),
            "outputs": outputs,
        }
        write_json_atomic(publication_dir / "diagnostics_manifest.json", manifest)
        if staging_dir is not None:
            os.rename(staging_dir, output_dir)
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root", type=Path, action="append", required=True,
        help="batch root containing matrix.json; repeat for multiple roots",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="write or refresh an explicitly partial snapshot under output-root/partial",
    )
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    allowed_root = (project_root / "outputs/analysis/prediction").resolve()
    try:
        output_relative = args.output_root.resolve().relative_to(allowed_root)
    except ValueError:
        parser.error(f"--output-root must be inside {allowed_root}")
    if len(output_relative.parts) != 1:
        parser.error(
            "--output-root must name one study directory under outputs/analysis/prediction"
        )
    print(build(args.matrix_root, args.output_root, allow_partial=args.allow_partial))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
