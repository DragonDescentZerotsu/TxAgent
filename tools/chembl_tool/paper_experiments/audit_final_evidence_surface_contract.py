"""Audit matched inputs and identity safety for final-evidence-surface runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.identity_blind import find_identity_blind_leaks
from tools.chembl_tool.common.json_utils import (
    canonical_json_bytes,
    sha256_file,
    write_json_atomic,
)

from .run_final_evidence_surface_experiment import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_RUN_ROOT,
    EXPECTED_TASK_ROWS,
    SURFACES,
    TASKS,
)


MATCHED_ARTIFACTS = (
    "retrieval.json",
    "single_molecule_reasoning_output.json",
    "group_reasoning_outputs.jsonl",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for surface in args.surfaces:
        for task in args.tasks:
            batch = (
                Path(args.output_root)
                / "runs_identity_blind_parent_disjoint"
                / surface
                / task
                / f"{task}__starling_full_flat__{surface}"
            )
            source_batch = (
                Path(args.source_run_root) / task / f"{task}__starling_full_flat"
            )
            condition_rows, condition_failures = _audit_condition(
                task,
                surface,
                batch,
                source_batch,
            )
            rows.extend(condition_rows)
            failures.extend(condition_failures)

    summaries = _summarize(rows, failures)
    output_dir = Path(args.output_root) / "contract_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "starling.final_evidence_surface_contract_audit.v1",
        "matched_artifacts": list(MATCHED_ARTIFACTS),
        "conditions": summaries,
        "n_rows_checked": len(rows),
        "n_failures": len(failures),
        "failures": failures[:100],
        "failure_output_truncated": len(failures) > 100,
    }
    write_json_atomic(output_dir / "audit.json", payload)
    _write_report(output_dir / "report.md", payload)
    print(json.dumps({"audit_dir": str(output_dir), "n_failures": len(failures)}, indent=2))
    return 1 if failures else 0


def _audit_condition(
    task: str,
    surface: str,
    batch: Path,
    source_batch: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not batch.exists():
        return [], [{"task": task, "surface": surface, "error": "missing_batch"}]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    target_runs = sorted((batch / "runs").glob("*_idx*"))
    if len(target_runs) != EXPECTED_TASK_ROWS[task]:
        failures.append(
            {
                "task": task,
                "surface": surface,
                "error": "incomplete_target_runs",
                "n_runs": len(target_runs),
                "expected_runs": EXPECTED_TASK_ROWS[task],
            }
        )
    for target_run in target_runs:
        index = _query_index(target_run)
        source_matches = sorted((source_batch / "runs").glob(f"*_idx{index:05d}"))
        if len(source_matches) != 1:
            failures.append(
                {
                    "task": task,
                    "surface": surface,
                    "query_index": index,
                    "error": "source_run_resolution",
                    "n_matches": len(source_matches),
                }
            )
            continue
        row, row_failures = _audit_run(
            task,
            surface,
            index,
            target_run,
            source_matches[0],
        )
        rows.append(row)
        failures.extend(row_failures)
    return rows, failures


def _audit_run(
    task: str,
    surface: str,
    index: int,
    target_run: Path,
    source_run: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {"task": task, "surface": surface, "query_index": index}
    failures: list[dict[str, Any]] = []
    artifact_match: dict[str, bool] = {}
    for name in MATCHED_ARTIFACTS:
        target = target_run / name
        source = source_run / name
        matches = (
            target.exists()
            and source.exists()
            and sha256_file(target) == sha256_file(source)
        )
        artifact_match[name] = matches
        if not matches:
            failures.append({**base, "error": "artifact_hash_mismatch", "artifact": name})

    final_path = target_run / "final_reasoning_output.json"
    trace_path = target_run / "trace_messages.jsonl"
    final = _read_json(final_path) if final_path.exists() else {}
    final_ok = final.get("status") == "ok"
    if not final_ok:
        failures.append({**base, "error": "final_not_ok"})

    card_audit = final.get("final_evidence_surface") or {}
    surface_matches = card_audit.get("surface") == surface
    if not surface_matches:
        failures.append({**base, "error": "surface_audit_mismatch"})

    prompt_payload = _final_user_payload(trace_path)
    card_payload = prompt_payload.get("final_evidence_cards") or {}
    cards = card_payload.get("cards") or []
    observed_card_hash = hashlib.sha256(canonical_json_bytes(cards)).hexdigest()
    card_hash_matches = (
        "final_evidence_cards" in prompt_payload
        and len(cards) == int(card_audit.get("n_cards") or 0)
        and observed_card_hash == card_audit.get("cards_sha256")
    )
    if not card_hash_matches:
        failures.append({**base, "error": "card_hash_mismatch"})

    source_retrieval = _read_json(source_run / "retrieval.json")
    leaks = find_identity_blind_leaks(source_retrieval, prompt_payload)
    n_leaks = sum(len(values) for values in leaks.values())
    if n_leaks:
        failures.append({**base, "error": "identity_leak", "leaks": leaks})

    return (
        {
            **base,
            "artifact_match": artifact_match,
            "final_ok": final_ok,
            "surface_matches": surface_matches,
            "card_hash_matches": card_hash_matches,
            "n_cards": len(cards),
            "cards_bytes": card_audit.get("cards_bytes"),
            "n_identity_leaks": n_leaks,
        },
        failures,
    )


def _final_user_payload(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        return {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") != "final_summary":
            continue
        messages = list(((row.get("raw_output") or {}).get("llm") or {}).get("messages") or [])
        if not messages:
            messages = list(row.get("messages") or [])
        # Validation retries append another user correction message.  Search
        # every user message for the original structured final payload instead
        # of assuming that the last user message is the evidence prompt.
        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, dict):
                if "final_evidence_cards" in content:
                    return content
                continue
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "final_evidence_cards" in parsed:
                    return parsed
    return {}


def _summarize(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["task"], row["surface"]) for row in rows})
    for task, surface in keys:
        selected = [row for row in rows if row["task"] == task and row["surface"] == surface]
        condition_failures = [
            failure
            for failure in failures
            if failure.get("task") == task and failure.get("surface") == surface
        ]
        summaries.append(
            {
                "task": task,
                "surface": surface,
                "n_runs_checked": len(selected),
                "n_failures": len(condition_failures),
                "all_matched": not condition_failures,
                "max_cards": max((int(row["n_cards"]) for row in selected), default=0),
                "max_cards_bytes": max(
                    (int(row["cards_bytes"] or 0) for row in selected),
                    default=0,
                ),
            }
        )
    return summaries


def _query_index(run_dir: Path) -> int:
    marker = run_dir.name.rsplit("_idx", 1)[-1]
    return int(marker)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Final evidence surface contract audit",
        "",
        f"- checked runs: {payload['n_rows_checked']}",
        f"- failures: {payload['n_failures']}",
        "",
        "| task | surface | runs | failures | all matched | max cards | max bytes |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in payload["conditions"]:
        lines.append(
            f"| {row['task']} | {row['surface']} | {row['n_runs_checked']} | "
            f"{row['n_failures']} | {row['all_matched']} | {row['max_cards']} | "
            f"{row['max_cards_bytes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--surfaces", nargs="+", choices=SURFACES, default=list(SURFACES))
    parser.add_argument("--source-run-root", default=str(DEFAULT_SOURCE_RUN_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
