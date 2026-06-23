"""Merge completed reasoning batch artifacts for final-only reruns."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_INDEX_RE = re.compile(r"_idx(\d{5})$")
SOURCE_ARTIFACTS = (
    "retrieval.json",
    "single_molecule_reasoning_output.json",
    "group_reasoning_outputs.jsonl",
)


def merge_source_batches(
    *,
    primary_batch: Path,
    extra_batches: list[Path],
    out_batch: Path,
    indices: list[int] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a final-only source batch from primary and extra completed batches."""
    primary_batch = primary_batch.resolve()
    extra_batches = [path.resolve() for path in extra_batches]
    out_batch = out_batch.resolve()

    _validate_batch(primary_batch)
    for batch in extra_batches:
        _validate_batch(batch)

    if out_batch.exists():
        if not overwrite:
            raise FileExistsError(f"Output batch already exists: {out_batch}")
        shutil.rmtree(out_batch)
    (out_batch / "runs").mkdir(parents=True, exist_ok=True)

    selected_indices = indices if indices is not None else _available_indices(primary_batch)
    if not selected_indices:
        raise ValueError("No query indices selected.")

    for query_index in selected_indices:
        primary_run = _find_run(primary_batch, query_index)
        extra_runs = [_find_run(batch, query_index) for batch in extra_batches]
        out_run = out_batch / "runs" / f"{out_batch.name}_idx{query_index:05d}"
        out_run.mkdir(parents=True, exist_ok=False)
        _merge_one_run(
            query_index=query_index,
            primary_run=primary_run,
            extra_runs=extra_runs,
            out_run=out_run,
        )

    manifest = {
        "type": "merged_final_only_source_batch",
        "primary_source_batch": str(primary_batch),
        "extra_source_batches": [str(path) for path in extra_batches],
        "out_batch": str(out_batch),
        "n_items": len(selected_indices),
        "indices": selected_indices,
        "created_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "paths": {
            "batch_dir": str(out_batch),
            "runs": str(out_batch / "runs"),
        },
    }
    (out_batch / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _merge_one_run(
    *,
    query_index: int,
    primary_run: Path,
    extra_runs: list[Path],
    out_run: Path,
) -> None:
    for source_run in [primary_run, *extra_runs]:
        for name in SOURCE_ARTIFACTS:
            path = source_run / name
            if not path.exists():
                raise FileNotFoundError(f"Missing source artifact for idx{query_index:05d}: {path}")

    primary_retrieval = _read_json(primary_run / "retrieval.json")
    extra_retrievals = [_read_json(run / "retrieval.json") for run in extra_runs]
    merged_retrieval = _merge_retrievals(primary_retrieval, extra_retrievals)
    (out_run / "retrieval.json").write_text(json.dumps(merged_retrieval, indent=2), encoding="utf-8")

    shutil.copy2(
        primary_run / "single_molecule_reasoning_output.json",
        out_run / "single_molecule_reasoning_output.json",
    )

    merged_group_lines = _merge_group_lines(
        [primary_run / "group_reasoning_outputs.jsonl", *[run / "group_reasoning_outputs.jsonl" for run in extra_runs]]
    )
    (out_run / "group_reasoning_outputs.jsonl").write_text("".join(merged_group_lines), encoding="utf-8")

    run_manifest = {
        "type": "merged_final_only_source_run",
        "query_index": query_index,
        "primary_source_run": str(primary_run),
        "extra_source_runs": [str(run) for run in extra_runs],
        "artifacts": {
            "retrieval": str(out_run / "retrieval.json"),
            "single_molecule_reasoning_output": str(out_run / "single_molecule_reasoning_output.json"),
            "group_reasoning_outputs": str(out_run / "group_reasoning_outputs.jsonl"),
        },
    }
    (out_run / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


def _merge_retrievals(primary: dict[str, Any], extras: list[dict[str, Any]]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for retrieval in [primary, *extras]:
        for group in retrieval.get("groups") or []:
            group_id = str(group.get("group_id") or "")
            if group_id in seen_group_ids:
                raise ValueError(f"Duplicate group_id across source batches: {group_id}")
            seen_group_ids.add(group_id)
            groups.append(group)

    coverage = {
        "n_groups": len(groups),
        "n_groups_with_neighbors": sum(1 for group in groups if group.get("neighbors")),
        "n_neighbors_total": sum(len(group.get("neighbors") or []) for group in groups),
    }
    min_similarities = [
        (retrieval.get("coverage") or {}).get("min_similarity")
        for retrieval in [primary, *extras]
        if (retrieval.get("coverage") or {}).get("min_similarity") is not None
    ]
    top_ks = [
        (retrieval.get("coverage") or {}).get("top_k_per_group")
        for retrieval in [primary, *extras]
        if (retrieval.get("coverage") or {}).get("top_k_per_group") is not None
    ]
    if min_similarities:
        coverage["min_similarity"] = min(min_similarities)
    if top_ks:
        coverage["top_k_per_group"] = max(top_ks)

    return {
        **primary,
        "groups": groups,
        "coverage": coverage,
        "evidence_source": {
            "type": "merged_final_only_artifacts",
            "sources": [primary.get("evidence_source") or {}, *[(extra.get("evidence_source") or {}) for extra in extras]],
        },
    }


def _merge_group_lines(paths: list[Path]) -> list[str]:
    merged: list[str] = []
    seen_group_ids: set[str] = set()
    for path in paths:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            group_id = str(record.get("group_id") or "")
            if group_id in seen_group_ids:
                raise ValueError(f"Duplicate group_id across group outputs: {group_id}")
            seen_group_ids.add(group_id)
            merged.append(json.dumps(record, ensure_ascii=False) + "\n")
    return merged


def _validate_batch(batch_dir: Path) -> None:
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")
    if not (batch_dir / "runs").is_dir():
        raise FileNotFoundError(f"Batch runs directory does not exist: {batch_dir / 'runs'}")


def _available_indices(batch_dir: Path) -> list[int]:
    indices = []
    for run_dir in (batch_dir / "runs").iterdir():
        if not run_dir.is_dir():
            continue
        match = RUN_INDEX_RE.search(run_dir.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _find_run(batch_dir: Path, query_index: int) -> Path:
    matches = sorted((batch_dir / "runs").glob(f"*_idx{query_index:05d}"))
    if not matches:
        raise FileNotFoundError(f"No source run for query index {query_index}: {batch_dir / 'runs'}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous source runs for query index {query_index}: {matches}")
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-batch", required=True, help="Completed batch providing base retrieval/single/group artifacts.")
    parser.add_argument(
        "--extra-batch",
        action="append",
        required=True,
        help="Completed batch whose retrieval groups and group outputs should be appended. Repeatable.",
    )
    parser.add_argument("--out-batch", required=True, help="Output merged source batch directory.")
    parser.add_argument("--indices", type=int, nargs="*", help="Optional subset of query indices to merge.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output batch directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = merge_source_batches(
        primary_batch=Path(args.primary_batch),
        extra_batches=[Path(path) for path in args.extra_batch],
        out_batch=Path(args.out_batch),
        indices=args.indices,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
