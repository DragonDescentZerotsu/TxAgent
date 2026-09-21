"""Prepare replayed retrievals for tasks supported only by the shared flat harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predict.harnesses.branches.prompt import attach_external_condition
from predict.utils.json import read_jsonl, write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--retrieval-replay-run-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if not args.prepare_only:
        parser.error("flat-only task pipelines execute through the shared scheduler")
    records = read_jsonl(args.input_jsonl)
    record = records[args.query_index]
    source = args.retrieval_replay_run_dir / "retrieval.json"
    retrieval = json.loads(source.read_text(encoding="utf-8"))
    if retrieval.get("status") != "ok":
        raise ValueError(f"Retrieval replay is not successful: {source}")
    expected = str(record[args.smiles_field])
    observed = str((retrieval.get("query") or {}).get("input_smiles") or "")
    if observed and observed != expected:
        raise ValueError(f"Retrieval replay query mismatch: {observed!r} != {expected!r}")
    attach_external_condition(retrieval, record)
    write_json_atomic(args.out_root / args.run_id / "retrieval.json", retrieval)
    return 0
