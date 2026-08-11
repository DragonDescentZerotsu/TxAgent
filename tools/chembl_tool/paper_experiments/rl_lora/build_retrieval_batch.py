"""Build replayable train-only full-flat retrieval with mixed identity policies."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.task_workflows.retrieve_neighbors import load_index

from .mixed_retrieval import (
    MIXED_IDENTITY_POLICY_VERSION,
    retrieve_full_flat_mixed_identity,
)


TASK_CONFIG_MODULES = {
    "bbb_martins": "tools.chembl_tool.tasks.bbb_martins.experiment_config",
    "bioavailability_ma": "tools.chembl_tool.tasks.bioavailability_ma.experiment_config",
    "skin_reaction": "tools.chembl_tool.tasks.skin_reaction.experiment_config",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_retrieval_batch(
    *,
    task: str,
    input_jsonl: Path,
    index_path: Path,
    output_batch: Path,
    indices: list[int],
    top_k_per_group: int,
    min_similarity: float,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    if not indices:
        raise ValueError("at least one query index is required")
    invalid = sorted(index for index in set(indices) if index < 0 or index >= len(rows))
    if invalid:
        raise IndexError(f"query indices outside [0, {len(rows)}): {invalid}")

    module = importlib.import_module(TASK_CONFIG_MODULES[task])
    config = module.get_source_config("starling")
    index = load_index(index_path)
    batch_id = output_batch.name
    receipts = []
    for query_index in sorted(set(indices)):
        query_smiles = str(rows[query_index]["drug"])
        retrieval = retrieve_full_flat_mixed_identity(
            query_smiles,
            index,
            config=config,
            top_k_per_group=top_k_per_group,
            min_similarity=min_similarity,
        )
        if retrieval.get("status") != "ok":
            raise ValueError(f"retrieval failed for query index {query_index}")
        run_dir = output_batch / "runs" / f"{batch_id}_idx{query_index:05d}"
        write_json_atomic(run_dir / "retrieval.json", retrieval)
        family_policy = retrieval["experiment"]["family_identity_policy"]
        receipts.append(
            {
                "query_index": query_index,
                "label": int(rows[query_index]["Y"]),
                "n_neighbors": int(retrieval["coverage"]["n_neighbors_total"]),
                "group_policies": family_policy["group_policies"],
                "retrieval": str(run_dir / "retrieval.json"),
            }
        )

    manifest = {
        "schema_version": "rl_lora_mixed_retrieval_batch.v1",
        "task": task,
        "formal_eligible": False,
        "purpose": "bounded local/Tinker feasibility and throughput test",
        "identity_policy": MIXED_IDENTITY_POLICY_VERSION,
        "input_jsonl": str(input_jsonl),
        "input_sha256": sha256_file(input_jsonl),
        "index": str(index_path),
        "index_meta": str(index_path.with_suffix(".meta.json")),
        "indices": sorted(set(indices)),
        "top_k_per_group": top_k_per_group,
        "min_similarity": min_similarity,
        "runs": receipts,
    }
    write_json_atomic(output_batch / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_CONFIG_MODULES), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-batch", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_retrieval_batch(
        task=args.task,
        input_jsonl=args.input_jsonl,
        index_path=args.index,
        output_batch=args.output_batch,
        indices=args.indices,
        top_k_per_group=args.top_k_per_group,
        min_similarity=args.min_similarity,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
