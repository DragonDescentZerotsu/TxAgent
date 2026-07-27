"""Materialize frozen BBB D/C/H1/H2 retrieval replay batches without calling an LLM."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.distance_retrieval import retrieve_all_distance_views
from tools.chembl_tool.common.evidence_distance import stable_config_hash, validate_distance_config
from tools.chembl_tool.tasks.bbb_martins.distance_config import DISTANCE_CONFIG


CONDITIONS = {
    "distance_d": ("D", "flat"),
    "distance_dc": ("D+C", "flat"),
    "distance_dc_h1": ("D+C+H1", "flat"),
    "distance_dc_h1_h2": ("D+C+H1+H2", "flat"),
    "distance_mechanism_dc": ("D+C", "mechanism"),
    "distance_mechanism_dc_h1": ("D+C+H1", "mechanism"),
    "distance_mechanism_dc_h1_h2": ("D+C+H1+H2", "mechanism"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default="data/processed/BBB_Martins/test.jsonl")
    parser.add_argument(
        "--index",
        default=(
            "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/evidence_library/v3/"
            "bbb_distance_superset_index.pkl"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "outputs/chembl_tool/tasks/bbb_martins/distance_expansion/retrieval_replay/v3"
        ),
    )
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.30)
    parser.add_argument("--neighbor-identity-policy", default="operational")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    validate_distance_config(DISTANCE_CONFIG)
    records = _read_jsonl(Path(args.input_jsonl), limit=args.limit)
    with Path(args.index).open("rb") as handle:
        index = pickle.load(handle)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {condition: 0 for condition in CONDITIONS}
    for sample_index, record in enumerate(records):
        query_smiles = str(record.get("drug") or "")
        if not query_smiles:
            raise ValueError(f"Sample {sample_index} has no `drug` SMILES.")
        views = retrieve_all_distance_views(
            query_smiles,
            index,
            config=DISTANCE_CONFIG,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            neighbor_identity_policy=args.neighbor_identity_policy,
        )
        for condition, (prefix, view) in CONDITIONS.items():
            retrieval = views[prefix][view]
            if retrieval.get("status") != "ok":
                raise ValueError(
                    f"Retrieval failed for sample {sample_index}, condition {condition}: "
                    f"{retrieval.get('errors')}"
                )
            run_dir = output_root / condition / "runs" / f"{condition}_idx{sample_index:05d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "retrieval.json").write_text(
                json.dumps(retrieval, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            counts[condition] += 1

    manifest = {
        "task_name": DISTANCE_CONFIG.task_name,
        "source_name": DISTANCE_CONFIG.source_name,
        "source_release": DISTANCE_CONFIG.source_release,
        "distance_config_hash": stable_config_hash(DISTANCE_CONFIG),
        "input_jsonl": str(args.input_jsonl),
        "index": str(args.index),
        "n_samples": len(records),
        "retrieval_policy": {
            "top_k_per_group": args.top_k_per_group,
            "min_similarity": args.min_similarity,
            "neighbor_identity_policy": args.neighbor_identity_policy,
        },
        "conditions": {
            condition: {"prefix": prefix, "view": view, "n_samples": counts[condition]}
            for condition, (prefix, view) in CONDITIONS.items()
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records


if __name__ == "__main__":
    raise SystemExit(main())
