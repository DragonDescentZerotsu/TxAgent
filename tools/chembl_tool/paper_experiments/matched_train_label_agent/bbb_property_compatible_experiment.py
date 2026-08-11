"""Run the single frozen E16 BBB property-compatible matched-agent candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
from tools.chembl_tool.common.retrieval_policy import decide_candidate
from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import MEANINGFUL_CNS_ADJUDICATION_V3

from .bbb_property_compatibility import (
    budget_key,
    morgan_top_n,
    select_compatible_neighbors,
)
from .bbb_property_compatible_contract import (
    AVAILABILITY_ROOT,
    BASELINE_V3_RUN_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SCHEMA_VERSION,
    SIMILARITY_BUDGET,
    SPEC,
    TOP_N,
)
from .bbb_property_compatible_report import summarize
from .contract import replay_batch
from .materialize import _neighbor, _retrieval
from .run import run_experiment


def materialize(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    train_path = SPEC.input_jsonl.parent / "train.jsonl"
    train, valid = _read_jsonl(train_path), _read_jsonl(SPEC.input_jsonl)
    availability_rows = _read_jsonl(AVAILABILITY_ROOT / "query_audit.jsonl")
    availability_summary = json.loads(
        (AVAILABILITY_ROOT / "summary.json").read_text(encoding="utf-8")
    )
    profiles = {
        row["key"]: row["features"]
        for row in _read_jsonl(AVAILABILITY_ROOT / "molecule_properties.jsonl")
    }
    train_features = [profiles[f"train:{index}"] for index in range(len(train))]
    valid_features = [profiles[f"valid:{index}"] for index in range(len(valid))]
    candidates = morgan_top_n(train, valid, top_n=TOP_N)
    audit_key = budget_key(SIMILARITY_BUDGET)
    batch = replay_batch(output_root, SPEC)
    relation_counts: dict[str, int] = {}

    if len(availability_rows) != len(valid):
        raise ValueError("Availability audit and valid counts differ")
    for query_index, (query, pool, audited) in enumerate(
        zip(valid, candidates, availability_rows, strict=True)
    ):
        selected = select_compatible_neighbors(
            pool,
            query_features=valid_features[query_index],
            train_features=train_features,
            scales=availability_summary["feature_scales"],
            top_k=3,
            similarity_floor=float(pool[2]["similarity"]) - SIMILARITY_BUDGET,
        )
        expected = audited["budgets"][audit_key]
        observed = [int(row["train_index"]) for row in selected]
        if observed != [int(value) for value in expected["train_indices"]]:
            raise ValueError(f"Availability parity failure at query {query_index}")
        query_identity = normalize_molecule_identity(str(query["drug"]))
        neighbors = []
        for rank, candidate in enumerate(selected, start=1):
            source = {**candidate, "drug": train[int(candidate["train_index"])]["drug"]}
            decision = decide_candidate(
                query_identity, {"canonical_smiles": source["drug"]}, "parent_disjoint"
            )
            if decision.excluded:
                raise ValueError(
                    f"query {query_index} neighbor {rank} violates parent_disjoint"
                )
            relation = decision.relation.value
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            neighbors.append(_neighbor(SPEC, source, rank, relation))
        retrieval = _retrieval(SPEC, str(query["drug"]), neighbors, len(train))
        retrieval["experiment"].update(
            {
                "source": "matched_train_label_property_compatible_v1",
                "matched_knn_neighbor_set": False,
                "neighbor_selector": {
                    "name": "bbb_property_compatible",
                    "version": "bbb_property_compatible.v1",
                    "candidate_pool": f"Morgan top-{TOP_N}",
                    "rank3_similarity_cost_budget": SIMILARITY_BUDGET,
                    "selection_uses_labels": False,
                    "selection_order": availability_summary["selection_order"],
                    "features": availability_summary["features"],
                },
            }
        )
        run_id = f"{batch.name}_idx{query_index:05d}"
        write_json_atomic(batch / "runs" / run_id / "retrieval.json", retrieval)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": SPEC.task,
        "evaluation_subset": "valid",
        "input_jsonl": str(SPEC.input_jsonl),
        "n_queries": len(valid),
        "k": 3,
        "candidate_pool": f"Morgan top-{TOP_N}",
        "similarity_cost_budget": SIMILARITY_BUDGET,
        "selection_uses_labels": False,
        "selection_order": availability_summary["selection_order"],
        "neighbor_identity_policy": "parent_disjoint",
        "visible_neighbor_supervision": "frozen_train_Y_and_task_label_meaning",
        "relation_counts": relation_counts,
        "availability_artifact": {
            "path": str(AVAILABILITY_ROOT / "query_audit.jsonl"),
            "sha256": sha256_file(AVAILABILITY_ROOT / "query_audit.jsonl"),
        },
        "batch": str(batch),
    }
    write_json_atomic(batch / "manifest.json", manifest)
    write_json_atomic(
        output_root / "retrieval_materialization_manifest.json",
        {"schema_version": SCHEMA_VERSION, "tasks": [manifest]},
    )
    return manifest


def run(*, output_root: Path = DEFAULT_OUTPUT_ROOT, parallelism: int = 128) -> dict[str, Any]:
    return run_experiment(
        output_root=output_root,
        tasks=["bbb_martins"],
        parallelism=parallelism,
        specs_override=[SPEC],
        single_analysis_root_overrides={"bbb_martins": BASELINE_V3_RUN_ROOT},
        retrieval_contract=(
            "frozen Morgan top-20, rank3 similarity cost <=0.02, label-blind "
            "ionization/property-compatible top-3 with train Y visible"
        ),
        bbb_prompt_profile=MEANINGFUL_CNS_ADJUDICATION_V3,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("materialize", "run", "summarize", "all"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallelism", type=int, default=128)
    args = parser.parse_args(argv)
    if args.action in {"materialize", "all"}:
        print(json.dumps(materialize(output_root=args.output_root), indent=2))
    if args.action in {"run", "all"}:
        print(json.dumps(run(output_root=args.output_root, parallelism=args.parallelism), indent=2))
    if args.action in {"summarize", "all"}:
        print(json.dumps(summarize(output_root=args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
