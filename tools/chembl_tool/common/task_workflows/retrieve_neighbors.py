"""Shared analog-neighbor retrieval for ChEMBL reasoning tasks."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from rdkit import DataStructs

from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp


def main(default_index: str, description: str, argv: list[str] | None = None) -> int:
    args = _parse_args(default_index, description, argv)
    started = time.monotonic()
    index = load_index(Path(args.index))
    if args.query_smiles:
        result = retrieve_neighbors(
            args.query_smiles,
            index,
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            groups=args.groups,
        )
        _write_single_result(result, args.out)
    elif args.query_jsonl:
        _run_jsonl(args, index)
    else:
        raise SystemExit("Provide --query-smiles or --query-jsonl.")
    _log(f"finished in {time.monotonic() - started:.3f}s")
    return 0


def load_index(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def retrieve_neighbors(
    query_smiles: str,
    index: dict[str, Any],
    *,
    top_k_per_group: int = 3,
    min_similarity: float = 0.3,
    groups: list[str] | None = None,
    neighbor_identity_policy: str = "operational",
) -> dict[str, Any]:
    from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
    from tools.chembl_tool.common.retrieval_policy import policy_metadata

    canonical_smiles, inchi_key, query_fp = standardize_smiles_and_fp(query_smiles)
    if query_fp is None:
        return {
            "query": {"input_smiles": query_smiles, "canonical_smiles": "", "standard_inchi_key": ""},
            "status": "error",
            "errors": [{"code": "INVALID_SMILES", "message": "Could not parse query SMILES.", "recoverable": True}],
            "groups": [],
            "coverage": {"n_groups": 0, "n_groups_with_neighbors": 0, "n_neighbors_total": 0},
        }

    query_identity = normalize_molecule_identity(query_smiles)
    similarities = list(DataStructs.BulkTanimotoSimilarity(query_fp, index["fingerprints"]))
    requested_groups = groups or sorted(index["group_to_molecule_indices"])
    output_groups = []
    n_neighbors_total = 0
    for group_id in requested_groups:
        molecule_indices = index["group_to_molecule_indices"].get(group_id, [])
        neighbors = _top_neighbors_for_group(
            group_id,
            molecule_indices,
            similarities,
            index,
            query_canonical_smiles=canonical_smiles,
            query_inchi_key=inchi_key,
            top_k=top_k_per_group,
            min_similarity=min_similarity,
            query_identity=query_identity,
            neighbor_identity_policy=neighbor_identity_policy,
        )
        n_neighbors_total += len(neighbors)
        tier, endpoint_group = _split_group_id(group_id)
        output_groups.append(
            {
                "group_id": group_id,
                "tier": tier,
                "endpoint_group": endpoint_group,
                "n_candidate_molecules": len(molecule_indices),
                "neighbors": neighbors,
            }
        )

    return {
        "status": "ok",
        "evidence_source": index.get("source", {}),
        "retrieval_policy": policy_metadata(neighbor_identity_policy),
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": canonical_smiles,
            "standard_inchi_key": inchi_key,
            "fingerprint": index.get("fingerprint", {}),
        },
        "groups": output_groups,
        "coverage": {
            "n_groups": len(output_groups),
            "n_groups_with_neighbors": sum(1 for group in output_groups if group["neighbors"]),
            "n_neighbors_total": n_neighbors_total,
            "min_similarity": min_similarity,
            "top_k_per_group": top_k_per_group,
        },
    }


def similarity_bucket(similarity: float) -> str:
    if similarity >= 0.95:
        return "very_close_analog"
    if similarity >= 0.80:
        return "close_analog"
    if similarity >= 0.60:
        return "moderate_analog"
    if similarity >= 0.40:
        return "weak_analog"
    if similarity >= 0.20:
        return "distant_analog"
    return "very_distant_analog"


def _top_neighbors_for_group(
    group_id: str,
    molecule_indices: list[int],
    similarities: list[float],
    index: dict[str, Any],
    *,
    query_canonical_smiles: str,
    query_inchi_key: str,
    top_k: int,
    min_similarity: float,
    query_identity: Any,
    neighbor_identity_policy: str,
) -> list[dict[str, Any]]:
    from tools.chembl_tool.common.retrieval_policy import decide_candidate

    ranked = sorted(
        (
            (similarities[molecule_index], molecule_index)
            for molecule_index in molecule_indices
            if similarities[molecule_index] >= min_similarity
        ),
        key=lambda item: (-item[0], index["molecules"][item[1]]["molecule_chembl_id"]),
    )
    neighbors: list[dict[str, Any]] = []
    for similarity, molecule_index in ranked:
        molecule = index["molecules"][molecule_index]
        decision = decide_candidate(query_identity, molecule, neighbor_identity_policy)
        if decision.excluded:
            continue
        evidence_rows = index["evidence_by_molecule_group"][molecule["molecule_chembl_id"]][group_id]
        neighbors.append(
            {
                "rank": len(neighbors) + 1,
                "molecule_chembl_id": molecule["molecule_chembl_id"],
                "canonical_smiles": molecule["canonical_smiles"],
                "standard_inchi_key": molecule.get("standard_inchi_key", ""),
                "similarity": round(float(similarity), 6),
                "similarity_bucket": similarity_bucket(float(similarity)),
                "molecule_relation": decision.relation.value,
                "n_evidence_rows": len(evidence_rows),
                "evidence_rows": evidence_rows,
            }
        )
        if len(neighbors) >= top_k:
            break
    return neighbors


def _is_exact_same_molecule(molecule: dict[str, Any], query_canonical_smiles: str, query_inchi_key: str) -> bool:
    molecule_inchi_key = str(molecule.get("standard_inchi_key") or "")
    if query_inchi_key and molecule_inchi_key == query_inchi_key:
        return True
    query_connectivity = _inchi_key_connectivity_layer(query_inchi_key)
    molecule_connectivity = _inchi_key_connectivity_layer(molecule_inchi_key)
    if query_connectivity and molecule_connectivity and query_connectivity == molecule_connectivity:
        return True
    if query_canonical_smiles and molecule.get("canonical_smiles") == query_canonical_smiles:
        return True
    return False


def _inchi_key_connectivity_layer(inchi_key: str) -> str:
    return str(inchi_key or "").strip().split("-", 1)[0]


def _run_jsonl(args: argparse.Namespace, index: dict[str, Any]) -> None:
    out_handle = Path(args.out).open("w", encoding="utf-8") if args.out else sys.stdout
    try:
        with Path(args.query_jsonl).open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if args.limit and i >= args.limit:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                smiles = str(record.get(args.smiles_field) or "")
                result = retrieve_neighbors(
                    smiles,
                    index,
                    top_k_per_group=args.top_k_per_group,
                    min_similarity=args.min_similarity,
                    groups=args.groups,
                )
                result["source"] = {"query_index": i, "smiles_field": args.smiles_field}
                out_handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    finally:
        if out_handle is not sys.stdout:
            out_handle.close()


def _write_single_result(result: dict[str, Any], out: str) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _split_group_id(group_id: str) -> tuple[str, str]:
    if "." not in group_id:
        return group_id, ""
    tier, group = group_id.split(".", 1)
    return tier, group


def _parse_args(default_index: str, description: str, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--index", default=default_index)
    parser.add_argument("--query-smiles")
    parser.add_argument("--query-jsonl")
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--out", default="")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--groups", nargs="*", default=None)
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[retrieve_neighbors] {message}", file=sys.stderr, flush=True)
