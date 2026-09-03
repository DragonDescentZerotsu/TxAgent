"""Execute indexed analog retrieval and return evidence-bearing neighbors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from predict.retrieval.compact import load_compact_index, read_compact_manifest
from predict.retrieval.policies import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
    NeighborCandidate,
    decide_candidate,
    normalize_molecule_identity,
    policy_metadata,
    select_neighbor_candidates,
    selector_metadata,
    standardize_smiles_and_fp,
)
from predict.retrieval.features import (
    load_retrieval_index,
    retrieval_feature_metadata,
    similarity_bucket_for_index,
    similarity_vector,
)


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
            neighbor_selector=args.morgan_neighbor_selector,
        )
        _write_single_result(result, args.out)
    elif args.query_jsonl:
        _run_jsonl(args, index)
    else:
        raise SystemExit("Provide --query-smiles or --query-jsonl.")
    _log(f"finished in {time.monotonic() - started:.3f}s")
    return 0


def load_index(path: Path) -> dict[str, Any]:
    if path.is_dir():
        manifest = read_compact_manifest(path)
        task_id = _compact_index_task_id(manifest)
        if task_id:
            return load_compact_index(path, manifest)
        raise ValueError(f"unsupported directory index format: {path}")
    return load_retrieval_index(path)


def _compact_index_task_id(manifest: dict[str, Any]) -> str:
    """Resolve the owning task of a compact directory index.

    Newer indices record ``task_id`` directly.  Indices written before that
    field existed are identified by the ``<task>.compact_neighbor_index.v1``
    version prefix, so an already-built index needs no rebuild to load.
    """
    task_id = str(manifest.get("task_id") or "").strip()
    if task_id:
        return task_id
    version = str(manifest.get("index_version") or "")
    prefix, separator, suffix = version.partition(".")
    return prefix if separator and suffix == "compact_neighbor_index.v1" else ""


def retrieve_neighbors(
    query_smiles: str,
    index: dict[str, Any],
    *,
    top_k_per_group: int = 3,
    min_similarity: float = 0.3,
    groups: list[str] | None = None,
    neighbor_identity_policy: str = "operational",
    neighbor_selector: str = SIMILARITY_SELECTOR,
) -> dict[str, Any]:
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
    similarities = similarity_vector(
        query_fp,
        canonical_smiles,
        index,
        neighbor_selector=neighbor_selector,
    )
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
            top_k=top_k_per_group,
            min_similarity=min_similarity,
            query_identity=query_identity,
            neighbor_identity_policy=neighbor_identity_policy,
            query_fingerprint=query_fp,
            neighbor_selector=neighbor_selector,
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

    retrieval_policy = {
        **policy_metadata(neighbor_identity_policy),
        "neighbor_selector": selector_metadata(neighbor_selector),
    }

    return {
        "status": "ok",
        "evidence_source": index.get("source", {}),
        "retrieval_policy": retrieval_policy,
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": canonical_smiles,
            "standard_inchi_key": inchi_key,
            "fingerprint": (
                index.get("fingerprint", {})
                if retrieval_feature_metadata(index)["feature"] == "morgan_fingerprint"
                else {}
            ),
            "retrieval_feature": retrieval_feature_metadata(index),
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
    """Backward-compatible Morgan/Tanimoto bucket helper."""
    return similarity_bucket_for_index(similarity, {})


def _top_neighbors_for_group(
    group_id: str,
    molecule_indices: list[int],
    similarities: list[float],
    index: dict[str, Any],
    *,
    top_k: int,
    min_similarity: float,
    query_identity: Any,
    neighbor_identity_policy: str,
    query_fingerprint: Any,
    neighbor_selector: str,
) -> list[dict[str, Any]]:
    eligible: list[NeighborCandidate] = []
    decisions: dict[int, Any] = {}
    evidence_by_index: dict[int, list[dict[str, Any]]] = {}
    for molecule_index in molecule_indices:
        similarity = float(similarities[molecule_index])
        if similarity < min_similarity:
            continue
        molecule = index["molecules"][molecule_index]
        decision = decide_candidate(query_identity, molecule, neighbor_identity_policy)
        if decision.excluded:
            continue
        evidence_rows = index["evidence_by_molecule_group"].get(
            molecule["molecule_chembl_id"], {}
        ).get(group_id, [])
        if not evidence_rows:
            continue
        eligible.append(
            NeighborCandidate(
                molecule_index=molecule_index,
                molecule_id=str(molecule["molecule_chembl_id"]),
                similarity=similarity,
            )
        )
        decisions[molecule_index] = decision
        evidence_by_index[molecule_index] = evidence_rows

    selected = select_neighbor_candidates(
        eligible,
        query_fingerprint=query_fingerprint,
        candidate_fingerprints=index["fingerprints"],
        top_k=top_k,
        selector=neighbor_selector,
    )
    neighbors: list[dict[str, Any]] = []
    for candidate in selected:
        molecule_index = candidate.molecule_index
        similarity = candidate.similarity
        molecule = index["molecules"][molecule_index]
        decision = decisions[molecule_index]
        evidence_rows = evidence_by_index[molecule_index]
        neighbors.append(
            {
                "rank": len(neighbors) + 1,
                "molecule_chembl_id": molecule["molecule_chembl_id"],
                "canonical_smiles": molecule["canonical_smiles"],
                "standard_inchi_key": molecule.get("standard_inchi_key", ""),
                "similarity": round(float(similarity), 6),
                "similarity_bucket": similarity_bucket_for_index(float(similarity), index),
                "similarity_metric": retrieval_feature_metadata(index)["similarity"],
                "molecule_relation": decision.relation.value,
                "n_evidence_rows": len(evidence_rows),
                "evidence_rows": evidence_rows,
            }
        )
    return neighbors


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
                    neighbor_selector=args.morgan_neighbor_selector,
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
    parser.add_argument(
        "--morgan-neighbor-selector",
        choices=NEIGHBOR_SELECTORS,
        default=SIMILARITY_SELECTOR,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--groups", nargs="*", default=None)
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(f"[retrieve_neighbors] {message}", file=sys.stderr, flush=True)
