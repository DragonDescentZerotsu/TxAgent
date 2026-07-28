"""Build precomputed MiniMol/cosine feature stores for Starling agent retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from baselines.minimol.embedding_runtime import (
    DEFAULT_MINIMOL_SOURCE,
    checkpoint_provenance,
    create_featurizer,
    embed_smiles,
)
from tools.chembl_tool.common.retrieval_features import (
    DESCRIPTOR_TYPE,
    candidate_order_sha256,
)
from tools.chembl_tool.common.starling.heldout_index import (
    identity_key,
    load_heldout_identity_keys,
)
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp
from tools.chembl_tool.paper_experiments.build_starling_benchmark_indices import BENCHMARK_SPLITS
from tools.chembl_tool.paper_experiments.minimol_retrieval_contract import (
    DEFAULT_FEATURE_ROOT,
    descriptor_path_for_experiment,
)
from tools.chembl_tool.paper_experiments.starling_benchmark_matrix import (
    experiments_for_starling_benchmark,
)


MODEL_NAME = "MiniMol"
MODEL_VERSION = "minimol_v1"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    splits = args.splits or list(BENCHMARK_SPLITS)
    experiments = [
        experiment
        for split in splits
        for experiment in experiments_for_starling_benchmark(split)
        if experiment.mode != "none"
    ]
    model_provenance = checkpoint_provenance(args.minimol_source)
    base_indices = _load_unique_indices(experiments)
    query_sets = _load_query_sets(experiments)
    all_smiles = sorted(
        {
            str(molecule["canonical_smiles"])
            for index in base_indices.values()
            for molecule in index["molecules"]
        }
        | {
            canonical
            for query_rows in query_sets.values()
            for canonical in query_rows
        }
    )
    registry_dir = args.output_root / "registry"
    registry_embeddings, registry_rows, registry_manifest = _build_registry(
        all_smiles,
        registry_dir=registry_dir,
        args=args,
    )
    candidate_stores = _materialize_candidate_stores(
        base_indices,
        registry_embeddings=registry_embeddings,
        registry_rows=registry_rows,
        output_root=args.output_root,
    )
    query_stores = _materialize_query_stores(
        query_sets,
        registry_embeddings=registry_embeddings,
        registry_rows=registry_rows,
        output_root=args.output_root,
    )

    descriptors = []
    for experiment in experiments:
        split = _split_from_input(experiment.input_jsonl)
        index_path = str(Path(experiment.index))
        query_key = (split, experiment.task, experiment.input_jsonl)
        descriptor_path = descriptor_path_for_experiment(
            split,
            experiment.name,
            output_root=args.output_root,
        )
        candidate = candidate_stores[index_path]
        query = query_stores[query_key]
        descriptor = {
            "type": DESCRIPTOR_TYPE,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_provenance": model_provenance,
            "normalization": "L2",
            "similarity": "cosine",
            "base_index_path": index_path,
            "candidate_embeddings_path": str(candidate["embeddings_path"]),
            "candidate_order_sha256": candidate["candidate_order_sha256"],
            "query_embeddings_path": str(query["embeddings_path"]),
            "query_manifest_path": str(query["manifest_path"]),
            "benchmark_split": split,
            "task": experiment.task,
            "experiment": experiment.name,
            "input_jsonl": experiment.input_jsonl,
            "input_jsonl_sha256": _sha256_file(Path(experiment.input_jsonl)),
            "registry_sha256": registry_manifest["canonical_smiles_sha256"],
        }
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
        descriptors.append(str(descriptor_path))

    audit = _audit_build(
        experiments,
        base_indices=base_indices,
        candidate_stores=candidate_stores,
        query_sets=query_sets,
        query_stores=query_stores,
    )
    summary = {
        "type": "minimol_agent_retrieval_feature_build.v1",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_provenance": model_provenance,
        "normalization": "L2",
        "similarity": "cosine",
        "splits": splits,
        "n_unique_smiles": len(all_smiles),
        "n_base_indices": len(base_indices),
        "n_query_sets": len(query_sets),
        "n_descriptors": len(descriptors),
        "registry": registry_manifest,
        "audit": audit,
        "descriptors": descriptors,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2), flush=True)
    return 0


def _load_unique_indices(experiments: Iterable[Any]) -> dict[str, dict[str, Any]]:
    indices: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        path = str(Path(experiment.index))
        if path in indices:
            continue
        with Path(path).open("rb") as handle:
            indices[path] = pickle.load(handle)
    return indices


def _load_query_sets(experiments: Iterable[Any]) -> dict[tuple[str, str, str], list[str]]:
    query_sets: dict[tuple[str, str, str], list[str]] = {}
    for experiment in experiments:
        split = _split_from_input(experiment.input_jsonl)
        key = (split, experiment.task, experiment.input_jsonl)
        if key in query_sets:
            continue
        canonical_smiles: list[str] = []
        with Path(experiment.input_jsonl).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                canonical, _, fingerprint = standardize_smiles_and_fp(str(row.get("drug") or ""))
                if fingerprint is None:
                    raise ValueError(
                        f"{experiment.input_jsonl}:{line_number} has invalid SMILES"
                    )
                canonical_smiles.append(canonical)
        query_sets[key] = canonical_smiles
    return query_sets


def _build_registry(
    canonical_smiles: list[str],
    *,
    registry_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    smiles_sha256 = hashlib.sha256(
        "".join(f"{smiles}\n" for smiles in canonical_smiles).encode("utf-8")
    ).hexdigest()
    embeddings_path = registry_dir / "embeddings.npy"
    manifest_path = registry_dir / "manifest.json"
    smiles_path = registry_dir / "canonical_smiles.jsonl"
    progress_path = registry_dir / "progress.json"
    fallback_path = registry_dir / "fallbacks.jsonl"
    if not args.force and embeddings_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("canonical_smiles_sha256") == smiles_sha256
            and int(manifest.get("n_rows") or -1) == len(canonical_smiles)
        ):
            embeddings = np.load(embeddings_path, mmap_mode="r")
            return embeddings, {smiles: i for i, smiles in enumerate(canonical_smiles)}, manifest

    registry_dir.mkdir(parents=True, exist_ok=True)
    featurizer = create_featurizer(
        batch_size=args.embedding_batch_size,
        minimol_source=args.minimol_source,
    )
    embeddings_memmap: np.memmap | None = None
    dimension = 0
    resume_row = 0
    fallback_records: list[dict[str, Any]] = []
    if embeddings_path.exists() and not args.force:
        partial = np.load(embeddings_path, mmap_mode="r+")
        if partial.ndim == 2 and partial.shape[0] == len(canonical_smiles):
            nonzero = np.any(partial != 0, axis=1)
            zero_rows = np.flatnonzero(~nonzero)
            resume_row = int(zero_rows[0]) if len(zero_rows) else len(canonical_smiles)
            if np.any(nonzero[resume_row:]):
                raise ValueError("Partial MiniMol registry has non-contiguous completed rows")
            embeddings_memmap = partial
            dimension = int(partial.shape[1])
            if progress_path.exists():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                fallback_records = list(progress.get("fallbacks") or [])
            print(
                f"[minimol_retrieval_features] resuming registry at row "
                f"{resume_row}/{len(canonical_smiles)}",
                flush=True,
            )
    for start in range(resume_row, len(canonical_smiles), args.chunk_size):
        chunk = canonical_smiles[start : start + args.chunk_size]
        tensor, chunk_fallbacks = _embed_with_audited_fallback(featurizer, chunk)
        tensor = F.normalize(tensor, p=2, dim=1)
        array = tensor.numpy().astype(np.float32, copy=False)
        if embeddings_memmap is None:
            dimension = int(array.shape[1])
            embeddings_memmap = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(canonical_smiles), dimension),
            )
        embeddings_memmap[start : start + len(chunk)] = array
        embeddings_memmap.flush()
        fallback_records.extend(
            {
                **record,
                "registry_row": start + int(record["chunk_row"]),
            }
            for record in chunk_fallbacks
        )
        progress_path.write_text(
            json.dumps(
                {
                    "canonical_smiles_sha256": smiles_sha256,
                    "completed_rows": start + len(chunk),
                    "fallbacks": fallback_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[minimol_retrieval_features] embedded={start + len(chunk)}/"
            f"{len(canonical_smiles)}",
            flush=True,
        )
    if embeddings_memmap is None:
        raise ValueError("No SMILES were collected for MiniMol feature construction")
    del embeddings_memmap
    fallback_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in fallback_records),
        encoding="utf-8",
    )
    smiles_path.write_text(
        "".join(json.dumps({"row": i, "canonical_smiles": smiles}) + "\n"
                for i, smiles in enumerate(canonical_smiles)),
        encoding="utf-8",
    )
    manifest = {
        "type": "minimol_embedding_registry.v1",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "normalization": "L2",
        "dimension": dimension,
        "n_rows": len(canonical_smiles),
        "canonical_smiles_sha256": smiles_sha256,
        "embeddings_path": str(embeddings_path),
        "canonical_smiles_path": str(smiles_path),
        "n_featurization_fallbacks": len(fallback_records),
        "featurization_fallback_version": "minimol_parseable_parent_or_fragment.v1",
        "featurization_fallbacks_path": str(fallback_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    return (
        np.load(embeddings_path, mmap_mode="r"),
        {smiles: i for i, smiles in enumerate(canonical_smiles)},
        manifest,
    )


def _embed_with_audited_fallback(
    featurizer: Any,
    smiles: list[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Isolate MiniMol graph failures and use a recorded parseable parent/fragment."""
    try:
        return embed_smiles(featurizer, smiles), []
    except AttributeError:
        if len(smiles) > 1:
            midpoint = len(smiles) // 2
            left, left_fallbacks = _embed_with_audited_fallback(
                featurizer,
                smiles[:midpoint],
            )
            right, right_fallbacks = _embed_with_audited_fallback(
                featurizer,
                smiles[midpoint:],
            )
            for record in right_fallbacks:
                record["chunk_row"] = int(record["chunk_row"]) + midpoint
            return torch.cat((left, right), dim=0), left_fallbacks + right_fallbacks

        original = smiles[0]
        fallback, reason = _minimol_fallback_smiles(original)
        if not fallback or fallback == original:
            raise
        tensor = embed_smiles(featurizer, [fallback])
        return tensor, [
            {
                "chunk_row": 0,
                "original_smiles": original,
                "embedded_smiles": fallback,
                "reason": reason,
            }
        ]


def _minimol_fallback_smiles(smiles: str) -> tuple[str, str]:
    from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity

    identity = normalize_molecule_identity(smiles)
    if identity.parent_smiles and identity.parent_smiles != smiles:
        return identity.parent_smiles, "rdkit_fragment_parent"
    candidates = []
    for fragment in str(smiles).split("."):
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            continue
        candidates.append(
            (
                molecule.GetNumHeavyAtoms(),
                Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
            )
        )
    if not candidates:
        return "", "no_parseable_fragment"
    _, fallback = max(candidates, key=lambda item: (item[0], item[1]))
    return fallback, "largest_parseable_fragment"


def _materialize_candidate_stores(
    indices: dict[str, dict[str, Any]],
    *,
    registry_embeddings: np.ndarray,
    registry_rows: dict[str, int],
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    stores: dict[str, dict[str, Any]] = {}
    for base_index_path, index in indices.items():
        order_sha256 = candidate_order_sha256(index)
        store_dir = output_root / "candidate_stores" / order_sha256[:16]
        embeddings_path = store_dir / "embeddings.npy"
        manifest_path = store_dir / "manifest.json"
        store_dir.mkdir(parents=True, exist_ok=True)
        if not embeddings_path.exists():
            rows = np.asarray(
                [registry_rows[str(item["canonical_smiles"])] for item in index["molecules"]],
                dtype=np.int64,
            )
            target = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), registry_embeddings.shape[1]),
            )
            for start in range(0, len(rows), 10000):
                target[start : start + 10000] = registry_embeddings[rows[start : start + 10000]]
            target.flush()
            del target
        manifest = {
            "type": "minimol_candidate_store.v1",
            "base_index_path": base_index_path,
            "n_rows": len(index["molecules"]),
            "dimension": int(registry_embeddings.shape[1]),
            "candidate_order_sha256": order_sha256,
            "embeddings_path": str(embeddings_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stores[base_index_path] = {
            **manifest,
            "manifest_path": manifest_path,
            "embeddings_path": embeddings_path,
        }
    return stores


def _audit_build(
    experiments: list[Any],
    *,
    base_indices: dict[str, dict[str, Any]],
    candidate_stores: dict[str, dict[str, Any]],
    query_sets: dict[tuple[str, str, str], list[str]],
    query_stores: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Gate row-order/query coverage and explicit Starling test-parent exclusion."""
    checked_starling_pairs: set[tuple[str, str]] = set()
    starling_overlap_counts: dict[str, int] = {}
    cache_parity: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        split = _split_from_input(experiment.input_jsonl)
        base_path = str(Path(experiment.index))
        index = base_indices[base_path]
        candidate_store = candidate_stores[base_path]
        embeddings = np.load(candidate_store["embeddings_path"], mmap_mode="r")
        if embeddings.shape[0] != len(index["molecules"]):
            raise AssertionError("Candidate feature row count changed after materialization")
        if candidate_order_sha256(index) != candidate_store["candidate_order_sha256"]:
            raise AssertionError("Candidate feature order changed after materialization")

        query_store = query_stores[(split, experiment.task, experiment.input_jsonl)]
        query_manifest = json.loads(
            Path(query_store["manifest_path"]).read_text(encoding="utf-8")
        )
        expected_queries = set(
            query_sets[(split, experiment.task, experiment.input_jsonl)]
        )
        stored_queries = set(query_manifest["canonical_smiles_to_row"])
        if expected_queries != stored_queries:
            raise AssertionError(f"Query embedding coverage mismatch for {experiment.name}")

        cache_key = f"{split}:{experiment.task}"
        if cache_key not in cache_parity:
            cache_path = (
                Path("outputs/baselines/minimol_starling")
                / _task_data_name(experiment.task)
                / split
                / "embeddings/test.pt"
            )
            cache_parity[cache_key] = _audit_formal_test_cache_parity(
                cache_path,
                query_store=query_store,
            )

        starling_key = (split, base_path)
        if experiment.source != "starling" or starling_key in checked_starling_pairs:
            continue
        heldout_path = (
            Path("data/processed_starling")
            / _task_data_name(experiment.task)
            / split
            / "test_molecule_labels.jsonl"
        )
        heldout_keys = load_heldout_identity_keys(heldout_path)
        candidate_keys = {identity_key(molecule) for molecule in index["molecules"]}
        overlap = heldout_keys & candidate_keys
        starling_overlap_counts[f"{split}:{base_path}"] = len(overlap)
        if overlap:
            raise AssertionError(
                f"Starling MiniMol candidate store retains {len(overlap)} held-out parents"
            )
        checked_starling_pairs.add(starling_key)
    return {
        "status": "passed",
        "descriptor_count": len(experiments),
        "candidate_row_order": "passed",
        "query_embedding_coverage": "passed",
        "formal_test_embedding_cache_parity": "passed",
        "formal_test_embedding_cache_cosine": cache_parity,
        "starling_test_parent_exclusion": "passed",
        "n_starling_index_split_pairs": len(checked_starling_pairs),
        "starling_parent_overlap_counts": starling_overlap_counts,
        "chembl_parent_exclusion": "enforced_at_query_time_by_parent_disjoint_policy",
    }


def _task_data_name(task: str) -> str:
    return {
        "bbb_martins": "BBB_Martins",
        "bioavailability_ma": "Bioavailability_Ma",
        "skin_reaction": "Skin_Reaction",
    }[task]


def _audit_formal_test_cache_parity(
    cache_path: Path,
    *,
    query_store: dict[str, Any],
) -> dict[str, Any]:
    """Confirm query rows are the same MiniMol feature used by the formal baseline."""
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing formal MiniMol test embedding cache: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached = F.normalize(payload["embeddings"].float(), p=2, dim=1).numpy()
    stored = np.load(query_store["embeddings_path"], mmap_mode="r")
    manifest = json.loads(Path(query_store["manifest_path"]).read_text(encoding="utf-8"))
    row_by_smiles = manifest["canonical_smiles_to_row"]
    similarities = []
    for row, smiles in enumerate(payload["smiles"]):
        canonical, _, fingerprint = standardize_smiles_and_fp(str(smiles))
        if fingerprint is None or canonical not in row_by_smiles:
            raise AssertionError(f"Formal MiniMol cache query is absent from feature store: {smiles}")
        stored_row = np.asarray(stored[int(row_by_smiles[canonical])], dtype=np.float32)
        similarities.append(float(np.dot(cached[row], stored_row)))
    minimum = min(similarities)
    if minimum < 0.99999:
        raise AssertionError(
            f"MiniMol query features differ from formal baseline cache: min cosine={minimum}"
        )
    return {
        "cache_path": str(cache_path),
        "cache_sha256": _sha256_file(cache_path),
        "n_rows": len(similarities),
        "minimum_cosine": minimum,
        "mean_cosine": float(np.mean(similarities)),
    }


def _materialize_query_stores(
    query_sets: dict[tuple[str, str, str], list[str]],
    *,
    registry_embeddings: np.ndarray,
    registry_rows: dict[str, int],
    output_root: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    stores: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (split, task, input_jsonl), canonical_smiles in query_sets.items():
        store_dir = output_root / split / "queries" / task
        embeddings_path = store_dir / "embeddings.npy"
        manifest_path = store_dir / "manifest.json"
        store_dir.mkdir(parents=True, exist_ok=True)
        unique_smiles = list(dict.fromkeys(canonical_smiles))
        rows = np.asarray([registry_rows[smiles] for smiles in unique_smiles], dtype=np.int64)
        np.save(
            embeddings_path,
            np.asarray(registry_embeddings[rows], dtype=np.float32),
        )
        manifest = {
            "type": "minimol_query_store.v1",
            "benchmark_split": split,
            "task": task,
            "input_jsonl": input_jsonl,
            "input_jsonl_sha256": _sha256_file(Path(input_jsonl)),
            "n_input_rows": len(canonical_smiles),
            "n_unique_canonical_smiles": len(unique_smiles),
            "dimension": int(registry_embeddings.shape[1]),
            "canonical_smiles_to_row": {
                smiles: row for row, smiles in enumerate(unique_smiles)
            },
            "embeddings_path": str(embeddings_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stores[(split, task, input_jsonl)] = {
            **manifest,
            "manifest_path": manifest_path,
            "embeddings_path": embeddings_path,
        }
    return stores


def _split_from_input(input_jsonl: str) -> str:
    split = Path(input_jsonl).parent.name
    if split not in BENCHMARK_SPLITS:
        raise ValueError(f"Could not infer Starling benchmark split from {input_jsonl}")
    return split


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", choices=BENCHMARK_SPLITS, default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--minimol-source", type=Path, default=DEFAULT_MINIMOL_SOURCE)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
