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
    write_finite_array_receipt,
)
from tools.chembl_tool.common.starling.heldout_index import (
    identity_key,
    load_heldout_identity_keys,
)
from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp
from tools.chembl_tool.paper_experiments.build_starling_benchmark_indices import (
    BENCHMARK_SPLITS,
    HELDOUT_SUBSETS,
    heldout_labels_path,
    normalize_heldout_subsets,
)
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
    args.heldout_subsets = list(normalize_heldout_subsets(args.heldout_subsets))
    if args.heldout_subsets == ["test"] and args.evaluation_subset != "test":
        raise SystemExit("A train+valid reference pool is allowed only for test evaluation")
    splits = args.splits or list(BENCHMARK_SPLITS)
    experiments = [
        experiment
        for split in splits
        for experiment in experiments_for_starling_benchmark(
            split,
            evaluation_subset=args.evaluation_subset,
            data_root=args.benchmark_data_root,
            canonical_paper_root=args.canonical_paper_root or None,
        )
        if experiment.mode != "none" and experiment.task in args.tasks
    ]
    if not experiments:
        raise SystemExit("No MiniMol retrieval experiments matched --tasks/--splits")
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
    if args.registry_source_root:
        registry_embeddings, registry_rows, registry_manifest = _load_registry_source(
            all_smiles,
            source_root=args.registry_source_root,
            expected_model_provenance=model_provenance,
            target_registry_dir=registry_dir,
            args=args,
        )
    else:
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
        benchmark_data_root=args.benchmark_data_root,
        evaluation_subset=args.evaluation_subset,
        formal_cache_root=args.formal_cache_root,
        heldout_subsets=args.heldout_subsets,
    )
    summary = {
        "type": "minimol_agent_retrieval_feature_build.v1",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_provenance": model_provenance,
        "normalization": "L2",
        "similarity": "cosine",
        "splits": splits,
        "evaluation_subset": args.evaluation_subset,
        "benchmark_lineage": args.benchmark_lineage,
        "benchmark_data_root": str(args.benchmark_data_root),
        "canonical_paper_root": str(args.canonical_paper_root),
        "heldout_subsets": args.heldout_subsets,
        "tasks": args.tasks,
        "n_unique_smiles": len(all_smiles),
        "n_base_indices": len(base_indices),
        "n_query_sets": len(query_sets),
        "n_descriptors": len(descriptors),
        "registry": registry_manifest,
        "registry_reused": bool(args.registry_source_root),
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


def _load_registry_source(
    canonical_smiles: list[str],
    *,
    source_root: Path,
    expected_model_provenance: dict[str, Any],
    target_registry_dir: Path | None = None,
    args: argparse.Namespace | None = None,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Reuse a superset registry built with the exact same frozen MiniMol checkpoint."""
    summary_path = source_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing reusable MiniMol summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("model_provenance") != expected_model_provenance:
        raise ValueError("Reusable MiniMol registry checkpoint provenance does not match")
    manifest = dict(summary.get("registry") or {})
    embeddings_path = Path(str(manifest.get("embeddings_path") or ""))
    smiles_path = Path(str(manifest.get("canonical_smiles_path") or ""))
    if not embeddings_path.is_file() or not smiles_path.is_file():
        raise FileNotFoundError("Reusable MiniMol registry artifacts are incomplete")
    rows: dict[str, int] = {}
    with smiles_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["canonical_smiles"])] = int(row["row"])
    missing = sorted(set(canonical_smiles) - set(rows))
    if missing:
        if target_registry_dir is not None and args is not None:
            return _augment_registry_source(
                canonical_smiles,
                missing=missing,
                source_embeddings=np.load(embeddings_path, mmap_mode="r"),
                source_rows=rows,
                source_manifest=manifest,
                source_root=source_root,
                target_registry_dir=target_registry_dir,
                args=args,
            )
        raise ValueError(
            "Reusable MiniMol registry is not a superset; missing "
            f"{len(missing)} canonical SMILES (first={missing[0]!r})"
        )
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise ValueError("Reusable MiniMol registry row count is inconsistent")
    return embeddings, rows, {
        **manifest,
        "reuse_source_root": str(source_root),
        "requested_unique_smiles": len(canonical_smiles),
        "reused_registry_rows": len(rows),
        "n_augmented_rows": 0,
    }


def _augment_registry_source(
    canonical_smiles: list[str],
    *,
    missing: list[str],
    source_embeddings: np.ndarray,
    source_rows: dict[str, int],
    source_manifest: dict[str, Any],
    source_root: Path,
    target_registry_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """Copy known rows and embed only molecules absent from an audited source registry."""
    target_registry_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = target_registry_dir / "embeddings.npy"
    smiles_path = target_registry_dir / "canonical_smiles.jsonl"
    fallback_path = target_registry_dir / "fallbacks.jsonl"
    dimension = int(source_embeddings.shape[1])
    target_rows = {smiles: row for row, smiles in enumerate(canonical_smiles)}
    target = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(canonical_smiles), dimension),
    )
    known = [smiles for smiles in canonical_smiles if smiles in source_rows]
    for start in range(0, len(known), 10000):
        chunk = known[start : start + 10000]
        source_indices = np.asarray([source_rows[smiles] for smiles in chunk], dtype=np.int64)
        target_indices = np.asarray([target_rows[smiles] for smiles in chunk], dtype=np.int64)
        target[target_indices] = source_embeddings[source_indices]

    featurizer = create_featurizer(
        batch_size=min(args.embedding_batch_size, max(1, len(missing))),
        minimol_source=args.minimol_source,
    )
    missing_tensor, fallback_records = _embed_with_audited_fallback(featurizer, missing)
    missing_array = F.normalize(missing_tensor, p=2, dim=1).numpy().astype(np.float32, copy=False)
    target_indices = np.asarray([target_rows[smiles] for smiles in missing], dtype=np.int64)
    target[target_indices] = missing_array
    target.flush()
    del target

    smiles_path.write_text(
        "".join(
            json.dumps({"row": row, "canonical_smiles": smiles}) + "\n"
            for row, smiles in enumerate(canonical_smiles)
        ),
        encoding="utf-8",
    )
    fallback_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in fallback_records),
        encoding="utf-8",
    )
    smiles_sha256 = hashlib.sha256(
        "".join(f"{smiles}\n" for smiles in canonical_smiles).encode("utf-8")
    ).hexdigest()
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
        "reuse_source_root": str(source_root),
        "reuse_source_registry_sha256": source_manifest.get("canonical_smiles_sha256"),
        "reused_requested_rows": len(known),
        "n_augmented_rows": len(missing),
    }
    (target_registry_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[minimol_retrieval_features] reused={len(known)} augmented={len(missing)}",
        flush=True,
    )
    return np.load(embeddings_path, mmap_mode="r"), target_rows, manifest


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
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"MiniMol candidate store contains non-finite values: {embeddings_path}")
        finite_receipt_path = write_finite_array_receipt(
            embeddings_path,
            array=embeddings,
        )
        manifest["finite_receipt_path"] = str(finite_receipt_path)
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
    benchmark_data_root: Path,
    evaluation_subset: str,
    formal_cache_root: Path | None,
    heldout_subsets: list[str],
) -> dict[str, Any]:
    """Gate row-order/query coverage and explicit Starling held-out-parent exclusion."""
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
        if cache_key not in cache_parity and formal_cache_root is not None:
            cache_path = _formal_cache_path(
                formal_cache_root,
                task=_task_data_name(experiment.task),
                split=split,
                evaluation_subset=evaluation_subset,
            )
            cache_parity[cache_key] = _audit_formal_test_cache_parity(
                cache_path,
                query_store=query_store,
            )

        starling_key = (split, base_path)
        if experiment.source != "starling" or starling_key in checked_starling_pairs:
            continue
        heldout_path = heldout_labels_path(
            benchmark_data_root,
            _task_data_name(experiment.task),
            split,
            heldout_subsets,
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
        "formal_embedding_cache_parity": (
            "passed" if formal_cache_root is not None else "not_requested"
        ),
        "formal_test_embedding_cache_cosine": cache_parity,
        "starling_heldout_parent_exclusion": "passed",
        "heldout_subsets": heldout_subsets,
        "n_starling_index_split_pairs": len(checked_starling_pairs),
        "starling_parent_overlap_counts": starling_overlap_counts,
        "chembl_parent_exclusion": "enforced_at_query_time_by_parent_disjoint_policy",
    }


def _formal_cache_path(
    root: Path,
    *,
    task: str,
    split: str,
    evaluation_subset: str,
) -> Path:
    candidates = (
        root / task / split / f"{evaluation_subset}.pt",
        root / task / split / "embeddings" / f"{evaluation_subset}.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Missing formal MiniMol cache; checked: "
        + ", ".join(str(path) for path in candidates)
    )


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
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"MiniMol query store contains non-finite values: {embeddings_path}")
        finite_receipt_path = write_finite_array_receipt(
            embeddings_path,
            array=embeddings,
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
            "finite_receipt_path": str(finite_receipt_path),
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
    parser.add_argument(
        "--evaluation-subset",
        choices=("valid", "test"),
        default="test",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("bbb_martins", "bioavailability_ma", "skin_reaction"),
        default=["bbb_martins", "bioavailability_ma", "skin_reaction"],
    )
    parser.add_argument(
        "--benchmark-data-root",
        type=Path,
        default=Path("data/processed_starling"),
    )
    parser.add_argument("--benchmark-lineage", default="record_agreement70_split811_v1")
    parser.add_argument(
        "--heldout-subsets",
        nargs="+",
        choices=HELDOUT_SUBSETS,
        default=list(HELDOUT_SUBSETS),
        help="Subsets excluded from candidate indices; use test for train+valid test retrieval.",
    )
    parser.add_argument("--canonical-paper-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument(
        "--registry-source-root",
        type=Path,
        default=None,
        help="Reuse an audited superset MiniMol registry with identical checkpoint provenance.",
    )
    parser.add_argument(
        "--formal-cache-root",
        type=Path,
        default=Path("outputs/baselines/minimol_starling"),
        help="Optional baseline cache root used for exact query-embedding parity audit.",
    )
    parser.add_argument("--minimol-source", type=Path, default=DEFAULT_MINIMOL_SOURCE)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
