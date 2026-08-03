"""Feature backends for analog retrieval.

Morgan/Tanimoto remains the default. A descriptor index can attach precomputed,
L2-normalized MiniMol candidate and query embeddings without copying the
evidence library or loading MiniMol inside reasoning workers.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np
from rdkit import DataStructs

from tools.chembl_tool.common.neighbor_selection import QUERY_FEATURE_COVERAGE_SELECTOR


DESCRIPTOR_TYPE = "retrieval_feature_index.v1"
MORGAN_FEATURE = "morgan_fingerprint"
MINIMOL_FEATURE = "minimol_embedding"
MINIMOL_SIMILARITY = "cosine"
_RUNTIME_KEY = "_retrieval_feature_runtime"


def load_retrieval_index(path: Path) -> dict[str, Any]:
    """Load a legacy pickle index or a lightweight feature descriptor."""
    if path.suffix.lower() != ".json":
        with path.open("rb") as handle:
            return pickle.load(handle)

    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if descriptor.get("type") != DESCRIPTOR_TYPE:
        raise ValueError(f"Unsupported retrieval index descriptor: {path}")
    base_index_path = _resolve_path(str(descriptor["base_index_path"]), descriptor_path=path)
    with base_index_path.open("rb") as handle:
        index = pickle.load(handle)
    candidate_path = _resolve_path(
        str(descriptor["candidate_embeddings_path"]),
        descriptor_path=path,
    )
    query_path = _resolve_path(
        str(descriptor["query_embeddings_path"]),
        descriptor_path=path,
    )
    query_manifest_path = _resolve_path(
        str(descriptor["query_manifest_path"]),
        descriptor_path=path,
    )
    runtime = {
        "feature": MINIMOL_FEATURE,
        "model": str(descriptor.get("model") or "MiniMol"),
        "model_version": str(descriptor.get("model_version") or "minimol_v1"),
        "normalization": "L2",
        "similarity": MINIMOL_SIMILARITY,
        "candidate_embeddings_path": str(candidate_path),
        "query_embeddings_path": str(query_path),
        "query_manifest_path": str(query_manifest_path),
        "candidate_order_sha256": str(descriptor["candidate_order_sha256"]),
    }
    _validate_runtime(index, runtime)
    index[_RUNTIME_KEY] = runtime
    return index


def retrieval_feature_metadata(index: Mapping[str, Any]) -> dict[str, Any]:
    runtime = index.get(_RUNTIME_KEY)
    if runtime:
        return {
            key: runtime[key]
            for key in ("feature", "model", "model_version", "normalization", "similarity")
        }
    return {
        "feature": MORGAN_FEATURE,
        "normalization": "binary_bits",
        "similarity": "tanimoto",
        "fingerprint": dict(index.get("fingerprint") or {}),
    }


def similarity_vector(
    query_fingerprint: Any,
    query_canonical_smiles: str,
    index: Mapping[str, Any],
    *,
    neighbor_selector: str,
) -> np.ndarray:
    """Return one similarity per evidence-library molecule in stable index order."""
    runtime = index.get(_RUNTIME_KEY)
    if runtime is None:
        return np.asarray(
            DataStructs.BulkTanimotoSimilarity(query_fingerprint, index["fingerprints"]),
            dtype=np.float32,
        )
    if neighbor_selector == QUERY_FEATURE_COVERAGE_SELECTOR:
        raise ValueError(
            "query_feature_coverage is Morgan-bit-specific and is not supported "
            "with MiniMol embedding retrieval"
        )

    candidate_embeddings = _load_array(str(runtime["candidate_embeddings_path"]))
    query_embeddings = _load_array(str(runtime["query_embeddings_path"]))
    query_rows = _load_query_rows(str(runtime["query_manifest_path"]))
    try:
        query_row = query_rows[query_canonical_smiles]
    except KeyError as exc:
        raise KeyError(
            "MiniMol retrieval query embedding is missing for canonical SMILES "
            f"{query_canonical_smiles!r}"
        ) from exc
    query_embedding = np.asarray(query_embeddings[query_row], dtype=np.float32)
    similarities = np.asarray(candidate_embeddings @ query_embedding, dtype=np.float32)
    return np.clip(similarities, -1.0, 1.0)


def similarity_bucket_for_index(similarity: float, index: Mapping[str, Any]) -> str:
    """Keep Morgan structural buckets and label MiniMol scores without overclaiming."""
    if index.get(_RUNTIME_KEY):
        return "minimol_embedding_cosine_not_structural_similarity"
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


def candidate_order_sha256(index: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for molecule in index["molecules"]:
        digest.update(str(molecule.get("molecule_chembl_id") or "").encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(molecule.get("canonical_smiles") or "").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_runtime(index: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    expected_order = candidate_order_sha256(index)
    if expected_order != runtime["candidate_order_sha256"]:
        raise ValueError("MiniMol candidate embeddings do not match the base index molecule order")
    candidate_embeddings = _load_array(str(runtime["candidate_embeddings_path"]))
    query_embeddings = _load_array(str(runtime["query_embeddings_path"]))
    if candidate_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("MiniMol retrieval embedding stores must be rank-2 arrays")
    if candidate_embeddings.shape[0] != len(index["molecules"]):
        raise ValueError(
            "MiniMol candidate embedding count does not match the base index: "
            f"{candidate_embeddings.shape[0]} != {len(index['molecules'])}"
        )
    if candidate_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("MiniMol candidate and query embedding dimensions differ")
    query_rows = _load_query_rows(str(runtime["query_manifest_path"]))
    if query_rows and max(query_rows.values()) >= query_embeddings.shape[0]:
        raise ValueError("MiniMol query manifest references a missing embedding row")


@lru_cache(maxsize=32)
def _load_array(path: str) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if not np.isfinite(array).all():
        raise ValueError(f"Retrieval embedding store contains non-finite values: {path}")
    return array


@lru_cache(maxsize=32)
def _load_query_rows(path: str) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("canonical_smiles_to_row")
    if not isinstance(rows, dict):
        raise ValueError(f"Invalid MiniMol query manifest: {path}")
    return {str(smiles): int(row) for smiles, row in rows.items()}


def _resolve_path(value: str, *, descriptor_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return descriptor_path.parent / path
