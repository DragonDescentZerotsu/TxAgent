"""Shared MiniMol embedding runtime used by baselines and retrieval builders."""

from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


DEFAULT_MINIMOL_SOURCE = Path(
    os.environ.get("TXAGENT_MINIMOL_SOURCE", Path(__file__).resolve().parents[3] / "minimol")
).expanduser()


def ensure_minimol_import(minimol_source: Path = DEFAULT_MINIMOL_SOURCE) -> None:
    if importlib.util.find_spec("minimol") is not None:
        return
    if minimol_source.exists():
        sys.path.insert(0, str(minimol_source))


def patch_graphium_float32_featurization() -> None:
    """Avoid SciPy sparse float16 failures in Graphium's CPU featurization path."""
    from scipy.sparse import coo_matrix

    import graphium.data.datamodule as graphium_datamodule
    import graphium.features as graphium_features
    import graphium.features.featurizer as graphium_featurizer
    import graphium.features.nmp as graphium_nmp

    if getattr(graphium_featurizer.mol_to_pyggraph, "_txagent_float32_patch", False):
        return

    original_mol_to_pyggraph = graphium_featurizer.mol_to_pyggraph

    def mol_to_adjacency_matrix_float32(
        mol: Any,
        use_bonds_weights: bool = False,
        add_self_loop: bool = False,
        dtype: Any = np.float32,
    ) -> Any:
        del dtype
        adj_idx = []
        adj_val = []
        for bond in mol.GetBonds():
            adj_idx.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            adj_idx.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            val = graphium_nmp.BOND_TYPES[bond.GetBondType()] if use_bonds_weights else 1.0
            adj_val.extend([val, val])

        if adj_val:
            data = np.asarray(adj_val, dtype=np.float32)
            coords = np.asarray(adj_idx, dtype=np.int64).T.reshape(2, -1)
            adj = coo_matrix(
                (data, coords),
                shape=(mol.GetNumAtoms(), mol.GetNumAtoms()),
                dtype=np.float32,
            )
        else:
            adj = coo_matrix(
                ([], np.array([[], []])),
                shape=(mol.GetNumAtoms(), mol.GetNumAtoms()),
                dtype=np.float32,
            )

        if add_self_loop:
            arange = np.arange(adj.shape[0], dtype=int)
            adj[arange, arange] = 1
        return adj

    def mol_to_pyggraph_float32(*args: Any, **kwargs: Any) -> Any:
        kwargs["dtype"] = np.float32
        return original_mol_to_pyggraph(*args, **kwargs)

    mol_to_pyggraph_float32._txagent_float32_patch = True
    graphium_featurizer.mol_to_adjacency_matrix = mol_to_adjacency_matrix_float32
    graphium_featurizer.mol_to_pyggraph = mol_to_pyggraph_float32
    graphium_features.mol_to_pyggraph = mol_to_pyggraph_float32
    graphium_datamodule.mol_to_pyggraph = mol_to_pyggraph_float32


def create_featurizer(
    *,
    batch_size: int,
    minimol_source: Path = DEFAULT_MINIMOL_SOURCE,
    device: str | torch.device | None = None,
) -> Any:
    """Load MiniMol once with the compatibility patches required on this host."""
    ensure_minimol_import(minimol_source)
    patch_graphium_float32_featurization()
    from hydra.core.global_hydra import GlobalHydra
    from minimol import Minimol

    original_torch_load = torch.load

    def torch_load_weights_compatible(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        torch.load = torch_load_weights_compatible
        featurizer = Minimol(batch_size=batch_size)
    finally:
        torch.load = original_torch_load
    featurizer.datamodule.featurization_n_jobs = 1
    if device is not None:
        _place_featurizer_on_device(featurizer, torch.device(device))
    return featurizer


def _place_featurizer_on_device(featurizer: Any, device: torch.device) -> None:
    """Move the upstream MiniMol predictor and each graph batch together."""
    if device.type == "cpu":
        return
    fingerprinter = featurizer.predictor
    if fingerprinter.predictor is None:
        raise RuntimeError("MiniMol Fingerprinter has no PredictorModule")
    fingerprinter.predictor.to(device)
    original_get_fingerprints = fingerprinter.get_fingerprints_for_batch

    def get_fingerprints_on_device(batch: dict[str, Any]) -> torch.Tensor:
        device_batch = dict(batch)
        device_batch["features"] = device_batch["features"].to(device)
        return original_get_fingerprints(device_batch)

    fingerprinter.get_fingerprints_for_batch = get_fingerprints_on_device
    featurizer._txagent_device = str(device)


def embed_smiles(featurizer: Any, smiles: Sequence[str]) -> torch.Tensor:
    """Return one finite CPU float32 embedding row per input SMILES."""
    with torch.no_grad():
        embeddings = featurizer(list(smiles))
    if len(embeddings) != len(smiles):
        raise RuntimeError(
            f"MiniMol returned {len(embeddings)} embeddings for {len(smiles)} molecules"
        )
    tensor = torch.stack([embedding.detach().cpu().float() for embedding in embeddings])
    if tensor.ndim != 2 or not torch.isfinite(tensor).all():
        raise RuntimeError("MiniMol returned invalid embeddings")
    if torch.any(torch.linalg.vector_norm(tensor, dim=1) == 0):
        raise RuntimeError("MiniMol returned a zero-norm embedding")
    return tensor


def checkpoint_provenance(
    minimol_source: Path = DEFAULT_MINIMOL_SOURCE,
) -> dict[str, Any]:
    """Return hashes for the installed MiniMol v1 weights and resolved configs."""
    ensure_minimol_import(minimol_source)
    import minimol

    checkpoint_dir = Path(minimol.__file__).resolve().parent / "ckpts" / "minimol_v1"
    files = {
        name: checkpoint_dir / name
        for name in ("state_dict.pth", "config.yaml", "base_shape.yaml")
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MiniMol checkpoint artifacts: " + ", ".join(missing))
    return {
        "model": "MiniMol",
        "model_version": "minimol_v1",
        "checkpoint_dir": str(checkpoint_dir),
        "files": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for name, path in files.items()
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
