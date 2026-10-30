"""Differentiable upstream MiniMol fingerprints and reusable CPU graph caches."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing as mp
from pathlib import Path
import pickle
import random

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool

from baselines.minimol.embedding_runtime import create_featurizer, checkpoint_provenance
from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic

_TRANSFORM = None


def _graph(smiles: str):
    seed = int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    graph = _TRANSFORM(smiles)
    if isinstance(graph, str) or graph is None:
        raise ValueError(f"Graph featurization failed: {smiles}: {graph}")
    for key, value in graph.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float16:
                graph[key] = value.float()
            elif value.dtype == torch.int32:
                graph[key] = value.long()
    # Passing tensors directly creates one shared-memory handle per feature tensor.
    # Byte payloads keep large preprocessing pools below the system FD limit.
    return pickle.dumps(graph, protocol=5)


def prepare_graphs(data_path: Path, output: Path, workers: int = 64) -> None:
    global _TRANSFORM
    torch.set_num_threads(1)
    data = torch.load(data_path, weights_only=False)
    featurizer = create_featurizer(batch_size=32, device="cpu")
    _TRANSFORM = featurizer.datamodule.smiles_transformer
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        graphs = [pickle.loads(value) for value in pool.map(_graph, data["smiles"], chunksize=32)]
    torch.save(graphs, output)
    write_json_atomic(output.with_suffix(".json"), dict(
        version="minimol_graph_cache.v1", data_sha256=sha256_file(data_path),
        graph_sha256=sha256_file(output), n_graphs=len(graphs),
        max_atoms=max(g.num_nodes for g in graphs),
        seed="sha256(smiles) first 32 bits", checkpoint=checkpoint_provenance(),
    ))


class MiniMolEncoder(nn.Module):
    """Same gnn:15/max-pool representation as Minimol(), with gradients enabled."""

    def __init__(self, featurizer=None):
        super().__init__()
        featurizer = featurizer or create_featurizer(batch_size=32, device="cpu")
        self.network = featurizer.predictor.network
        assert len(self.network.gnn.layers) == 16, "Checkpoint fingerprint boundary changed"
        # Pretraining heads are unused. Avoid their compute and cached node activations.
        self.network.task_heads = None
        self.network._disable_readout_cache()

    def forward(self, graph: Batch) -> torch.Tensor:
        batch_index = graph.batch
        # Upstream layers write features in place; callers may reuse graph batches.
        output = self.network(graph.clone())
        return global_max_pool(output["feat"], batch_index)


class GraphRows(torch.utils.data.Dataset):
    def __init__(self, graphs, rows, auxiliary=False):
        self.graphs, self.rows, self.auxiliary = graphs, rows, auxiliary

    def __len__(self):
        return len(self.rows["indices"])

    def __getitem__(self, index):
        graph = self.graphs[int(self.rows["indices"][index])]
        if self.auxiliary:
            return graph, self.rows["buckets"][index], self.rows["targets"][index], self.rows["binary"][index]
        return graph, self.rows["conditions"][index], self.rows["labels"][index]


def collate_graphs(items):
    graphs, *values = zip(*items)
    return (Batch.from_data_list(list(graphs)), *(torch.stack(v) for v in values))


def auxiliary_loss(logits, targets, binary):
    """Only observed heads contribute; the batch estimates a task-balanced mean."""
    losses = torch.where(
        binary,
        nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none"),
        (logits - targets).square(),
    )
    return losses.mean()
