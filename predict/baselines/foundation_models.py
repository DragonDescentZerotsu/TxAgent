"""Official molecular encoders behind the shared paper training protocol."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch
from torch import nn
from rdkit import Chem

from predict.utils.json import atomic_output_path, sha256_file, write_json_atomic
from predict.baselines.foundation_cache import cached_input

MODELS = ("chemeleon", "unimol", "grover", "chemberta", "molformer", "smited", "kermt")
_BUILDER = None
_FEATURE_CACHE = None


def _feature(smiles):
    # Reproducible official random cropping, independent of process scheduling.
    seed = int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    random.seed(seed)
    path = None
    if _FEATURE_CACHE is not None:
        key = hashlib.sha256(smiles.encode()).hexdigest()
        path = _FEATURE_CACHE / key[:2] / (key + ".pkl")
        if path.exists():
            return pickle.loads(path.read_bytes())
    try:
        result = _BUILDER(smiles)
    except Exception as error:
        raise ValueError(f"Featurization failed for {smiles}: {error}") from error
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_bytes(pickle.dumps(result, protocol=5))
        temporary.replace(path)
    return result


class FoundationEncoder(nn.Module):
    """Rows index a pinned input cache; only the encoder carries trainable weights."""

    def __init__(self, name, runtime_root, cache, smiles, *, frozen=False, prepare=False):
        super().__init__()
        if name not in MODELS:
            raise ValueError(f"Unknown foundation model: {name}")
        self.name, self.cache, self.smiles = name, Path(cache), list(smiles)
        self.register_buffer("device_marker", torch.empty(0), persistent=False)
        runtime_root = Path(runtime_root)
        source = json.loads((runtime_root / "manifest.json").read_text())["models"][name]
        self.weights = runtime_root / "weights" / name
        self.frozen = frozen
        if not prepare:
            receipt = json.loads((self.cache / "receipt.json").read_text())
            if receipt["binding"]["model"] != name or receipt["binding"]["source"] != source:
                raise ValueError("Feature cache uses a different pretrained release")
        if frozen:
            payload = torch.load(cached_input(self.cache / "embeddings.pt", receipt["files"]["embeddings.pt"]),
                                 map_location="cpu", weights_only=False)
            if payload["smiles"] != self.smiles:
                raise ValueError("Frozen embeddings are not aligned with prepared molecules")
            self.register_buffer("embeddings", payload["embeddings"], persistent=False)
            self.output_dim = self.embeddings.shape[1]
            return

        self.weights = cached_input(self.weights, source)
        # Cached features are assembled on CPU; avoid a CUDA -> CPU index round trip.
        self.input_device = torch.device("cpu")

        if name in ("chemberta", "molformer"):
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.weights, trust_remote_code=name == "molformer")
            options = dict(trust_remote_code=True, deterministic_eval=True) if name == "molformer" else dict(add_pooling_layer=False)
            self.network, info = AutoModel.from_pretrained(self.weights, output_loading_info=True, **options)
            if info["missing_keys"] or info["mismatched_keys"]:
                raise ValueError(f"Uninitialized encoder weights: {info}")
            self.output_dim = self.network.config.hidden_size
        elif name == "chemeleon":
            from chemprop import featurizers
            from chemprop.nn import BondMessagePassing, MeanAggregation
            self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
            checkpoint = torch.load(self.weights / "chemeleon_mp.pt", map_location="cpu", weights_only=True)
            self.network = BondMessagePassing(**checkpoint["hyper_parameters"])
            self.network.load_state_dict(checkpoint["state_dict"], strict=True)
            self.readout = MeanAggregation()
            self.output_dim = self.network.output_dim
        elif name == "unimol":
            os.environ["UNIMOL_WEIGHT_DIR"] = str(self.weights)
            sys.path.insert(0, str(runtime_root / "repos/unimol/unimol_tools"))
            from unimol_tools.models.unimolv2 import UniMolV2Model
            from unimol_tools.data.conformer import UniMolV2Feature
            self.network = UniMolV2Model(output_dim=1, model_size="84m")
            self.network.classification_head = nn.Identity()
            state = torch.load(self.weights / "modelzoo/84M/checkpoint.pt", map_location="cpu", weights_only=False)["model"]
            missing, _ = self.network.load_state_dict(state, strict=False)
            if missing:
                raise ValueError(f"Uninitialized Uni-Mol encoder: {missing}")
            if not prepare:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
                # Preserve the FP32 batch-32 objective while bounding pair-feature
                # activation memory; PyTorch preserves dropout RNG on recompute.
                self.network.encoder.layers = nn.ModuleList(
                    [checkpoint_wrapper(layer, preserve_rng_state=True) for layer in self.network.encoder.layers])
            self.featurizer = UniMolV2Feature(seed=42, mode="fast", remove_hs=True, max_atoms=128, multi_process=False)
            self.output_dim = 768
        elif name in ("grover", "kermt"):
            sys.path.insert(0, str(runtime_root / "repos" / name))
            if name == "grover":
                from grover.model.models import GroverFpGeneration as Fingerprint
                from grover.data.molgraph import MolGraph, BatchMolGraph
                filename = "grover_base.pt"
            else:
                from kermt.model.models import KermtFpGeneration as Fingerprint
                from kermt.data.molgraph import MolGraph, BatchMolGraph
                filename = "kermt_contrastive_v2.0.pt"
            checkpoint = torch.load(self.weights / filename, map_location="cpu", weights_only=False)
            self.args = checkpoint["args"]
            self.args.cuda = False
            self.args.dropout = getattr(self.args, "dropout", 0.0)
            self.args.fingerprint_source = "both"
            self.args.no_cache = False
            self.network = Fingerprint(self.args)
            state = {k: v for k, v in checkpoint["state_dict"].items() if k in self.network.state_dict()}
            missing, _ = self.network.load_state_dict(state, strict=False)
            if any(not k.startswith("readout.") for k in missing):
                raise ValueError(f"Uninitialized {name} encoder: {missing}")
            self.MolGraph, self.BatchMolGraph = MolGraph, BatchMolGraph
            self.output_dim = 4 * self.args.hidden_size
        else:
            path = self.weights / "smi-ted/finetune/smi_ted_light/load.py"
            spec = importlib.util.spec_from_file_location("txagent_smited", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.network = module.load_smi_ted(folder=str(path.parent),
                                             ckpt_filename=str(self.weights / "smi-ted-Light_40.pt"), eval=True)
            self.network.net = nn.Identity()
            self.output_dim = self.network.n_embd
            # Only encoder and the pretrained compression layer define this representation.
            self.network.decoder.autoencoder.decoder = nn.Identity()
            self.network.encoder.lang_model = nn.Identity()
            if hasattr(self.network.decoder, "lang_model"):
                self.network.decoder.lang_model = nn.Identity()
            state = torch.load(self.weights / "smi-ted-Light_40.pt", map_location="cpu", weights_only=False)["MODEL_STATE"]
            missing, _ = self.network.load_state_dict(state, strict=False)
            if missing:
                raise ValueError(f"Uninitialized SMI-TED encoder: {missing}")

        if not prepare:
            payload = torch.load(cached_input(self.cache / "features.pt", receipt["files"]["features.pt"]),
                                 map_location="cpu", weights_only=False)
            if payload["smiles"] != self.smiles or payload["model"] != name:
                raise ValueError("Feature cache molecule/model identity mismatch")
            self.features = payload["features"]

    def make_feature(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid benchmark SMILES: {smiles}")
        if self.name == "chemeleon":
            return self.featurizer(mol)
        if self.name == "unimol":
            from unimol_tools.data.conformer import inner_smi2coords, mol2unimolv2
            mol = inner_smi2coords(smiles, seed=42, mode="fast", remove_hs=True, return_mol=True)
            restored = False
            try:
                result = mol2unimolv2(mol, max_atoms=128, remove_hs=True)
            except Chem.KekulizeException:
                # MMFF may change aromatic flags. Restore the valid input graph,
                # retaining the official coordinates and identical atom ordering.
                original = Chem.AddHs(Chem.MolFromSmiles(smiles))
                if [a.GetAtomicNum() for a in original.GetAtoms()] != [a.GetAtomicNum() for a in mol.GetAtoms()]:
                    raise ValueError("Cannot restore MMFF aromaticity: atom ordering changed")
                original.AddConformer(Chem.Conformer(mol.GetConformer()))
                result = mol2unimolv2(original, max_atoms=128, remove_hs=True)
                restored = True
            return result, dict(is_3d=bool(mol.GetConformer().Is3D()),
                                aromaticity_restored=restored,
                                cropped=mol.GetNumHeavyAtoms() > 128,
                                hydrogen_only=mol.GetNumHeavyAtoms() == 0,
                                timed_out=mol.HasProp("_txagent_embedding_timeout") and mol.GetBoolProp("_txagent_embedding_timeout"))
        if self.name in ("grover", "kermt"):
            graph = self.MolGraph(smiles, self.args)
            # RingInfo is a non-pickleable construction helper; atom/bond features
            # already contain its outputs and BatchMolGraph never reads it.
            del graph.ring_info
            return graph
        canonical = Chem.MolToSmiles(mol, isomericSmiles=self.name != "molformer")
        if self.name == "smited":
            return canonical
        tokens = self.tokenizer(canonical)
        limit = 512 if self.name == "chemberta" else None
        if limit and len(tokens["input_ids"]) > limit:
            return self.tokenizer(canonical, truncation=True, max_length=limit)
        return tokens

    def forward(self, indices):
        if self.frozen:
            return self.embeddings[indices]
        ids = indices.detach().cpu().tolist()
        records = [self.features[i] for i in ids]
        device = self.device_marker.device
        if self.name in ("chemberta", "molformer"):
            inputs = self.tokenizer.pad(records, padding=True, return_tensors="pt").to(device)
            output = self.network(**inputs)
            if self.name == "molformer":
                return output.pooler_output
            # MTR's representation is the first token, before its descriptor head.
            return output.last_hidden_state[:, 0]
        if self.name == "chemeleon":
            from chemprop.data import BatchMolGraph
            batch = BatchMolGraph(records)
            batch.to(device)
            return self.readout(self.network(batch), batch.batch)
        if self.name == "unimol":
            self.network.dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else torch.float32
            batch, _ = self.network.batch_collate_fn([(r[0], 0) for r in records])
            return self.network(**{k: v.to(device) for k, v in batch.items()}, return_repr=True)["cls_repr"]
        if self.name in ("grover", "kermt"):
            batch = self.BatchMolGraph(records, self.args).get_components()
            batch = tuple(v.to(device) if isinstance(v, torch.Tensor) else v for v in batch)
            return self.network(batch, [None] * len(ids))
        return self.network.extract_embeddings(records)


def conformer_cache(source, scratch):
    version = dict(source=source, seed=42, mode="fast", remove_hs=True, max_atoms=128,
                   crop_seed="sha256(smiles) first 32 bits", format=1)
    key = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()
    return Path(scratch) / key


def warm_conformers(runtime_root, run_root, scratch, workers, partition, partitions):
    """Precompute disjoint SMILES partitions on CPU while GPU training continues."""
    global _BUILDER, _FEATURE_CACHE
    source = json.loads((Path(runtime_root) / "manifest.json").read_text())["models"]["unimol"]
    smiles = sorted({s for path in (Path(run_root) / "inputs").glob("*/*/data.pt")
                     for s in torch.load(path, weights_only=False)["smiles"]})
    smiles = [s for s in smiles if int(hashlib.sha256(s.encode()).hexdigest(), 16) % partitions == partition]
    model = FoundationEncoder("unimol", runtime_root, scratch, smiles, prepare=True)
    _BUILDER, _FEATURE_CACHE = model.make_feature, conformer_cache(source, scratch)
    missing = [s for s in smiles if not (_FEATURE_CACHE / hashlib.sha256(s.encode()).hexdigest()[:2] /
                                       (hashlib.sha256(s.encode()).hexdigest()+".pkl")).exists()]
    completed, errors = len(smiles) - len(missing), []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = {pool.submit(_feature, s): s for s in missing}
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
            except Exception as error:
                errors.append(dict(smiles=futures[future], error=str(error)))
            receipt = dict(partition=partition, partitions=partitions, completed=completed, total=len(smiles),
                           workers=workers, errors=errors)
            write_json_atomic(Path(run_root) / "conformers" / f"partition_{partition}.json", receipt)
    receipt = dict(partition=partition, partitions=partitions, completed=completed, total=len(smiles), workers=workers, errors=errors)
    write_json_atomic(Path(run_root) / "conformers" / f"partition_{partition}.json", receipt)
    print(json.dumps(receipt), flush=True)
    if errors:
        raise ValueError("Some conformers failed; inspect the partition receipt")


def prepare_features(name, runtime_root, prepared, cache, workers=32, device="cuda", batch_size=128, scratch=None,
                     features_only=False):
    global _BUILDER, _FEATURE_CACHE
    cache = Path(cache)
    data_path = Path(prepared) / "data.pt"
    data_sha = json.loads((Path(prepared) / "manifest.json").read_text())["data_sha256"]
    local_data = cached_input(data_path, data_sha)
    if sha256_file(local_data) != data_sha:
        raise ValueError("Prepared target cache hash changed")
    data = torch.load(local_data, map_location="cpu", weights_only=False)
    source = json.loads((Path(runtime_root) / "manifest.json").read_text())["models"][name]
    binding = dict(model=name, source=source, data_sha256=data_sha)
    if (cache / "receipt.json").exists():
        old = json.loads((cache / "receipt.json").read_text())
        if old["binding"] != binding:
            raise ValueError("Existing feature cache has different inputs/model")
        if features_only:
            write_json_atomic(cache / "features_ready.json", dict(binding=binding, sha256=old["files"]["features.pt"]))
            return
        for file, digest in old["files"].items():
            if sha256_file(cache / file) != digest:
                raise ValueError(f"Changed feature cache: {file}")
        return
    cache.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    model = FoundationEncoder(name, runtime_root, cache, data["smiles"], prepare=True)
    _BUILDER = model.make_feature
    _FEATURE_CACHE = None
    if name == "unimol" and scratch is not None:
        # Conformers are label-free and expensive; share identical molecular
        # inputs across the 18 cohorts and retain work after an interruption.
        _FEATURE_CACHE = conformer_cache(source, scratch)
    with (cache / "preprocessing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = cache / "features_ready.json"
        if ready.exists():
            receipt = json.loads(ready.read_text())
            if receipt["binding"] != binding:
                raise ValueError("Prepared molecular features use different inputs")
            local_features = cached_input(cache / "features.pt", receipt["sha256"])
            if sha256_file(local_features) != receipt["sha256"]:
                raise ValueError("Prepared molecular feature bytes changed")
            payload = torch.load(local_features, map_location="cpu", weights_only=False)
            if payload["smiles"] != data["smiles"] or payload["model"] != name:
                raise ValueError("Molecular feature identities changed")
            features = payload["features"]
        else:
            if name in ("chemberta", "molformer", "smited"):
                features = [model.make_feature(s) for s in data["smiles"]]
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
                    features = []
                    for result in pool.map(_feature, data["smiles"], chunksize=16):
                        features.append(result)
                        if len(features) % 1000 == 0:
                            print(json.dumps(dict(model=name, prepared_molecules=len(features), total=len(data["smiles"]))), flush=True)
            with atomic_output_path(cache / "features.pt") as temporary:
                torch.save(dict(smiles=data["smiles"], model=name, features=features), temporary)
            write_json_atomic(ready, dict(binding=binding, sha256=sha256_file(cache / "features.pt")))
    if features_only:
        return
    model.features = features
    model.to(device).eval()
    # Auxiliary-only molecules have no role in the frozen-head arm.
    main_ids = sorted({int(i) for rows in data["splits"].values() for i in rows["indices"]})
    embeddings = torch.zeros(len(data["smiles"]), model.output_dim)
    with torch.no_grad():
        for start in range(0, len(main_ids), batch_size):
            ids = torch.tensor(main_ids[start:start + batch_size])
            values = model(ids).float().cpu()
            if values.shape != (len(ids), model.output_dim) or not torch.isfinite(values).all():
                raise ValueError("Invalid molecular embeddings")
            embeddings[ids.cpu()] = values
    torch.save(dict(smiles=data["smiles"], embeddings=embeddings, main_indices=main_ids), cache / "embeddings.pt")
    diagnostics = {}
    if name in ("chemberta", "smited"):
        tokenizer = model.tokenizer if name == "chemberta" else model.network.tokenizer
        limit = 512 if name == "chemberta" else model.network.max_len
        truncated = [i for i, s in enumerate(data["smiles"])
                     if len(tokenizer(Chem.MolToSmiles(Chem.MolFromSmiles(s)))["input_ids"]) > limit]
        diagnostics = dict(max_tokens=limit, truncated_indices=truncated, rows_dropped=0)
    if name == "unimol":
        diagnostics = dict(two_dimensional=sum(not r[1]["is_3d"] for r in features),
                           zero_coordinates=sum(not np.any(r[0]["src_coord"]) for r in features),
                           hydrogen_only=sum(r[1].get("hydrogen_only", False) for r in features),
                           aromaticity_restored=sum(r[1].get("aromaticity_restored", False) for r in features),
                           three_dimensional_timeout_seconds=300,
                           timed_out_indices=[i for i, r in enumerate(features) if r[1].get("timed_out", False)],
                           cropped=sum(r[1]["cropped"] for r in features),
                           conformer_seed=42, max_atoms=128, mode="fast", remove_hs=True,
                           crop_seed="sha256(original SMILES) first 32 bits")
    write_json_atomic(cache / "receipt.json", dict(binding=binding, output_dim=model.output_dim,
                      molecules=len(features), frozen_molecules=len(main_ids), diagnostics=diagnostics,
                      files={f: sha256_file(cache / f) for f in ("features.pt", "embeddings.pt")}))
