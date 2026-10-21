"""Prepare parent-disjoint, bucket-local targets without source-quality filtering."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from predict.baselines.minimol.condition_features import condition_one_hot, condition_vocabulary
from predict.utils.json import sha256_file, write_json_atomic
from predict.retrieval.policies import normalize_molecule_identity


def parent_keys(smiles: str) -> tuple[str, str]:
    identity = normalize_molecule_identity(smiles)
    if identity.status != "ok":
        raise ValueError(f"Unresolved parent: {smiles}")
    if identity.parent_smiles:
        return identity.parent_smiles, identity.parent_connectivity_key
    # Uncharging some valid ions (e.g. PF6-) creates an invalid valence. Keep
    # their charge, using the same fragment-parent rule for main and aux rows.
    parent = rdMolStandardize.FragmentParent(Chem.MolFromSmiles(identity.canonical_smiles))
    Chem.SanitizeMol(parent)
    key = Chem.MolToInchiKey(parent)
    if not key:
        raise ValueError(f"Unresolved parent: {smiles}")
    return Chem.MolToSmiles(parent), key.split("-", 1)[0]


def aggregate_targets(rows: list[dict], heldout: set[str], identities: dict) -> tuple[list, list, dict]:
    """Filter identities before aggregating or fitting any normalization statistics."""
    groups = defaultdict(list)
    counts = Counter(input_rows=len(rows))
    for row in rows:
        smiles = row["parent_smiles"]
        parent, connectivity = identities[smiles]
        if parent in heldout or (connectivity and connectivity in heldout):
            counts["heldout_rows_removed"] += 1
            continue
        values = [float(v) for v in row["canonical_measurements"]]
        if not values or not np.isfinite(values).all():
            raise ValueError(f"Invalid measurements for {smiles}: {row['pair_bucket_key']}")
        # Renormalized equivalent SMILES must remain one molecule-bucket unit.
        groups[(parent, row["pair_bucket_key"])].extend(values)
    by_bucket = defaultdict(list)
    for (parent, key), values in groups.items():
        binary = json.loads(key)[2] == "binary_outcome_class"
        if binary and not set(values) <= {-1.0, 1.0}:
            raise ValueError(f"Unexpected binary domain: {key}")
        target = float(np.mean(np.asarray(values) == 1)) if binary else float(np.median(values))
        by_bucket[key].append((parent, target, len(values)))
    targets, buckets = [], []
    for key in sorted(by_bucket):
        entries = sorted(by_bucket[key])
        binary = json.loads(key)[2] == "binary_outcome_class"
        values = np.asarray([entry[1] for entry in entries], dtype=np.float64)
        mean = 0.0 if binary else float(values.mean())
        std = 1.0 if binary else float(values.std())
        constant = std == 0.0 if not binary else len(set(values)) == 1
        # Retain constant and small buckets too; never divide by zero.
        scale = std if std > 0 else 1.0
        bucket_id = len(buckets)
        buckets.append(dict(key=key, binary=binary, mean=mean, scale=scale,
                            n_parents=len(entries), constant=constant))
        for parent, value, n in entries:
            targets.append(dict(smiles=parent, bucket=bucket_id,
                                target=(value - mean) / scale, raw_target=value,
                                measurements=n, binary=binary))
    counts.update(output_rows=len(targets), buckets=len(buckets),
                  parents=len({r["smiles"] for r in targets}))
    return targets, buckets, dict(counts)


def prepare(data_dir: Path, auxiliary: Path, output: Path, workers: int = 64, *,
            condition_field: str | None = "condition_group", preserve_main_split: bool = False) -> dict:
    """Freeze row order, train-only conditions, targets and exact input hashes."""
    output.mkdir(parents=True, exist_ok=True)
    splits = {s: [json.loads(line) for line in (data_dir / f"{s}.jsonl").read_text().splitlines()]
              for s in ("train", "valid", "test")}
    if any(not records or any(row["Y"] not in (0, 1) for row in records) for records in splits.values()):
        raise ValueError("Main-task splits must be nonempty binary classification datasets")
    rows = pq.read_table(auxiliary).to_pylist()
    all_smiles = sorted({r["parent_smiles"] for r in rows} |
                        {r["drug"] for split in splits.values() for r in split})
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        identities = dict(zip(all_smiles, pool.map(parent_keys, all_smiles, chunksize=64)))
    heldout = {key for s in ("valid", "test") for r in splits[s]
               for key in identities[r["drug"]] if key}
    train_keys = {key for r in splits["train"] for key in identities[r["drug"]] if key}
    overlap = heldout & train_keys
    if overlap and not preserve_main_split:
        raise ValueError("Benchmark train/heldout parent overlap")
    targets, buckets, counts = aggregate_targets(rows, heldout, identities)
    if not targets:
        raise ValueError("No auxiliary targets remain after heldout-parent exclusion")
    vocabulary = condition_vocabulary([r[condition_field] for r in splits["train"]]) if condition_field else []
    # Main-task molecular surfaces remain byte-for-byte those of the benchmark.
    molecules = sorted({r["drug"] for split in splits.values() for r in split} |
                       {r["smiles"] for r in targets})
    indices = {s: i for i, s in enumerate(molecules)}
    payload = dict(smiles=molecules, buckets=buckets, conditions=vocabulary, splits={})
    for s, records in splits.items():
        payload["splits"][s] = dict(
            indices=torch.tensor([indices[r["drug"]] for r in records]),
            labels=torch.tensor([r["Y"] for r in records], dtype=torch.float32),
            conditions=(condition_one_hot([r[condition_field] for r in records], vocabulary)
                        if condition_field else torch.zeros((len(records), 0))),
            condition_groups=[r.get("condition_group") or "no_reported_external_condition" for r in records],
        )
    payload["auxiliary"] = dict(
        indices=torch.tensor([indices[r["smiles"]] for r in targets]),
        buckets=torch.tensor([r["bucket"] for r in targets]),
        targets=torch.tensor([r["target"] for r in targets], dtype=torch.float32),
        binary=torch.tensor([r["binary"] for r in targets]),
    )
    torch.save(payload, output / "data.pt")
    write_json_atomic(output / "buckets.json", buckets)
    manifest = dict(
        version="minimol_indirect_data.v1", data_dir=str(data_dir), auxiliary=str(auxiliary),
        inputs={str(p): sha256_file(p) for p in [auxiliary, *[data_dir / f"{s}.jsonl" for s in splits]]},
        counts=counts, split_sizes={s: len(rs) for s, rs in splits.items()},
        graph_molecules=len(molecules), conditions=vocabulary,
        condition_field=condition_field, preserve_main_split=preserve_main_split,
        main_heldout_identity_overlap=len(overlap),
        exclusion="valid+test normalized parent SMILES or parent connectivity; no scaffold exclusion",
        identity_fallback="charge-preserving fragment parent if uncharging fails for a valid molecule",
        filters="no source, voter, quality, support-count or variance filtering",
        aggregation="parent x bucket: median continuous; positive fraction binary; equal unit weight",
        normalization="per-bucket population mean/std of aggregated auxiliary training targets; no log transform",
        data_sha256=sha256_file(output / "data.pt"),
    )
    write_json_atomic(output / "manifest.json", manifest)
    return manifest
