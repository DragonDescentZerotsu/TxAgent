"""Prepare HF assay-transfer features for a simple trained MLP baseline.

The input format is the prompt/completion/metadata JSONL used by the
activity-transfer LLM benchmark. This script extracts endpoint text and two
molecule SMILES strings, computes endpoint semantic embeddings, computes
deduplicated RDKit molecule features, and writes compact index arrays for
training/evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_HF_REPO = "jiosephlee/proper_assay_transfer_no_prop_no_tanimoto"
DEFAULT_VALIDATION_JSONL = (
    "outputs/chembl_tool/activity_transfer_benchmark/hf_jiosephlee_valid20k/"
    "proper_assay_transfer_no_prop_no_tanimoto/validation.jsonl"
)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/"
    "qwen3_embedding_rdkit_v1/preprocessed"
)
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

LABEL_TO_ID = {"different": 0, "similar": 1}
COMPLETION_TO_LABEL = {"A": "similar", "B": "different"}

ENDPOINT_RE = re.compile(
    r"## Endpoint\s*(?P<endpoint>.*?)\s*## Molecule A\b",
    flags=re.DOTALL,
)
MOLECULE_A_RE = re.compile(
    r"## Molecule A\b.*?^\s*-\s*SMILES:\s*(?P<smiles>\S+)",
    flags=re.DOTALL | re.MULTILINE,
)
MOLECULE_B_RE = re.compile(
    r"## Molecule B\b.*?^\s*-\s*SMILES:\s*(?P<smiles>\S+)",
    flags=re.DOTALL | re.MULTILINE,
)


@dataclass(frozen=True)
class ParsedPrompt:
    endpoint_text: str
    molecule_a_smiles: str
    molecule_b_smiles: str


@dataclass(frozen=True)
class CleanRow:
    split: str
    source_index: int
    label: str
    endpoint_key: str
    endpoint_text: str
    endpoint_text_hash: str
    molecule_a_smiles: str
    molecule_b_smiles: str
    metadata: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    out_dir = Path(args.out_dir)
    ensure_output_dirs(out_dir)

    log_stage("load_train_start", source=args.train_jsonl or args.hf_repo, split=args.train_split)
    if args.train_jsonl:
        train_records = iter_jsonl(Path(args.train_jsonl), args.limit_train)
    else:
        train_records = load_hf_records(args.hf_repo, args.train_split, args.limit_train)
    log_stage("load_validation_start", source=args.validation_jsonl)
    validation_records = read_jsonl(Path(args.validation_jsonl), args.limit_validation)

    endpoint_by_key: dict[str, dict[str, Any]] = {}
    raw_smiles: set[str] = set()
    clean_dir = out_dir / "clean_splits"
    train_stats, train_invalid = clean_records_to_disk(
        "train",
        train_records,
        clean_dir,
        endpoint_by_key=endpoint_by_key,
        raw_smiles=raw_smiles,
        progress_every=args.progress_every,
    )
    validation_stats, validation_invalid = clean_records_to_disk(
        "validation",
        validation_records,
        clean_dir,
        endpoint_by_key=endpoint_by_key,
        raw_smiles=raw_smiles,
        progress_every=args.progress_every,
        total=len(validation_records),
    )
    clean_stats_by_split = {"train": train_stats, "validation": validation_stats}
    invalid_rows_by_split = {
        "train": train_invalid,
        "validation": validation_invalid,
    }

    log_stage("collect_endpoints_start")
    endpoints = list(endpoint_by_key.values())
    endpoint_texts = [endpoint["endpoint_text"] for endpoint in endpoints]
    embedding_devices = [] if args.embedding_backend == "dummy" else resolve_devices(args.devices)
    log_stage(
        "endpoint_embedding_start",
        n_texts=len(endpoint_texts),
        backend=args.embedding_backend,
        devices=embedding_devices or ["not_applicable"],
    )
    endpoint_embeddings = embed_endpoint_texts(
        endpoint_texts,
        model_name_or_path=args.embedding_model,
        backend=args.embedding_backend,
        batch_size=args.embedding_batch_size,
        devices=embedding_devices,
    )
    write_jsonl(out_dir / "endpoints.jsonl", endpoints)
    np.save(out_dir / "endpoint_embeddings.npy", endpoint_embeddings.astype(np.float16, copy=False))
    log_stage("endpoint_embedding_done", shape=list(endpoint_embeddings.shape))

    log_stage("rdkit_features_start", workers=args.rdkit_workers)
    molecule_features, descriptor_names = build_molecule_feature_table_from_smiles(
        sorted(raw_smiles),
        workers=args.rdkit_workers,
        progress_every=args.progress_every,
    )
    write_jsonl(out_dir / "molecules.jsonl", molecule_features["records"])
    np.savez_compressed(
        out_dir / "molecule_features.npz",
        fingerprints=molecule_features["fingerprints"],
        descriptors=molecule_features["descriptors"],
    )
    log_stage(
        "rdkit_features_done",
        unique_molecules=len(molecule_features["records"]),
        invalid_molecules=len(molecule_features["invalid_molecules"]),
    )

    log_stage("row_indices_start")
    row_index_stats = write_row_indices_from_clean_files(
        out_dir / "row_indices",
        clean_dir,
        ["train", "validation"],
        endpoints,
        molecule_features,
    )
    log_stage("row_indices_done", stats=row_index_stats)
    manifest = {
        "created_at_unix": time.time(),
        "wall_s": round(time.time() - started, 3),
        "source": {
            "hf_repo": args.hf_repo,
            "train_split": args.train_split,
            "train_jsonl": str(args.train_jsonl) if args.train_jsonl else None,
            "validation_jsonl": str(args.validation_jsonl),
            "limit_train": args.limit_train,
            "limit_validation": args.limit_validation,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "clean_splits": str(out_dir / "clean_splits"),
            "endpoints_jsonl": str(out_dir / "endpoints.jsonl"),
            "endpoint_embeddings_npy": str(out_dir / "endpoint_embeddings.npy"),
            "molecules_jsonl": str(out_dir / "molecules.jsonl"),
            "molecule_features_npz": str(out_dir / "molecule_features.npz"),
            "row_indices": str(out_dir / "row_indices"),
        },
        "counts": {
            "clean_rows": {split: stats["clean"] for split, stats in clean_stats_by_split.items()},
            "invalid_rows": {split: len(rows) for split, rows in invalid_rows_by_split.items()},
            "indexed_rows": row_index_stats,
            "unique_endpoints": len(endpoints),
            "unique_molecules": len(molecule_features["records"]),
            "invalid_molecules": len(molecule_features["invalid_molecules"]),
        },
        "endpoint_embedding": {
            "backend": args.embedding_backend,
            "model": args.embedding_model,
            "shape": list(endpoint_embeddings.shape),
            "dtype_on_disk": "float16",
            "normalized": True,
            "batch_size": args.embedding_batch_size,
            "devices": args.devices,
            "resolved_devices": embedding_devices,
        },
        "molecule_features": {
            "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048, "dtype_on_disk": "uint8"},
            "descriptors": {
                "source": "rdkit.Chem.Descriptors._descList",
                "names": descriptor_names,
                "dtype_on_disk": "float32",
                "missing_value": "NaN",
            },
            "rdkit_workers": args.rdkit_workers,
            "thread_env": rdkit_thread_env(),
        },
        "invalid_molecules": molecule_features["invalid_molecules"][:1000],
        "invalid_rows": {split: rows[:1000] for split, rows in invalid_rows_by_split.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "counts": manifest["counts"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-jsonl", default="", help="Optional local train JSONL override for smoke tests.")
    parser.add_argument("--validation-jsonl", default=DEFAULT_VALIDATION_JSONL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformers", "dummy"],
        default="sentence-transformers",
        help="Use dummy only for parser/cache smoke tests.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--devices", default="auto", help="'auto', 'cpu', or comma-separated cuda device ids.")
    parser.add_argument("--rdkit-workers", type=int, default=min(256, os.cpu_count() or 1))
    parser.add_argument("--limit-train", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--limit-validation", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--progress-every", type=int, default=50000)
    return parser.parse_args(argv)


def ensure_output_dirs(out_dir: Path) -> None:
    for child in ("clean_splits", "row_indices"):
        (out_dir / child).mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit > 0 and index >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path, limit: int = 0) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit > 0 and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def load_hf_records(repo: str, split: str, limit: int = 0) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(repo, split=split, streaming=True)
    if limit <= 0:
        return dataset
    return iter_limited(dataset, limit)


def iter_limited(rows: Iterable[dict[str, Any]], limit: int) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(rows):
        if index >= limit:
            break
        yield row


def parse_prompt(prompt: str) -> ParsedPrompt:
    endpoint_match = ENDPOINT_RE.search(prompt)
    molecule_a_match = MOLECULE_A_RE.search(prompt)
    molecule_b_match = MOLECULE_B_RE.search(prompt)
    missing = []
    if endpoint_match is None:
        missing.append("endpoint")
    if molecule_a_match is None:
        missing.append("molecule_a_smiles")
    if molecule_b_match is None:
        missing.append("molecule_b_smiles")
    if missing:
        raise ValueError(f"Prompt is missing required sections: {', '.join(missing)}")

    endpoint_text = normalize_endpoint_text(endpoint_match.group("endpoint"))
    return ParsedPrompt(
        endpoint_text=endpoint_text,
        molecule_a_smiles=molecule_a_match.group("smiles").strip(),
        molecule_b_smiles=molecule_b_match.group("smiles").strip(),
    )


def normalize_endpoint_text(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def completion_to_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    try:
        return COMPLETION_TO_LABEL[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported completion label: {value!r}") from exc


def clean_records(
    split: str,
    records: Iterable[dict[str, Any]],
    *,
    progress_every: int = 0,
    total: int | None = None,
) -> tuple[list[CleanRow], list[dict[str, Any]]]:
    clean = []
    invalid = []
    started = time.time()
    for source_index, record in enumerate(records):
        metadata = dict(record.get("metadata") or {})
        try:
            parsed = parse_prompt(str(record.get("prompt") or ""))
            label = completion_to_label(record.get("completion"))
        except Exception as exc:
            invalid.append(
                {
                    "split": split,
                    "source_index": source_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "metadata": metadata,
                }
            )
            continue
        endpoint_text_hash = stable_hash(parsed.endpoint_text)
        endpoint_key = str(metadata.get("endpoint_id") or f"text_sha256:{endpoint_text_hash}")
        clean.append(
            CleanRow(
                split=split,
                source_index=source_index,
                label=label,
                endpoint_key=endpoint_key,
                endpoint_text=parsed.endpoint_text,
                endpoint_text_hash=endpoint_text_hash,
                molecule_a_smiles=parsed.molecule_a_smiles,
                molecule_b_smiles=parsed.molecule_b_smiles,
                metadata=metadata,
            )
        )
        completed = source_index + 1
        if progress_every > 0 and completed % progress_every == 0:
            elapsed = max(time.time() - started, 1e-9)
            progress = {
                "stage": "clean_records",
                "split": split,
                "completed": completed,
                "clean": len(clean),
                "invalid": len(invalid),
                "rate_per_s": round(completed / elapsed, 2),
                "elapsed_s": round(elapsed, 1),
            }
            if total is not None:
                remaining = max(total - completed, 0)
                progress["total"] = total
                progress["eta_s"] = round(remaining / (completed / elapsed), 1) if completed else None
            print(json.dumps(progress), flush=True)
    log_stage("clean_records_done", split=split, completed=len(clean) + len(invalid), clean=len(clean), invalid=len(invalid))
    return clean, invalid


def clean_records_to_disk(
    split: str,
    records: Iterable[dict[str, Any]],
    out_dir: Path,
    *,
    endpoint_by_key: dict[str, dict[str, Any]],
    raw_smiles: set[str],
    progress_every: int = 0,
    total: int | None = None,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_path = out_dir / f"{split}.jsonl"
    invalid_path = out_dir / f"{split}_invalid_rows.jsonl"
    invalid = []
    clean_count = 0
    started = time.time()
    with clean_path.open("w", encoding="utf-8") as clean_handle, invalid_path.open("w", encoding="utf-8") as invalid_handle:
        for source_index, record in enumerate(records):
            metadata = dict(record.get("metadata") or {})
            try:
                parsed = parse_prompt(str(record.get("prompt") or ""))
                label = completion_to_label(record.get("completion"))
            except Exception as exc:
                invalid_row = {
                    "split": split,
                    "source_index": source_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "metadata": metadata,
                }
                invalid.append(invalid_row)
                invalid_handle.write(json.dumps(invalid_row, ensure_ascii=False, sort_keys=True) + "\n")
                continue

            endpoint_text_hash = stable_hash(parsed.endpoint_text)
            endpoint_key = str(metadata.get("endpoint_id") or f"text_sha256:{endpoint_text_hash}")
            if endpoint_key not in endpoint_by_key:
                endpoint_by_key[endpoint_key] = {
                    "endpoint_index": len(endpoint_by_key),
                    "endpoint_key": endpoint_key,
                    "endpoint_text_hash": endpoint_text_hash,
                    "endpoint_text": parsed.endpoint_text,
                    "first_split": split,
                    "metadata_example": metadata,
                    "n_rows": 0,
                }
            endpoint_by_key[endpoint_key]["n_rows"] += 1
            raw_smiles.add(parsed.molecule_a_smiles)
            raw_smiles.add(parsed.molecule_b_smiles)

            clean_row = CleanRow(
                split=split,
                source_index=source_index,
                label=label,
                endpoint_key=endpoint_key,
                endpoint_text=parsed.endpoint_text,
                endpoint_text_hash=endpoint_text_hash,
                molecule_a_smiles=parsed.molecule_a_smiles,
                molecule_b_smiles=parsed.molecule_b_smiles,
                metadata=metadata,
            )
            clean_handle.write(json.dumps(clean_row_to_json(clean_row), ensure_ascii=False, sort_keys=True) + "\n")
            clean_count += 1

            completed = source_index + 1
            if progress_every > 0 and completed % progress_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                progress = {
                    "stage": "clean_records",
                    "split": split,
                    "completed": completed,
                    "clean": clean_count,
                    "invalid": len(invalid),
                    "rate_per_s": round(completed / elapsed, 2),
                    "elapsed_s": round(elapsed, 1),
                }
                if total is not None:
                    remaining = max(total - completed, 0)
                    progress["total"] = total
                    progress["eta_s"] = round(remaining / (completed / elapsed), 1) if completed else None
                print(json.dumps(progress), flush=True)
    completed = clean_count + len(invalid)
    log_stage("clean_records_done", split=split, completed=completed, clean=clean_count, invalid=len(invalid))
    return {"completed": completed, "clean": clean_count, "invalid": len(invalid)}, invalid


def collect_endpoints(clean_by_split: dict[str, list[CleanRow]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        for row in clean_by_split.get(split, []):
            if row.endpoint_key not in by_key:
                by_key[row.endpoint_key] = {
                    "endpoint_index": len(by_key),
                    "endpoint_key": row.endpoint_key,
                    "endpoint_text_hash": row.endpoint_text_hash,
                    "endpoint_text": row.endpoint_text,
                    "first_split": split,
                    "metadata_example": row.metadata,
                    "n_rows": 0,
                }
            by_key[row.endpoint_key]["n_rows"] += 1
    return list(by_key.values())


def write_clean_splits(out_dir: Path, clean_by_split: dict[str, list[CleanRow]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in clean_by_split.items():
        write_jsonl(out_dir / f"{split}.jsonl", [clean_row_to_json(row) for row in rows])


def write_invalid_rows(out_dir: Path, invalid_rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in invalid_rows_by_split.items():
        write_jsonl(out_dir / f"{split}_invalid_rows.jsonl", rows)


def clean_row_to_json(row: CleanRow) -> dict[str, Any]:
    return {
        "split": row.split,
        "source_index": row.source_index,
        "label": row.label,
        "label_id": LABEL_TO_ID[row.label],
        "endpoint_key": row.endpoint_key,
        "endpoint_text_hash": row.endpoint_text_hash,
        "endpoint_text": row.endpoint_text,
        "molecule_a_smiles": row.molecule_a_smiles,
        "molecule_b_smiles": row.molecule_b_smiles,
        "metadata": row.metadata,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def embed_endpoint_texts(
    texts: list[str],
    *,
    model_name_or_path: str,
    backend: str,
    batch_size: int,
    devices: list[str],
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float16)
    if backend == "dummy":
        return dummy_embeddings(texts, dim=32)
    return sentence_transformer_embeddings(texts, model_name_or_path, batch_size, devices)


def dummy_embeddings(texts: list[str], dim: int) -> np.ndarray:
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)
    for row_index, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer((digest * ((dim // len(digest)) + 1))[:dim], dtype=np.uint8).astype(np.float32)
        values = (values / 127.5) - 1.0
        norm = np.linalg.norm(values)
        embeddings[row_index] = values / norm if norm > 0 else values
    return embeddings


def resolve_devices(devices_arg: str) -> list[str]:
    value = str(devices_arg or "auto").strip().lower()
    if value == "cpu":
        return ["cpu"]
    import torch

    if value == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        return ["cpu"]
    devices = []
    for item in devices_arg.split(","):
        item = item.strip()
        if not item:
            continue
        devices.append(item if item.startswith(("cuda", "cpu")) else f"cuda:{item}")
    return devices or ["cpu"]


def sentence_transformer_embeddings(
    texts: list[str],
    model_name_or_path: str,
    batch_size: int,
    devices: list[str],
) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if len(devices) <= 1:
        device = devices[0] if devices else None
        try:
            model = SentenceTransformer(model_name_or_path, model_kwargs=model_kwargs, device=device)
        except TypeError:
            model = SentenceTransformer(model_name_or_path, device=device)
        if devices:
            try:
                model = model.to(devices[0])
            except Exception:
                pass
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    try:
        model = SentenceTransformer(model_name_or_path, model_kwargs=model_kwargs, device="cpu")
    except TypeError:
        model = SentenceTransformer(model_name_or_path, device="cpu")
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        embeddings = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    finally:
        model.stop_multi_process_pool(pool)
    return np.asarray(embeddings, dtype=np.float32)


def build_molecule_feature_table(
    clean_by_split: dict[str, list[CleanRow]],
    *,
    workers: int,
    progress_every: int,
) -> tuple[dict[str, Any], list[str]]:
    raw_smiles = []
    seen = set()
    for rows in clean_by_split.values():
        for row in rows:
            for smiles in (row.molecule_a_smiles, row.molecule_b_smiles):
                if smiles not in seen:
                    seen.add(smiles)
                    raw_smiles.append(smiles)

    return build_molecule_feature_table_from_smiles(raw_smiles, workers=workers, progress_every=progress_every)


def build_molecule_feature_table_from_smiles(
    raw_smiles: list[str],
    *,
    workers: int,
    progress_every: int,
) -> tuple[dict[str, Any], list[str]]:
    results = compute_rdkit_features_parallel(raw_smiles, workers=workers, progress_every=progress_every)
    descriptor_names = results[0]["descriptor_names"] if results else descriptor_names_for_current_rdkit()

    records: list[dict[str, Any]] = []
    fingerprints = []
    descriptors = []
    invalid_molecules = []
    canonical_to_index: dict[str, int] = {}
    raw_to_index: dict[str, int] = {}

    for result in results:
        raw = result["raw_smiles"]
        if not result["valid"]:
            invalid_molecules.append({"raw_smiles": raw, "error": result["error"]})
            continue
        canonical = result["canonical_smiles"]
        if canonical not in canonical_to_index:
            canonical_to_index[canonical] = len(records)
            records.append(
                {
                    "molecule_index": len(records),
                    "canonical_smiles": canonical,
                    "first_raw_smiles": raw,
                    "inchi_key": result.get("inchi_key"),
                }
            )
            fingerprints.append(result["fingerprint"])
            descriptors.append(result["descriptors"])
        raw_to_index[raw] = canonical_to_index[canonical]

    feature_table = {
        "records": records,
        "fingerprints": np.asarray(fingerprints, dtype=np.uint8)
        if fingerprints
        else np.zeros((0, 2048), dtype=np.uint8),
        "descriptors": np.asarray(descriptors, dtype=np.float32)
        if descriptors
        else np.zeros((0, len(descriptor_names)), dtype=np.float32),
        "raw_to_index": raw_to_index,
        "invalid_molecules": invalid_molecules,
    }
    return feature_table, descriptor_names


def compute_rdkit_features_parallel(raw_smiles: list[str], *, workers: int, progress_every: int) -> list[dict[str, Any]]:
    if not raw_smiles:
        return []
    workers = max(1, min(int(workers), len(raw_smiles)))
    if workers == 1:
        init_rdkit_worker()
        return [compute_one_rdkit_feature(smiles) for smiles in raw_smiles]

    results = []
    started = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers, initializer=init_rdkit_worker) as pool:
        for count, result in enumerate(pool.imap(compute_one_rdkit_feature, raw_smiles, chunksize=64), start=1):
            results.append(result)
            if progress_every > 0 and count % progress_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                rate = count / elapsed
                remaining = len(raw_smiles) - count
                eta = remaining / rate if rate > 0 else math.inf
                print(
                    json.dumps(
                        {
                            "stage": "rdkit_features",
                            "completed": count,
                            "total": len(raw_smiles),
                            "rate_per_s": round(rate, 2),
                            "elapsed_s": round(elapsed, 1),
                            "eta_s": round(eta, 1),
                        }
                    ),
                    flush=True,
                )
    return results


def rdkit_thread_env() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "RDKIT_NUM_THREADS": "1",
    }


def init_rdkit_worker() -> None:
    for key, value in rdkit_thread_env().items():
        os.environ.setdefault(key, value)
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def descriptor_names_for_current_rdkit() -> list[str]:
    from rdkit.Chem import Descriptors

    return [name for name, _ in Descriptors._descList]


def compute_one_rdkit_feature(raw_smiles: str) -> dict[str, Any]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdFingerprintGenerator

    descriptor_items = list(Descriptors._descList)
    descriptor_names = [name for name, _ in descriptor_items]
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return {
            "raw_smiles": raw_smiles,
            "valid": False,
            "error": "RDKit could not parse SMILES",
            "descriptor_names": descriptor_names,
        }
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as exc:
        return {
            "raw_smiles": raw_smiles,
            "valid": False,
            "error": f"canonicalization failed: {type(exc).__name__}: {exc}",
            "descriptor_names": descriptor_names,
        }

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprint(mol)
    fp_array = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)

    descriptor_values = np.empty((len(descriptor_items),), dtype=np.float32)
    for index, (_, func) in enumerate(descriptor_items):
        try:
            value = float(func(mol))
        except Exception:
            value = math.nan
        descriptor_values[index] = value if math.isfinite(value) else math.nan

    try:
        inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        inchi_key = None
    return {
        "raw_smiles": raw_smiles,
        "valid": True,
        "canonical_smiles": canonical,
        "inchi_key": inchi_key,
        "fingerprint": fp_array,
        "descriptors": descriptor_values,
        "descriptor_names": descriptor_names,
    }


def write_row_indices(
    out_dir: Path,
    clean_by_split: dict[str, list[CleanRow]],
    endpoints: list[dict[str, Any]],
    molecule_features: dict[str, Any],
) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_to_index = {row["endpoint_key"]: int(row["endpoint_index"]) for row in endpoints}
    molecule_to_index = molecule_features["raw_to_index"]
    stats = {}
    for split, rows in clean_by_split.items():
        endpoint_indices = []
        molecule_a_indices = []
        molecule_b_indices = []
        label_ids = []
        source_indices = []
        skipped = 0
        for row in rows:
            mol_a_index = molecule_to_index.get(row.molecule_a_smiles)
            mol_b_index = molecule_to_index.get(row.molecule_b_smiles)
            if mol_a_index is None or mol_b_index is None:
                skipped += 1
                continue
            endpoint_indices.append(endpoint_to_index[row.endpoint_key])
            molecule_a_indices.append(mol_a_index)
            molecule_b_indices.append(mol_b_index)
            label_ids.append(LABEL_TO_ID[row.label])
            source_indices.append(row.source_index)
        np.savez_compressed(
            out_dir / f"{split}.npz",
            endpoint_index=np.asarray(endpoint_indices, dtype=np.int64),
            molecule_a_index=np.asarray(molecule_a_indices, dtype=np.int64),
            molecule_b_index=np.asarray(molecule_b_indices, dtype=np.int64),
            label=np.asarray(label_ids, dtype=np.int64),
            source_index=np.asarray(source_indices, dtype=np.int64),
        )
        stats[split] = {"n_rows": len(rows), "n_indexed": len(label_ids), "n_skipped_invalid_molecule": skipped}
    return stats


def write_row_indices_from_clean_files(
    out_dir: Path,
    clean_dir: Path,
    splits: list[str],
    endpoints: list[dict[str, Any]],
    molecule_features: dict[str, Any],
) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_to_index = {row["endpoint_key"]: int(row["endpoint_index"]) for row in endpoints}
    molecule_to_index = molecule_features["raw_to_index"]
    stats = {}
    for split in splits:
        clean_path = clean_dir / f"{split}.jsonl"
        n_rows = count_lines(clean_path)
        arrays = {
            "endpoint_index": np.memmap(out_dir / f"{split}.endpoint_index.tmp", dtype=np.int64, mode="w+", shape=(n_rows,)),
            "molecule_a_index": np.memmap(out_dir / f"{split}.molecule_a_index.tmp", dtype=np.int64, mode="w+", shape=(n_rows,)),
            "molecule_b_index": np.memmap(out_dir / f"{split}.molecule_b_index.tmp", dtype=np.int64, mode="w+", shape=(n_rows,)),
            "label": np.memmap(out_dir / f"{split}.label.tmp", dtype=np.int64, mode="w+", shape=(n_rows,)),
            "source_index": np.memmap(out_dir / f"{split}.source_index.tmp", dtype=np.int64, mode="w+", shape=(n_rows,)),
        }
        n_indexed = 0
        skipped = 0
        with clean_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                mol_a_index = molecule_to_index.get(row["molecule_a_smiles"])
                mol_b_index = molecule_to_index.get(row["molecule_b_smiles"])
                if mol_a_index is None or mol_b_index is None:
                    skipped += 1
                    continue
                arrays["endpoint_index"][n_indexed] = endpoint_to_index[row["endpoint_key"]]
                arrays["molecule_a_index"][n_indexed] = mol_a_index
                arrays["molecule_b_index"][n_indexed] = mol_b_index
                arrays["label"][n_indexed] = int(row["label_id"])
                arrays["source_index"][n_indexed] = int(row["source_index"])
                n_indexed += 1
        for array in arrays.values():
            array.flush()
        np.savez_compressed(
            out_dir / f"{split}.npz",
            endpoint_index=np.asarray(arrays["endpoint_index"][:n_indexed]),
            molecule_a_index=np.asarray(arrays["molecule_a_index"][:n_indexed]),
            molecule_b_index=np.asarray(arrays["molecule_b_index"][:n_indexed]),
            label=np.asarray(arrays["label"][:n_indexed]),
            source_index=np.asarray(arrays["source_index"][:n_indexed]),
        )
        for key in arrays:
            tmp_path = out_dir / f"{split}.{key}.tmp"
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        stats[split] = {"n_rows": n_rows, "n_indexed": n_indexed, "n_skipped_invalid_molecule": skipped}
    return stats


def count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_stage(stage: str, **payload: Any) -> None:
    message = {"stage": stage, **payload}
    print(json.dumps(message, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
