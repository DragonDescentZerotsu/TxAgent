"""Prepare HF assay-transfer features for a simple trained MLP baseline.

The input format is the prompt/completion/metadata JSONL used by the
activity-transfer LLM benchmark. This script extracts endpoint text and two
molecule SMILES strings, computes endpoint semantic embeddings, computes
deduplicated molecule features, and writes compact index arrays for
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


DEFAULT_HF_REPO = os.environ.get("TXAGENT_TRANSFER_HF_REPO")
DEFAULT_VALIDATION_JSONL = (
    "outputs/chembl_tool/activity_transfer_benchmark/hf_transfer_valid20k/"
    "proper_assay_transfer_no_prop_no_tanimoto/validation.jsonl"
)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/activity_transfer_benchmark/mlp_baselines/"
    "qwen3_embedding_rdkit_v1/preprocessed"
)
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_MOLFORMER_MODEL = "ibm-research/MoLFormer-XL-both-10pct"

DEFAULT_LABEL_TO_ID = {"different": 0, "similar": 1}
DEFAULT_COMPLETION_TO_LABEL = {"A": "similar", "B": "different"}
MAX_DESCRIPTOR_ABS_VALUE = 1.0e12

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
    validation_records = iter_jsonl(Path(args.validation_jsonl), args.limit_validation)
    test_records = None
    if args.test_jsonl:
        log_stage("load_test_start", source=args.test_jsonl)
        test_records = iter_jsonl(Path(args.test_jsonl), args.limit_test)

    completion_label_map = {"A": args.completion_a_label, "B": args.completion_b_label}
    label_to_id = {args.completion_b_label: 0, args.completion_a_label: 1}
    endpoint_by_key: dict[str, dict[str, Any]] = {}
    raw_smiles: set[str] = set()
    clean_dir = out_dir / "clean_splits"
    train_stats, train_invalid = clean_records_to_disk(
        "train",
        train_records,
        clean_dir,
        endpoint_by_key=endpoint_by_key,
        raw_smiles=raw_smiles,
        completion_label_map=completion_label_map,
        label_to_id=label_to_id,
        progress_every=args.progress_every,
    )
    validation_stats, validation_invalid = clean_records_to_disk(
        "validation",
        validation_records,
        clean_dir,
        endpoint_by_key=endpoint_by_key,
        raw_smiles=raw_smiles,
        completion_label_map=completion_label_map,
        label_to_id=label_to_id,
        progress_every=args.progress_every,
    )
    clean_stats_by_split = {"train": train_stats, "validation": validation_stats}
    invalid_rows_by_split = {
        "train": train_invalid,
        "validation": validation_invalid,
    }
    splits = ["train", "validation"]
    if test_records is not None:
        test_stats, test_invalid = clean_records_to_disk(
            "test",
            test_records,
            clean_dir,
            endpoint_by_key=endpoint_by_key,
            raw_smiles=raw_smiles,
            completion_label_map=completion_label_map,
            label_to_id=label_to_id,
            progress_every=args.progress_every,
        )
        clean_stats_by_split["test"] = test_stats
        invalid_rows_by_split["test"] = test_invalid
        splits.append("test")

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

    molecule_devices = [] if args.molecule_feature_backend == "rdkit" else resolve_devices(args.molformer_devices)
    log_stage(
        "molecule_features_start",
        backend=args.molecule_feature_backend,
        rdkit_workers=args.rdkit_workers,
        molformer_model=args.molformer_model if args.molecule_feature_backend == "molformer" else None,
        molformer_devices=molecule_devices or ["not_applicable"],
    )
    molecule_features, descriptor_names = build_molecule_feature_table_from_smiles(
        sorted(raw_smiles),
        backend=args.molecule_feature_backend,
        workers=args.rdkit_workers,
        progress_every=args.progress_every,
        molformer_model=args.molformer_model,
        molformer_batch_size=args.molformer_batch_size,
        molformer_devices=molecule_devices,
        molformer_dtype=args.molformer_dtype,
        molformer_max_length=args.molformer_max_length,
        molformer_normalize=not args.no_molformer_normalize,
        molformer_random_seed=args.molformer_random_seed,
        work_dir=out_dir,
    )
    write_jsonl(out_dir / "molecules.jsonl", molecule_features["records"])
    save_molecule_features_npz(out_dir / "molecule_features.npz", molecule_features)
    log_stage(
        "molecule_features_done",
        backend=args.molecule_feature_backend,
        unique_molecules=len(molecule_features["records"]),
        invalid_molecules=len(molecule_features["invalid_molecules"]),
    )

    log_stage("row_indices_start")
    row_index_stats = write_row_indices_from_clean_files(
        out_dir / "row_indices",
        clean_dir,
        splits,
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
            "test_jsonl": str(args.test_jsonl) if args.test_jsonl else None,
            "limit_train": args.limit_train,
            "limit_validation": args.limit_validation,
            "limit_test": args.limit_test,
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
        "label_mapping": {
            "completion_to_label": completion_label_map,
            "label_to_id": label_to_id,
            "id_to_label": {str(value): key for key, value in label_to_id.items()},
            "positive_label": args.completion_a_label,
            "negative_label": args.completion_b_label,
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
            "backend": args.molecule_feature_backend,
            "rdkit_workers": args.rdkit_workers,
            "thread_env": rdkit_thread_env(),
        },
        "invalid_molecules": molecule_features["invalid_molecules"][:1000],
        "invalid_rows": {split: rows[:1000] for split, rows in invalid_rows_by_split.items()},
    }
    if args.molecule_feature_backend == "rdkit":
        manifest["molecule_features"].update(
            {
                "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 2048, "dtype_on_disk": "uint8"},
                "descriptors": {
                    "source": "rdkit.Chem.Descriptors._descList",
                    "names": descriptor_names,
                    "dtype_on_disk": "float32",
                    "missing_value": "NaN",
                    "max_abs_value": MAX_DESCRIPTOR_ABS_VALUE,
                },
            }
        )
    else:
        manifest["molecule_features"].update(
            {
                "molformer": {
                    "model": args.molformer_model,
                    "source": "outputs.pooler_output from Hugging Face AutoModel",
                    "shape": list(molecule_features["molformer_embeddings"].shape),
                    "dtype_on_disk": "float16",
                    "normalized": not args.no_molformer_normalize,
                    "batch_size": args.molformer_batch_size,
                    "devices": args.molformer_devices,
                    "resolved_devices": molecule_devices,
                    "dtype": args.molformer_dtype,
                    "max_length": args.molformer_max_length,
                    "random_seed": args.molformer_random_seed,
                    "trust_remote_code": True,
                },
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "counts": manifest["counts"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-jsonl", default="", help="Optional local train JSONL override for smoke tests.")
    parser.add_argument("--validation-jsonl", default=DEFAULT_VALIDATION_JSONL)
    parser.add_argument("--test-jsonl", default="", help="Optional local test JSONL.")
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
    parser.add_argument(
        "--molecule-feature-backend",
        choices=["rdkit", "molformer"],
        default="rdkit",
        help="rdkit keeps the legacy fingerprint+descriptor features; molformer replaces them with frozen MolFormer embeddings.",
    )
    parser.add_argument("--molformer-model", default=DEFAULT_MOLFORMER_MODEL)
    parser.add_argument("--molformer-batch-size", type=int, default=1)
    parser.add_argument("--molformer-devices", default="auto", help="'auto' uses every visible CUDA GPU.")
    parser.add_argument("--molformer-max-length", type=int, default=202)
    parser.add_argument(
        "--molformer-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--molformer-random-seed",
        type=int,
        default=2,
        help="Seed for MolFormer's non-persistent random feature-map buffers; shared by all GPU workers.",
    )
    parser.add_argument("--no-molformer-normalize", action="store_true")
    parser.add_argument("--rdkit-workers", type=int, default=min(256, os.cpu_count() or 1))
    parser.add_argument("--limit-train", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--limit-validation", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--limit-test", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--completion-a-label", default="similar")
    parser.add_argument("--completion-b-label", default="different")
    parser.add_argument("--progress-every", type=int, default=50000)
    args = parser.parse_args(argv)
    if not args.train_jsonl and not args.hf_repo:
        parser.error("provide --train-jsonl or --hf-repo (or TXAGENT_TRANSFER_HF_REPO)")
    return args


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


def completion_to_label(value: Any, completion_label_map: dict[str, str] | None = None) -> str:
    mapping = completion_label_map or DEFAULT_COMPLETION_TO_LABEL
    text = str(value or "").strip().upper()
    try:
        return mapping[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported completion label: {value!r}") from exc


def clean_records(
    split: str,
    records: Iterable[dict[str, Any]],
    *,
    completion_label_map: dict[str, str] | None = None,
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
            label = completion_to_label(record.get("completion"), completion_label_map)
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
    completion_label_map: dict[str, str],
    label_to_id: dict[str, int],
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
                label = completion_to_label(record.get("completion"), completion_label_map)
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
            clean_handle.write(json.dumps(clean_row_to_json(clean_row, label_to_id), ensure_ascii=False, sort_keys=True) + "\n")
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


def write_clean_splits(
    out_dir: Path,
    clean_by_split: dict[str, list[CleanRow]],
    label_to_id: dict[str, int] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in clean_by_split.items():
        write_jsonl(out_dir / f"{split}.jsonl", [clean_row_to_json(row, label_to_id) for row in rows])


def write_invalid_rows(out_dir: Path, invalid_rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in invalid_rows_by_split.items():
        write_jsonl(out_dir / f"{split}_invalid_rows.jsonl", rows)


def clean_row_to_json(row: CleanRow, label_to_id: dict[str, int] | None = None) -> dict[str, Any]:
    mapping = label_to_id or DEFAULT_LABEL_TO_ID
    return {
        "split": row.split,
        "source_index": row.source_index,
        "label": row.label,
        "label_id": mapping[row.label],
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
    backend: str = "rdkit",
    workers: int,
    progress_every: int,
    molformer_model: str = DEFAULT_MOLFORMER_MODEL,
    molformer_batch_size: int = 1,
    molformer_devices: list[str] | None = None,
    molformer_dtype: str = "float32",
    molformer_max_length: int = 202,
    molformer_normalize: bool = True,
    molformer_random_seed: int = 2,
    work_dir: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if backend == "rdkit":
        results = compute_rdkit_features_parallel(raw_smiles, workers=workers, progress_every=progress_every)
        descriptor_names = results[0]["descriptor_names"] if results else descriptor_names_for_current_rdkit()
    elif backend == "molformer":
        results = compute_molecule_identities_parallel(raw_smiles, workers=workers, progress_every=progress_every)
        descriptor_names = []
    else:
        raise ValueError(f"Unsupported molecule feature backend: {backend}")

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
            if backend == "rdkit":
                fingerprints.append(result["fingerprint"])
                descriptors.append(result["descriptors"])
        raw_to_index[raw] = canonical_to_index[canonical]

    feature_table = {
        "records": records,
        "raw_to_index": raw_to_index,
        "invalid_molecules": invalid_molecules,
        "backend": backend,
    }
    if backend == "rdkit":
        feature_table["fingerprints"] = (
            np.asarray(fingerprints, dtype=np.uint8) if fingerprints else np.zeros((0, 2048), dtype=np.uint8)
        )
        feature_table["descriptors"] = (
            np.asarray(descriptors, dtype=np.float32)
            if descriptors
            else np.zeros((0, len(descriptor_names)), dtype=np.float32)
        )
    else:
        smiles_for_embedding = [str(record["canonical_smiles"]) for record in records]
        feature_table["molformer_embeddings"] = compute_molformer_embeddings_parallel(
            smiles_for_embedding,
            model_name_or_path=molformer_model,
            batch_size=molformer_batch_size,
            devices=molformer_devices or ["cpu"],
            dtype=molformer_dtype,
            max_length=molformer_max_length,
            normalize=molformer_normalize,
            random_seed=molformer_random_seed,
            progress_every=progress_every,
            work_dir=work_dir,
        )
    return feature_table, descriptor_names


def save_molecule_features_npz(path: Path, molecule_features: dict[str, Any]) -> None:
    if molecule_features.get("backend") == "molformer":
        np.savez_compressed(
            path,
            molformer_embeddings=molecule_features["molformer_embeddings"],
            backend=np.asarray(["molformer"]),
        )
        return
    np.savez_compressed(
        path,
        fingerprints=molecule_features["fingerprints"],
        descriptors=molecule_features["descriptors"],
        backend=np.asarray(["rdkit"]),
    )


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


def compute_molecule_identities_parallel(raw_smiles: list[str], *, workers: int, progress_every: int) -> list[dict[str, Any]]:
    if not raw_smiles:
        return []
    workers = max(1, min(int(workers), len(raw_smiles)))
    if workers == 1:
        init_rdkit_worker()
        return [compute_one_molecule_identity(smiles) for smiles in raw_smiles]

    results = []
    started = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers, initializer=init_rdkit_worker) as pool:
        for count, result in enumerate(pool.imap(compute_one_molecule_identity, raw_smiles, chunksize=256), start=1):
            results.append(result)
            if progress_every > 0 and count % progress_every == 0:
                elapsed = max(time.time() - started, 1e-9)
                rate = count / elapsed
                remaining = len(raw_smiles) - count
                eta = remaining / rate if rate > 0 else math.inf
                print(
                    json.dumps(
                        {
                            "stage": "molecule_identity",
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
        if math.isfinite(value) and abs(value) <= MAX_DESCRIPTOR_ABS_VALUE:
            descriptor_values[index] = value
        else:
            descriptor_values[index] = math.nan

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


def compute_one_molecule_identity(raw_smiles: str) -> dict[str, Any]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is None:
        return {
            "raw_smiles": raw_smiles,
            "valid": False,
            "error": "RDKit could not parse SMILES",
        }
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception as exc:
        return {
            "raw_smiles": raw_smiles,
            "valid": False,
            "error": f"canonicalization failed: {type(exc).__name__}: {exc}",
        }
    try:
        inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        inchi_key = None
    return {
        "raw_smiles": raw_smiles,
        "valid": True,
        "canonical_smiles": canonical,
        "inchi_key": inchi_key,
    }


def compute_molformer_embeddings_parallel(
    smiles: list[str],
    *,
    model_name_or_path: str,
    batch_size: int,
    devices: list[str],
    dtype: str,
    max_length: int,
    normalize: bool,
    random_seed: int,
    progress_every: int,
    work_dir: Path | None,
) -> np.ndarray:
    if not smiles:
        return np.zeros((0, 0), dtype=np.float16)
    devices = devices or ["cpu"]
    log_stage("molformer_multi_gpu_start", n_smiles=len(smiles), devices=devices, batch_size=batch_size)
    ctx = get_context("spawn")
    chunks = split_contiguous(smiles, len(devices))
    tasks = []
    for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
        if not chunk:
            continue
        tasks.append(
            (
                worker_id,
                chunk,
                model_name_or_path,
                batch_size,
                device,
                dtype,
                max_length,
                normalize,
                random_seed,
                progress_every,
            )
        )
    with ctx.Pool(processes=len(tasks)) as pool:
        chunk_embeddings = pool.starmap(molformer_embed_worker, tasks)
    embeddings = np.concatenate(chunk_embeddings, axis=0).astype(np.float16, copy=False)
    log_stage("molformer_multi_gpu_done", shape=list(embeddings.shape), dtype=str(embeddings.dtype))
    return embeddings


def split_contiguous(values: list[str], n_chunks: int) -> list[list[str]]:
    n_chunks = max(1, n_chunks)
    chunk_size = int(math.ceil(len(values) / n_chunks))
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def molformer_embed_worker(
    worker_id: int,
    smiles: list[str],
    model_name_or_path: str,
    batch_size: int,
    device: str,
    dtype: str,
    max_length: int,
    normalize: bool,
    random_seed: int,
    progress_every: int,
) -> np.ndarray:
    device = isolate_worker_cuda_device(device)
    import torch
    from transformers import AutoModel, AutoTokenizer

    install_transformers_onnx_compat()
    install_transformers_pruning_compat(torch)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch_dtype = molformer_torch_dtype(dtype, torch)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "deterministic_eval": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("deterministic_eval", None)
        model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    install_molformer_model_method_compat(model, torch)
    model.to(device)
    reset_molformer_rotary_embeddings(model, torch, device)
    model.eval()

    outputs = []
    started = time.time()
    hidden_fallback_count = 0
    for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # MolFormer remote-code can produce NaNs for some molecules under
        # torch.no_grad() in the current stack. We keep grad mode enabled,
        # never backpropagate, and immediately detach the pooled embeddings.
        model_output = model(**inputs, output_hidden_states=True)
        pooled = getattr(model_output, "pooler_output", None)
        if pooled is None:
            pooled = molformer_fallback_pool(model_output, inputs)
        elif not torch.isfinite(pooled).all():
            fallback, fallback_layer = molformer_last_finite_hidden_pool(model_output, inputs)
            if fallback is not None and torch.isfinite(fallback).all():
                hidden_fallback_count += 1
                if hidden_fallback_count <= 5 or hidden_fallback_count % 500 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "molformer_hidden_state_fallback",
                                "worker_id": worker_id,
                                "device": device,
                                "batch_start": start,
                                "batch_size": len(batch),
                                "hidden_state_index": fallback_layer,
                                "fallback_count": hidden_fallback_count,
                            }
                        ),
                        flush=True,
                    )
                pooled = fallback
            else:
                fallback = molformer_fallback_pool(model_output, inputs)
                if torch.isfinite(fallback).all():
                    hidden_fallback_count += 1
                    if hidden_fallback_count <= 5 or hidden_fallback_count % 500 == 0:
                        print(
                            json.dumps(
                                {
                                    "stage": "molformer_last_hidden_fallback",
                                    "worker_id": worker_id,
                                    "device": device,
                                    "batch_start": start,
                                    "batch_size": len(batch),
                                    "fallback_count": hidden_fallback_count,
                                }
                            ),
                            flush=True,
                        )
                    pooled = fallback
        if pooled is None:
            pooled, fallback_layer = molformer_last_finite_hidden_pool(model_output, inputs)
            if pooled is not None:
                hidden_fallback_count += 1
                if hidden_fallback_count <= 5 or hidden_fallback_count % 500 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "molformer_missing_pooler_hidden_fallback",
                                "worker_id": worker_id,
                                "device": device,
                                "batch_start": start,
                                "batch_size": len(batch),
                                "hidden_state_index": fallback_layer,
                                "fallback_count": hidden_fallback_count,
                            }
                        ),
                        flush=True,
                    )
            if pooled is None:
                pooled = molformer_fallback_pool(model_output, inputs)
        if normalize:
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        else:
            pooled = pooled.float()
        if not torch.isfinite(pooled).all():
            token_lengths = inputs["attention_mask"].sum(dim=1).detach().cpu().tolist() if "attention_mask" in inputs else []
            log_stage(
                "molformer_nonfinite_batch",
                worker_id=worker_id,
                device=device,
                batch_start=start,
                batch_size=len(batch),
                token_lengths=token_lengths,
                smiles=batch,
            )
            raise FloatingPointError(
                f"MolFormer produced non-finite embeddings on {device} with dtype={dtype}. "
                "Retry with --molformer-dtype float32 or inspect the logged molformer_nonfinite_batch SMILES."
            )
        outputs.append(pooled.detach().cpu().to(torch.float16).numpy())
        del model_output, pooled, inputs
        completed = min(start + len(batch), len(smiles))
        if progress_every > 0 and completed % progress_every < len(batch):
            elapsed = max(time.time() - started, 1e-9)
            rate = completed / elapsed
            remaining = len(smiles) - completed
            eta = remaining / rate if rate > 0 else math.inf
            print(
                json.dumps(
                    {
                        "stage": "molformer_embeddings",
                        "worker_id": worker_id,
                        "device": device,
                        "completed": completed,
                        "total": len(smiles),
                        "rate_per_s": round(rate, 2),
                        "elapsed_s": round(elapsed, 1),
                        "eta_s": round(eta, 1),
                    }
                ),
                flush=True,
            )
    if hidden_fallback_count:
        log_stage(
            "molformer_hidden_fallback_done",
            worker_id=worker_id,
            device=device,
            fallback_count=hidden_fallback_count,
            total=len(smiles),
        )
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float16)


def isolate_worker_cuda_device(device: str) -> str:
    """Expose one physical CUDA device per MolFormer worker and use it as cuda:0.

    In the current MolFormer remote-code stack, some molecules can produce NaNs
    when a worker targets nonzero device ids such as cuda:6 directly. Remapping
    the selected physical GPU to process-local cuda:0 keeps multi-GPU parallelism
    while avoiding that device-index-sensitive numerical path.
    """

    if not device.startswith("cuda"):
        return device
    if ":" in device:
        try:
            logical_index = int(device.split(":", 1)[1])
        except ValueError:
            logical_index = 0
    else:
        logical_index = 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        physical_id = visible_ids[logical_index] if logical_index < len(visible_ids) else str(logical_index)
    else:
        physical_id = str(logical_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_id
    return "cuda:0"


def install_transformers_onnx_compat() -> None:
    """Provide the old transformers.onnx.OnnxConfig symbol required by MolFormer remote code.

    Newer/minimal Transformers installs can omit the `transformers.onnx` module.
    MolFormer's config imports OnnxConfig only to declare ONNX dynamic axes, which
    is irrelevant for feature extraction. A small shim is enough for AutoConfig.
    """

    import sys
    import types

    if "transformers.onnx" in sys.modules:
        return
    module = types.ModuleType("transformers.onnx")

    class OnnxConfig:  # pragma: no cover - compatibility shim for remote code imports.
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.task = kwargs.get("task", "default")

    module.OnnxConfig = OnnxConfig
    sys.modules["transformers.onnx"] = module


def install_transformers_pruning_compat(torch_module: Any) -> None:
    import transformers.pytorch_utils as pytorch_utils

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(
        heads: Any,
        n_heads: int,
        head_size: int,
        already_pruned_heads: set[int],
    ) -> tuple[set[int], Any]:
        heads = set(heads) - already_pruned_heads
        mask = torch_module.ones(n_heads, head_size)
        for head in sorted(heads):
            adjusted_head = head - sum(1 for pruned_head in already_pruned_heads if pruned_head < head)
            mask[adjusted_head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch_module.arange(len(mask), dtype=torch_module.long)[mask].long()
        return heads, index

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def install_molformer_model_method_compat(model: Any, torch_module: Any) -> None:
    import types

    def get_head_mask(self: Any, head_mask: Any, num_hidden_layers: int, is_attention_chunked: bool = False) -> Any:
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask.to(dtype=next(self.parameters()).dtype)

    def get_extended_attention_mask(self: Any, attention_mask: Any, input_shape: Any, device: Any = None) -> Any:
        if device is None:
            device = attention_mask.device
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")
        dtype = next(self.parameters()).dtype
        extended_attention_mask = extended_attention_mask.to(device=device, dtype=dtype)
        return (1.0 - extended_attention_mask) * -10000.0

    if not hasattr(model, "get_head_mask"):
        model.get_head_mask = types.MethodType(get_head_mask, model)
    model.get_extended_attention_mask = types.MethodType(get_extended_attention_mask, model)


def reset_molformer_rotary_embeddings(model: Any, torch_module: Any, device: str) -> None:
    target_device = torch_module.device(device)
    dtype = next(model.parameters()).dtype
    reset_count = 0
    for module in model.modules():
        if hasattr(module, "_set_cos_sin_cache") and hasattr(module, "max_position_embeddings"):
            if hasattr(module, "dim") and hasattr(module, "base"):
                inv_freq = 1.0 / (
                    float(module.base)
                    ** (torch_module.arange(0, int(module.dim), 2, device=target_device).float() / int(module.dim))
                )
                module.register_buffer("inv_freq", inv_freq, persistent=False)
            module._set_cos_sin_cache(
                seq_len=int(module.max_position_embeddings),
                device=target_device,
                dtype=dtype,
            )
            reset_count += 1
        if hasattr(module, "feature_map") and hasattr(module.feature_map, "deterministic"):
            module.feature_map.deterministic = True
    if reset_count:
        log_stage("molformer_rotary_cache_reset", device=device, count=reset_count, dtype=str(dtype))


def molformer_torch_dtype(dtype: str, torch_module: Any) -> Any:
    if dtype == "auto":
        return None
    if dtype == "float32":
        return torch_module.float32
    if dtype == "float16":
        return torch_module.float16
    if dtype == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported MolFormer dtype: {dtype}")


def molformer_fallback_pool(model_output: Any, inputs: dict[str, Any]) -> Any:
    import torch

    hidden = getattr(model_output, "last_hidden_state", None)
    if hidden is None and isinstance(model_output, (tuple, list)) and model_output:
        hidden = model_output[0]
    if hidden is None:
        raise RuntimeError("MolFormer output does not include pooler_output or last_hidden_state.")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        return hidden[:, 0]
    mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(torch.tensor(1.0, device=hidden.device, dtype=hidden.dtype))
    return summed / denom


def molformer_last_finite_hidden_pool(model_output: Any, inputs: dict[str, Any]) -> tuple[Any | None, int | None]:
    hidden_states = getattr(model_output, "hidden_states", None)
    if not hidden_states:
        return None, None
    import torch

    for index in range(len(hidden_states) - 1, -1, -1):
        hidden = hidden_states[index]
        if hidden is None or not torch.isfinite(hidden).all():
            continue
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return hidden[:, 0], index
        mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(torch.tensor(1.0, device=hidden.device, dtype=hidden.dtype))
        return summed / denom, index
    return None, None


def write_row_indices(
    out_dir: Path,
    clean_by_split: dict[str, list[CleanRow]],
    endpoints: list[dict[str, Any]],
    molecule_features: dict[str, Any],
    label_to_id: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint_to_index = {row["endpoint_key"]: int(row["endpoint_index"]) for row in endpoints}
    molecule_to_index = molecule_features["raw_to_index"]
    mapping = label_to_id or DEFAULT_LABEL_TO_ID
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
            label_ids.append(mapping[row.label])
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
