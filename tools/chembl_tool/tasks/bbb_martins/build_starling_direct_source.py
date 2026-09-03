"""Materialize the pinned complete Starling BBB direct source as Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.paths import raw_starling_task_root
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_benchmark import (
    SOURCE_DATASET,
    SOURCE_REVISION,
)


DEFAULT_OUTPUT = raw_starling_task_root("bbb_martins") / "Direct_BBB/records.parquet"
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_name("source_manifest.json")
EXPECTED_ROWS = 304_845


def build_direct_source(
    *,
    output: str | Path = DEFAULT_OUTPUT,
    manifest: str | Path = DEFAULT_MANIFEST,
    revision: str = SOURCE_REVISION,
    overwrite: bool = False,
) -> dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    destination = Path(output)
    manifest_path = Path(manifest)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists: {destination}; pass --overwrite")
    dataset = load_dataset(SOURCE_DATASET, split="train", revision=revision)
    resolved = HfApi().dataset_info(SOURCE_DATASET, revision=revision).sha
    if resolved != SOURCE_REVISION:
        raise ValueError(
            f"resolved Starling BBB revision drift: expected {SOURCE_REVISION}, found {resolved}"
        )
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(
            f"Starling BBB row-count drift: expected {EXPECTED_ROWS:,}, found {len(dataset):,}"
        )
    frame = dataset.to_pandas()
    frame.insert(0, "source_index", range(len(frame)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".direct-bbb-") as name:
        temporary = Path(name) / destination.name
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
            compression_level=9,
        )
        os.replace(temporary, destination)
    payload = {
        "dataset": SOURCE_DATASET,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "split": "train",
        "rows": len(frame),
        "columns": list(frame.columns),
        "source_index": {"minimum": 0, "maximum": len(frame) - 1, "unique": True},
        "path": str(destination),
        "sha256": _sha256(destination),
        "construction": {
            "row_filtering": False,
            "text_rewriting": False,
            "added_columns": ["source_index"],
            "parquet_compression": "zstd:9",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_direct_source(
        output=args.output,
        manifest=args.manifest,
        revision=args.revision,
        overwrite=args.overwrite,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
