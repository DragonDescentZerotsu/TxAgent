"""Publish validated node-local Stage-1 inputs for the conditioned gold-v2 build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from data.processing.gold_labels.benchmark_dataset import sha256_file
from data.processing.gold_labels.build_repaired_v2 import validate_stage1_authority
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "conditioned_gold_v2_stage1_publication.v1"
TASKS = ("bbb_martins", "bioavailability_ma")


def publish(*, staging_root: Path, canonical_root: Path) -> dict[str, Any]:
    """Copy immutable Stage 0/1 outputs, relocate manifest paths, and revalidate."""
    if canonical_root.exists():
        raise ValueError(f"canonical publication root already exists: {canonical_root}")
    outputs: dict[str, Any] = {}
    for task in TASKS:
        source = staging_root / task / "v10"
        target = canonical_root / task / "v10"
        for stage in ("00_source", "01_cleaned"):
            shutil.copytree(source / stage, target / stage)
        for relative in ("00_source/manifest.json", "01_cleaned/manifest.json"):
            path = target / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            relocated = _replace_prefix(payload, str(source), str(target))
            write_json_atomic(path, relocated)
        authority = validate_stage1_authority(target)
        files = {
            str(path.relative_to(target)): sha256_file(path)
            for path in sorted(target.rglob("*"))
            if path.is_file()
        }
        outputs[task] = {
            "staging_path": str(source),
            "canonical_path": str(target),
            "authority": authority,
            "files": files,
        }
    receipt = {
        "version": VERSION,
        "staging_root": str(staging_root),
        "canonical_root": str(canonical_root),
        "outputs": outputs,
    }
    write_json_atomic(canonical_root / "publication_receipt.json", receipt)
    return receipt


def _replace_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_prefix(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_prefix(item, old, new) for item in value]
    if isinstance(value, str) and value.startswith(old):
        return new + value[len(old) :]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(publish(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
