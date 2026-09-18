"""Import pinned scaffold gold labels from the upstream TxAgent release."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from data.processing.gold_labels.conditioned_benchmark import BENCHMARK_ROOT, CONTRACT


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = "26d44f121141f3738764c0b00dc09e185445341b"
TASKS = {
    "ames": "Ames",
    "dili": "DILI",
    "carcinogens": "Carcinogens",
    "skin_reaction": "Skin_Reaction",
}
METADATA_ROOT = ROOT / "data/artifacts/gold_labels/conditioned_benchmark"


def _git(*args: str) -> bytes:
    return subprocess.check_output(("git", *args), cwd=ROOT)


def _upstream_files(directory: str) -> list[str]:
    prefix = f"data/conditioned_benchmark/{directory}"
    paths = _git("ls-tree", "-r", "--name-only", UPSTREAM_COMMIT, "--", prefix).decode().splitlines()
    return [
        path
        for path in paths
        if (
            ("/scaffold/" in path and not Path(path).name.startswith("."))
            or "/provenance/" in path
            or path == f"{prefix}/manifest.json"
        )
    ]


def _materialize_task(directory: str) -> dict:
    source_prefix = f"data/conditioned_benchmark/{directory}"
    task_root = BENCHMARK_ROOT / directory
    task_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upstream-gold-", dir=task_root) as temporary_name:
        temporary = Path(temporary_name) / "v1"
        for source_path in _upstream_files(directory):
            relative = Path(source_path).relative_to(source_prefix)
            if relative == Path("manifest.json"):
                relative = Path("upstream_manifest.json")
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_git("show", f"{UPSTREAM_COMMIT}:{source_path}"))
        summary = json.loads((temporary / "scaffold/summary.json").read_text())
        upstream_manifest = temporary / "upstream_manifest.json"
        if "target_definition" not in summary and upstream_manifest.is_file():
            summary["target_definition"] = json.loads(
                upstream_manifest.read_text()
            )["target_definition"]
        for split in ("train", "valid", "test"):
            expected = summary["splits"][split]
            split_path = temporary / f"scaffold/{split}.jsonl"
            digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in split_path.read_text().splitlines()]
            expected.update({
                "n": len(rows),
                "source_sha256": expected.get("source_sha256", digest),
                "canonical_sha256": expected.get("canonical_sha256", digest),
                "byte_identical": expected.get("byte_identical", True),
                "same_ordered_drug_labels": expected.get("same_ordered_drug_labels", True),
            })
            if expected["canonical_sha256"] != digest or expected["source_sha256"] != digest:
                raise ValueError(f"upstream {directory}/{split} is not byte-identical")
        destination = task_root / "v1"
        backup = task_root / ".v1-before-upstream-import"
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    (task_root / "CURRENT").write_text("v1\n")
    return summary


def main() -> None:
    summaries = {task: _materialize_task(directory) for task, directory in TASKS.items()}
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = METADATA_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt_path = METADATA_ROOT / "migration_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({
        "published_at": _git("show", "-s", "--format=%cI", UPSTREAM_COMMIT).decode().strip(),
        "active_root": "data/gold_labels",
    })
    for task, directory in TASKS.items():
        summary = summaries[task]
        splits = summary["splits"]
        manifest["tasks"][task] = {
            "root": f"data/gold_labels/{directory}/v1/scaffold",
            "target_definition": summary["target_definition"],
            "split_counts": {split: splits[split]["n"] for split in ("train", "valid", "test")},
        }
        receipt["tasks"][task] = {
            "task": task,
            "task_directory": directory,
            "target_definition": summary["target_definition"],
            "legacy_source_root": f"{UPSTREAM_COMMIT}:data/conditioned_benchmark/{directory}/scaffold",
            "canonical_root": f"data/gold_labels/{directory}/v1/scaffold",
            "input_rows_byte_identical_expected": True,
            "upstream_commit": UPSTREAM_COMMIT,
            "splits": splits,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
