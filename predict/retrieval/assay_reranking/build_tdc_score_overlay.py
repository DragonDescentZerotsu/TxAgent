"""Index exact frozen TDC scores without model calls or rescoring."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import tempfile

from data.processing.gold_labels.conditioned_benchmark import TDC_MANIFEST, tdc_config, tdc_split_path
from data.processing.paths import REPO_ROOT
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic

BUNDLE = TDC_MANIFEST
OUTPUT = REPO_ROOT / "data/caches/assay_reranking/active/flat_v5/tdc"
LEVELS = {task: tuple(f"L{i}" for i in range(2, spec["max_level"] + 1))
          for task, spec in tdc_config()["tasks"].items()}


def _level(index_path: Path, subset: str, level: str, inputs: dict[str, str]) -> tuple[Path, dict]:
    index = json.loads(index_path.read_text())
    if index.get("status") != "complete":
        raise ValueError(f"Incomplete score input: {index_path}")
    entry = index["splits"][subset]["levels"][level]
    path = (index_path.parent / entry["manifest"]).resolve()
    if sha256_file(path) != entry["manifest_sha256"]:
        raise ValueError(f"Score input manifest changed: {path}")
    manifest = json.loads(path.read_text())
    database = path.with_name(manifest["database"])
    if not database.is_file():
        raise ValueError(f"Score input database changed: {database}")
    inputs[str(index_path)] = sha256_file(index_path)
    inputs[str(path)] = entry["manifest_sha256"]
    inputs[str(database)] = manifest["database_sha256"]
    return database, manifest



def build(task: str, output_root: Path = OUTPUT) -> Path:
    """Copy available score values and exact keys into a query-indexed snapshot."""
    spec = tdc_config(task)
    target = output_root / task / "scores"
    if target.exists():
        raise FileExistsError(f"Score snapshot already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    inputs = {str(BUNDLE): sha256_file(BUNDLE)}
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix=".tdc-scores-", dir=target.parent) as name:
        stage = Path(name)
        database = stage / "scores.sqlite3"
        with sqlite3.connect(database) as output:
            output.execute("CREATE TABLE scores(query_id TEXT NOT NULL,level TEXT NOT NULL,item_id TEXT NOT NULL,parent_id TEXT NOT NULL,score REAL NOT NULL,score_key TEXT NOT NULL,PRIMARY KEY(query_id,level,item_id)) WITHOUT ROWID")
            for subset in ("valid", "test"):
                queries = {str(row["benchmark_row_id"]) for row in read_jsonl(tdc_split_path(task, subset))}
                for level in ("L1", *LEVELS[task]):
                    index = REPO_ROOT / spec["retrieval"]["L1" if level == "L1" else "later"]
                    source, manifest = _level(index, subset, level, inputs)
                    unit = "morgan_context_id" if level == "L1" else "item_id"
                    with sqlite3.connect(f"file:{source.resolve()}?mode=ro&immutable=1", uri=True) as reader:
                        rows = reader.execute(f"SELECT benchmark_row_id,{unit},parent_id,assay_transfer_score,score_key FROM rankings WHERE assay_transfer_score IS NOT NULL")
                        count = 0
                        for query_id, item_id, parent_id, score, key in rows:
                            if query_id not in queries or not key or not 0 <= score <= 1:
                                raise ValueError(f"Invalid frozen score: {task}/{subset}/{level}/{query_id}")
                            output.execute("INSERT INTO scores VALUES (?,?,?,?,?,?)", (query_id, level, item_id, parent_id, score, key))
                            count += 1
                    counts[f"{subset}/{level}"] = count
        write_json_atomic(stage / "manifest.json", {"schema_version": "tdc_exact_scores", "status": "complete", "task": task,
                          "inputs": inputs, "counts": dict(counts), "score_count": sum(counts.values()),
                          "database": database.name, "database_sha256": sha256_file(database)})
        stage.rename(target)
    return target / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(LEVELS), default=list(LEVELS))
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for task in args.tasks:
        print(build(task, args.output_root))


if __name__ == "__main__":
    main()
