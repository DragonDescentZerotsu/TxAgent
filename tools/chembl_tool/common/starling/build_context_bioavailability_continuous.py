"""Build record-complete continuous values for current context-conditioned gold."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = "context_conditioned_bioavailability_continuous.v1"
TASK = "Bioavailability_Ma"
NUMERIC_METHODS = {
    "observed_point", "observed_reported_mean", "observed_closed_range_midpoint",
    "reported_point", "reported_mean_plus_minus", "reported_range",
    "lower_bound", "upper_bound", "train_class_mean_one_sided_bound",
}


def _paths(txagent: Path, starling: Path, output: Path) -> dict[str, Path]:
    gold = txagent / "data/processed_starling_context_conditioned_selected_v1" / TASK / "scaffold"
    return {
        "train": gold / "train_molecule_condition_labels.jsonl",
        "valid": gold / "valid_molecule_condition_labels.jsonl",
        "reviews": gold / "source_condition_review.jsonl",
        "base_votes": txagent / "data/record_supported_v2_processed" / TASK / "scaffold/voting_records.jsonl",
        "source_records": starling / "assay_transfer/context_conditioned/artifacts/continuous_source_v1/bioavailability_ma/records.parquet",
        "v7_splits": starling / "assay_transfer/context_conditioned/artifacts/v7/splits/Bioavailability_Ma/vote_mean/rows.parquet",
        "output": output,
    }


def _votes(paths: Mapping[str, Path]) -> dict[str, int]:
    votes = {str(row["voting_record_key"]): int(row["record_vote"])
             for row in read_jsonl(paths["base_votes"])}
    for row in read_jsonl(paths["reviews"]):
        if row.get("review_status") == "accepted":
            votes[f"{TASK}:review:{row['source_record_id']}"] = int(row["Y"])
    return votes


def _records(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    votes = _votes(paths)
    rows = pq.read_table(paths["source_records"]).to_pylist()
    missing = sorted({str(row["voting_record_key"]) for row in rows} - votes.keys())
    if missing:
        raise ValueError(f"missing binary votes for {len(missing)} records")
    for row in rows:
        row["record_vote"] = votes[str(row["voting_record_key"])]
    return rows


def _calibration(records: Iterable[Mapping[str, Any]], train_ids: set[str]) -> dict[str, Any]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in records:
        value = row.get("raw_observed_continuous_value")
        if str(row["benchmark_row_id"]) in train_ids and value is not None:
            values[int(row["record_vote"])].append(float(value))
    means = {label: math.fsum(items) / len(items) for label, items in values.items()}
    if set(means) != {0, 1} or not means[0] < 20.0 <= means[1]:
        raise ValueError(f"invalid train-only class means: {means}")
    return {
        "fit_split": "v7_train_96", "positive_threshold_percent": 20.0,
        "class_means": {str(key): value for key, value in means.items()},
        "observed_record_counts": {str(key): len(values[key]) for key in sorted(values)},
    }


def _contribution(row: Mapping[str, Any], means: Mapping[str, float]) -> dict[str, Any]:
    observed = row.get("raw_observed_continuous_value")
    method = str(row.get("raw_contribution_method") or "")
    if observed is not None:
        value, contribution_method = float(observed), method
    else:
        value = float(means[str(row["record_vote"])])
        contribution_method = (
            "v7_train_class_mean_one_sided_bound"
            if method in NUMERIC_METHODS else "v7_train_class_mean_categorical"
        )
    return {
        **dict(row), "voting_value_type": "continuous" if method in NUMERIC_METHODS else "categorical",
        "observed_continuous_value": None if observed is None else float(observed),
        "continuous_value_contribution": value,
        "continuous_value_contribution_method": contribution_method,
        "continuous_value_unit": "percent_oral_bioavailability",
    }


def _enrich(node: Mapping[str, Any], records: list[Mapping[str, Any]]) -> dict[str, Any]:
    observed = sum(row["observed_continuous_value"] is not None for row in records)
    record_type = (
        "continuous_only" if observed == len(records)
        else "imputed_only" if observed == 0 else "mixed"
    )
    values = [float(row["continuous_value_contribution"]) for row in records]
    return {
        **dict(node), "voting_record_type": record_type,
        "voting_record_count": len(records), "observed_record_count": observed,
        "imputed_record_count": len(records) - observed,
        "continuous_value_mean": math.fsum(values) / len(values),
        "continuous_value_unit": "percent_oral_bioavailability",
        "continuous_value_method": "mean_observed_and_v7_train_class_mean_imputed_votes",
        "voting_record_keys": [str(row["voting_record_key"]) for row in records],
    }


def _artifact_rows(paths: Mapping[str, Path]) -> tuple[dict[str, list[dict]], list[dict], dict]:
    source = _records(paths)
    train_ids = {str(row["benchmark_row_id"]) for row in pq.read_table(
        paths["v7_splits"], columns=["benchmark_row_id", "_transfer_split"],
    ).to_pylist() if row["_transfer_split"] == "train"}
    calibration = _calibration(source, train_ids)
    records = [_contribution(row, calibration["class_means"]) for row in source]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[str(row["benchmark_row_id"])].append(row)
    nodes = {split: [_enrich(node, grouped[str(node["benchmark_row_id"])])
                     for node in read_jsonl(paths[split])] for split in ("train", "valid")}
    if sum(map(len, nodes.values())) != len(grouped):
        raise ValueError("record-complete node coverage changed")
    return nodes, records, calibration


def _summary(paths: Mapping[str, Path], nodes: Mapping[str, list[dict]],
             records: list[dict], calibration: Mapping[str, Any]) -> dict[str, Any]:
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
              for name, path in paths.items() if name != "output"}
    all_nodes = nodes["train"] + nodes["valid"]
    return {
        "schema_version": SCHEMA_VERSION, "task": TASK,
        "source_gold": "processed_starling_context_conditioned_selected_v1",
        "official_test_policy": "sealed_not_read_or_emitted",
        "value_policy": dict(calibration), "inputs": inputs,
        "counts": {"train_nodes": len(nodes["train"]), "valid_nodes": len(nodes["valid"]),
                   "voting_records": len(records),
                   "record_types": dict(sorted(Counter(
                       row["voting_record_type"] for row in all_nodes).items()))},
    }


def build(paths: Mapping[str, Path]) -> Path:
    nodes, records, calibration = _artifact_rows(paths)
    output = paths["output"]
    write_jsonl_atomic(output / "train_molecule_condition_labels.jsonl", nodes["train"])
    write_jsonl_atomic(output / "valid_molecule_condition_labels.jsonl", nodes["valid"])
    write_jsonl_atomic(output / "voting_records.jsonl", records)
    summary = _summary(paths, nodes, records, calibration)
    summary["artifacts"] = {name: sha256_file(output / name) for name in (
        "train_molecule_condition_labels.jsonl", "valid_molecule_condition_labels.jsonl",
        "voting_records.jsonl",
    )}
    write_json_atomic(output / "summary.json", summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txagent-root", type=Path, default=Path("."))
    parser.add_argument("--starling-root", type=Path, default=Path("../starling_assay_transfer"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed_starling_context_conditioned_continuous_v1/Bioavailability_Ma/scaffold"
    ))
    args = parser.parse_args()
    print(build(_paths(args.txagent_root, args.starling_root, args.output)))


if __name__ == "__main__":
    main()
