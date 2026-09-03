"""Run sparse collapsed-assay baselines on the three promoted scaffold-valid tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import rdkit
import scipy
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    normalize_molecule_identity,
)
from tools.chembl_tool.common.neighbor_selection import (
    NeighborCandidate,
    select_neighbor_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_VERSION = "collapsed_assay_knn.v1"
DEFAULT_K_VALUES = (5, 10, 25)
FP_RADIUS = 2
FP_BITS = 2048
RANDOM_SEED = 0
TASKS = {
    "BBB_Martins": {
        "lineage": "experimental_meaningful_cns_access_v2",
        "data_dir": REPO_ROOT
        / "data/gold_labels/legacy/processed_starling_experimental_meaningful_cns_access_v2/BBB_Martins/scaffold",
        "records": REPO_ROOT
        / "outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7/06_collapsed_records/records.parquet",
        "records_manifest": REPO_ROOT
        / "outputs/chembl_tool/tasks/bbb_martins/evidence_library/starling_normalized_v7/06_collapsed_records/manifest.json",
    },
    "Bioavailability_Ma": {
        "lineage": "record_supported_v2",
        "data_dir": REPO_ROOT
        / "data/gold_labels/legacy/processed_starling_record_supported_v2/Bioavailability_Ma/scaffold",
        "records": REPO_ROOT
        / "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v7/06_collapsed_records/records.parquet",
        "records_manifest": REPO_ROOT
        / "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v7/06_collapsed_records/manifest.json",
    },
    "Skin_Reaction": {
        "lineage": "record_supported_v2",
        "data_dir": REPO_ROOT
        / "data/gold_labels/legacy/processed_starling_record_supported_v2/Skin_Reaction/scaffold",
        "records": REPO_ROOT
        / "outputs/chembl_tool/tasks/skin_reaction/evidence_library/starling_normalized_v7/06_collapsed_records/records.parquet",
        "records_manifest": REPO_ROOT
        / "outputs/chembl_tool/tasks/skin_reaction/evidence_library/starling_normalized_v7/06_collapsed_records/manifest.json",
    },
}
REQUIRED_STAGE06_VALIDATIONS = (
    "configured_group_ids_preserved",
    "direct_and_indirect_grouping_are_disjoint",
    "transfer_eligibility_matches_measurement_contract",
    "one_output_per_molecule_context",
    "pending_semantic_records_are_not_retrieval_eligible",
)


def read_split(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        label = int(row["Y"])
        if label not in (0, 1):
            raise ValueError(f"{path}:{line_number} has non-binary Y={label}")
        identity = normalize_molecule_identity(str(row["drug"]))
        parent_key = identity.parent_inchi_key or identity.parent_smiles
        if identity.status != "ok" or not parent_key:
            raise ValueError(f"{path}:{line_number} has unresolved molecule identity")
        rows.append(
            {
                "index": len(rows),
                "drug": str(row["drug"]),
                "Y": label,
                "parent_key": parent_key,
            }
        )
    keys = [row["parent_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate normalized parents")
    return rows


def load_numeric_assays(
    path: Path, benchmark_parent_keys: set[str]
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    required = {
        "canonical_smiles",
        "pair_bucket_key",
        "finite_scalar_value",
        "aggregation_method",
        "aggregation_status",
        "assay_transfer_eligible",
        "retrieval_source_id",
    }
    dataset = ds.dataset(path, format="parquet")
    missing = required - set(dataset.schema.names)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    predicate = (
        (ds.field("aggregation_method") == "continuous_median")
        & (ds.field("aggregation_status") == "valid")
        & (ds.field("assay_transfer_eligible") == True)  # noqa: E712
        & ds.field("pair_bucket_key").is_valid()
        & ds.field("finite_scalar_value").is_valid()
    )
    table = dataset.to_table(
        columns=["canonical_smiles", "pair_bucket_key", "finite_scalar_value"],
        filter=predicate,
    )
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    identity_cache: dict[str, str] = {}
    finite_rows = 0
    for row in table.to_pylist():
        value = float(row["finite_scalar_value"])
        if not math.isfinite(value):
            continue
        finite_rows += 1
        smiles = str(row["canonical_smiles"] or "")
        if smiles not in identity_cache:
            identity = normalize_molecule_identity(smiles)
            identity_cache[smiles] = identity.parent_inchi_key or identity.parent_smiles
        parent_key = identity_cache[smiles]
        if parent_key in benchmark_parent_keys:
            grouped[(parent_key, str(row["pair_bucket_key"]))].append(value)
    assays: dict[str, dict[str, float]] = defaultdict(dict)
    collisions = 0
    for (parent_key, assay_key), values in grouped.items():
        assays[parent_key][assay_key] = float(median(values))
        collisions += len(values) - 1
    return dict(assays), {
        "eligible_rows_read": table.num_rows,
        "finite_rows_read": finite_rows,
        "unique_stage06_structures_normalized": len(identity_cache),
        "matched_benchmark_parents": len(assays),
        "matched_parent_assay_values": len(grouped),
        "parent_assay_collisions_collapsed_by_median": collisions,
    }


def build_feature_catalog(
    train_rows: list[dict[str, Any]], assays: dict[str, dict[str, float]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        for assay_key, value in assays.get(row["parent_key"], {}).items():
            values[assay_key].append(value)
    catalog = []
    for index, assay_key in enumerate(sorted(values)):
        observed = np.asarray(values[assay_key], dtype=np.float64)
        catalog.append(
            {
                "feature_index": index,
                "value_column": index,
                "presence_column": len(values) + index,
                "pair_bucket_key": assay_key,
                "train_support": int(observed.size),
                "train_mean": float(observed.mean()),
                "train_standard_deviation": float(observed.std(ddof=0)),
            }
        )
    return catalog, {row["pair_bucket_key"]: row["feature_index"] for row in catalog}


def encode_assays(
    rows: list[dict[str, Any]],
    assays: dict[str, dict[str, float]],
    catalog: list[dict[str, Any]],
    feature_index: dict[str, int],
) -> tuple[sparse.csr_matrix, list[dict[str, int]]]:
    n_assays = len(catalog)
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    coverage: list[dict[str, int]] = []
    for row_index, row in enumerate(rows):
        known = assays.get(row["parent_key"], {})
        used = 0
        ignored = 0
        for assay_key, value in known.items():
            column = feature_index.get(assay_key)
            if column is None:
                ignored += 1
                continue
            stats = catalog[column]
            scale = stats["train_standard_deviation"]
            standardized = (value - stats["train_mean"]) / scale if scale else 0.0
            if standardized:
                matrix_rows.append(row_index)
                matrix_columns.append(column)
                matrix_values.append(float(standardized))
            matrix_rows.append(row_index)
            matrix_columns.append(n_assays + column)
            matrix_values.append(1.0)
            used += 1
        coverage.append({"observed_train_catalog_assays": used, "ignored_unseen_assays": ignored})
    matrix = sparse.csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(rows), 2 * n_assays),
        dtype=np.float64,
    )
    return matrix, coverage


def morgan_fingerprints(rows: list[dict[str, Any]]) -> list[Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS,
        includeChirality=False,
        useBondTypes=True,
    )
    fingerprints = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["drug"])
        if molecule is None:
            raise ValueError(f"Invalid benchmark SMILES: {row['drug']!r}")
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def retrieve_neighbors(
    queries: list[dict[str, Any]],
    query_fingerprints: list[Any],
    train_rows: list[dict[str, Any]],
    train_fingerprints: list[Any],
    *,
    top_k: int,
    leave_one_out: bool,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    selected_indices: list[list[int]] = []
    audit_rows: list[dict[str, Any]] = []
    for query_index, (query, query_fp) in enumerate(
        zip(queries, query_fingerprints, strict=True)
    ):
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, train_fingerprints)
        candidates = [
            NeighborCandidate(index, f"{index:012d}", float(similarity))
            for index, similarity in enumerate(similarities)
            if not (leave_one_out and index == query_index)
        ]
        selected = select_neighbor_candidates(
            candidates,
            query_fingerprint=query_fp,
            candidate_fingerprints=train_fingerprints,
            top_k=top_k,
        )
        if len(selected) != top_k:
            raise ValueError(f"Query {query_index} has fewer than {top_k} train neighbors")
        indices = [candidate.molecule_index for candidate in selected]
        selected_indices.append(indices)
        audit_rows.append(
            {
                "query_index": query_index,
                "query_parent_key": query["parent_key"],
                "neighbors": [
                    {
                        "rank": rank,
                        "train_index": candidate.molecule_index,
                        "train_parent_key": train_rows[candidate.molecule_index]["parent_key"],
                        "Y": train_rows[candidate.molecule_index]["Y"],
                        "similarity": candidate.similarity,
                    }
                    for rank, candidate in enumerate(selected, 1)
                ],
            }
        )
    return selected_indices, audit_rows


def mean_neighbor_matrix(
    train_matrix: sparse.csr_matrix, neighbor_indices: list[list[int]], k: int
) -> sparse.csr_matrix:
    row_indices = np.repeat(np.arange(len(neighbor_indices)), k)
    column_indices = np.asarray(
        [index for neighbors in neighbor_indices for index in neighbors[:k]], dtype=np.int64
    )
    selector = sparse.csr_matrix(
        (np.full(column_indices.size, 1.0 / k), (row_indices, column_indices)),
        shape=(len(neighbor_indices), train_matrix.shape[0]),
    )
    return (selector @ train_matrix).tocsr()


def neighbor_label_scores(
    train_labels: np.ndarray, neighbor_indices: list[list[int]], k: int
) -> np.ndarray:
    return np.asarray(
        [train_labels[neighbors[:k]].mean() for neighbors in neighbor_indices],
        dtype=np.float64,
    )


def fit_logistic(
    train_matrix: sparse.csr_matrix,
    train_labels: np.ndarray,
    valid_matrix: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2_000,
        random_state=RANDOM_SEED,
    )
    model.fit(train_matrix, train_labels)
    scores = model.predict_proba(valid_matrix)[:, 1]
    return (scores >= 0.5).astype(np.int64), scores, {
        "n_iter": int(model.n_iter_[0]),
        "intercept": float(model.intercept_[0]),
        "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
    }


def binary_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    per_class = f1_score(labels, predictions, labels=[0, 1], average=None, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "per_class_f1": {"0": float(per_class[0]), "1": float(per_class[1])},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def prediction_rows(
    task: str,
    method: str,
    k: int | None,
    valid_rows: list[dict[str, Any]],
    predictions: np.ndarray,
    scores: np.ndarray,
    coverage: list[dict[str, int]],
    neighbor_coverage: list[float] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "task": task,
            "query_index": row["index"],
            "drug": row["drug"],
            "parent_key": row["parent_key"],
            "Y": row["Y"],
            "method": method,
            "k": k,
            "prediction": int(predictions[index]),
            "score": float(scores[index]),
            "correct": int(predictions[index]) == row["Y"],
            **coverage[index],
            **(
                {"neighbor_assay_coverage": neighbor_coverage[index]}
                if neighbor_coverage is not None
                else {}
            ),
        }
        for index, row in enumerate(valid_rows)
    ]


def run_task(task: str, output_dir: Path, k_values: tuple[int, ...]) -> list[dict[str, Any]]:
    spec = TASKS[task]
    data_dir = spec["data_dir"]
    records_path = spec["records"]
    records_manifest = json.loads(spec["records_manifest"].read_text(encoding="utf-8"))
    validations = records_manifest.get("validations", {})
    failed = [key for key in REQUIRED_STAGE06_VALIDATIONS if validations.get(key) is not True]
    if failed:
        raise ValueError(f"{task} Stage 06 manifest failed validations: {failed}")
    train_rows = read_split(data_dir / "train.jsonl")
    valid_rows = read_split(data_dir / "valid.jsonl")
    train_keys = {row["parent_key"] for row in train_rows}
    valid_keys = {row["parent_key"] for row in valid_rows}
    if train_keys & valid_keys:
        raise ValueError(f"{task} train/valid parent overlap is nonzero")
    assays, assay_stats = load_numeric_assays(records_path, train_keys | valid_keys)
    catalog, feature_index = build_feature_catalog(train_rows, assays)
    if not catalog:
        raise ValueError(f"{task} has no eligible train assay features")
    self_train, train_coverage = encode_assays(train_rows, assays, catalog, feature_index)
    self_valid, valid_coverage = encode_assays(valid_rows, assays, catalog, feature_index)
    train_labels = np.asarray([row["Y"] for row in train_rows], dtype=np.int64)
    valid_labels = np.asarray([row["Y"] for row in valid_rows], dtype=np.int64)

    train_fingerprints = morgan_fingerprints(train_rows)
    valid_fingerprints = morgan_fingerprints(valid_rows)
    top_k = max(k_values)
    train_neighbors, train_neighbor_audit = retrieve_neighbors(
        train_rows,
        train_fingerprints,
        train_rows,
        train_fingerprints,
        top_k=top_k,
        leave_one_out=True,
    )
    valid_neighbors, valid_neighbor_audit = retrieve_neighbors(
        valid_rows,
        valid_fingerprints,
        train_rows,
        train_fingerprints,
        top_k=top_k,
        leave_one_out=False,
    )

    output_dir.mkdir(parents=True)
    sparse.save_npz(output_dir / "query_self_train.npz", self_train)
    sparse.save_npz(output_dir / "query_self_valid.npz", self_valid)
    _write_jsonl(output_dir / "feature_catalog.jsonl", catalog)
    _write_jsonl(
        output_dir / "rows.jsonl",
        [
            {"split": "train", **row, **train_coverage[index]}
            for index, row in enumerate(train_rows)
        ]
        + [
            {"split": "valid", **row, **valid_coverage[index]}
            for index, row in enumerate(valid_rows)
        ],
    )
    _write_jsonl(
        output_dir / "neighbors.jsonl",
        [{"split": "train", **row} for row in train_neighbor_audit]
        + [{"split": "valid", **row} for row in valid_neighbor_audit],
    )

    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    self_prediction, self_score, self_model = fit_logistic(
        self_train, train_labels, self_valid
    )
    self_metrics = binary_metrics(valid_labels, self_prediction, self_score)
    self_metrics.update(
        {
            "task": task,
            "method": "query_self_indirect",
            "k": None,
            "model": self_model,
            "validation_assay_coverage": sum(
                row["observed_train_catalog_assays"] > 0 for row in valid_coverage
            )
            / len(valid_coverage),
        }
    )
    metrics.append(self_metrics)
    predictions.extend(
        prediction_rows(
            task,
            "query_self_indirect",
            None,
            valid_rows,
            self_prediction,
            self_score,
            valid_coverage,
        )
    )

    train_has_assay = np.asarray(
        [row["observed_train_catalog_assays"] > 0 for row in train_coverage]
    )
    for k in k_values:
        train_mean = mean_neighbor_matrix(self_train, train_neighbors, k)
        valid_mean = mean_neighbor_matrix(self_train, valid_neighbors, k)
        sparse.save_npz(output_dir / f"neighbor_indirect_train_k{k}.npz", train_mean)
        sparse.save_npz(output_dir / f"neighbor_indirect_valid_k{k}.npz", valid_mean)
        train_label_score = neighbor_label_scores(train_labels, train_neighbors, k)
        valid_label_score = neighbor_label_scores(train_labels, valid_neighbors, k)
        neighbor_coverage = [
            float(train_has_assay[neighbors[:k]].mean()) for neighbors in valid_neighbors
        ]

        indirect_prediction, indirect_score, indirect_model = fit_logistic(
            train_mean, train_labels, valid_mean
        )
        combined_train = sparse.hstack(
            [train_mean, sparse.csr_matrix(train_label_score[:, None])], format="csr"
        )
        combined_valid = sparse.hstack(
            [valid_mean, sparse.csr_matrix(valid_label_score[:, None])], format="csr"
        )
        combined_prediction, combined_score, combined_model = fit_logistic(
            combined_train, train_labels, combined_valid
        )
        anchor_prediction = (valid_label_score >= 0.5).astype(np.int64)
        for method, method_prediction, method_score, model in (
            ("neighbor_indirect", indirect_prediction, indirect_score, indirect_model),
            (
                "neighbor_indirect_plus_label",
                combined_prediction,
                combined_score,
                combined_model,
            ),
            ("morgan_label_knn", anchor_prediction, valid_label_score, None),
        ):
            row = binary_metrics(valid_labels, method_prediction, method_score)
            row.update(
                {
                    "task": task,
                    "method": method,
                    "k": k,
                    "model": model,
                    "mean_neighbor_assay_coverage": float(np.mean(neighbor_coverage)),
                    "queries_with_any_assay_neighbor": sum(value > 0 for value in neighbor_coverage),
                }
            )
            metrics.append(row)
            predictions.extend(
                prediction_rows(
                    task,
                    method,
                    k,
                    valid_rows,
                    method_prediction,
                    method_score,
                    valid_coverage,
                    neighbor_coverage,
                )
            )

    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    _write_json(output_dir / "metrics.json", metrics)
    output_names = [
        "feature_catalog.jsonl",
        "rows.jsonl",
        "neighbors.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "query_self_train.npz",
        "query_self_valid.npz",
        *[
            f"neighbor_indirect_{split}_k{k}.npz"
            for k in k_values
            for split in ("train", "valid")
        ],
    ]
    task_manifest = {
        "version": EXPERIMENT_VERSION,
        "task": task,
        "benchmark_lineage": spec["lineage"],
        "evaluation_split": "valid",
        "reference_split": "train",
        "k_values": list(k_values),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_assay_features": len(catalog),
        "matrix_columns": 2 * len(catalog),
        "assay_loading": assay_stats,
        "coverage": {
            "train_with_assays": sum(
                row["observed_train_catalog_assays"] > 0 for row in train_coverage
            ),
            "valid_with_assays": sum(
                row["observed_train_catalog_assays"] > 0 for row in valid_coverage
            ),
            "valid_unseen_assays_ignored": sum(
                row["ignored_unseen_assays"] for row in valid_coverage
            ),
        },
        "inputs": {
            "train_jsonl": _file_entry(data_dir / "train.jsonl"),
            "valid_jsonl": _file_entry(data_dir / "valid.jsonl"),
            "stage06_records": _file_entry(records_path),
            "stage06_manifest": _file_entry(spec["records_manifest"]),
        },
        "outputs": {
            name: _file_entry(output_dir / name) for name in output_names
        },
        "validations": {
            "train_valid_parent_overlap": 0,
            "train_leave_one_out_self_neighbors": sum(
                row["query_index"]
                in {neighbor["train_index"] for neighbor in row["neighbors"]}
                for row in train_neighbor_audit
            ),
            "validation_neighbors_outside_train": 0,
            "direct_stage06_rows_used": 0,
            "test_split_read": False,
            "all_validation_queries_retained": len(valid_rows),
        },
    }
    _write_json(output_dir / "manifest.json", task_manifest)
    return metrics


def run(output_dir: Path, tasks: list[str], k_values: tuple[int, ...]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not k_values or min(k_values) <= 0 or len(set(k_values)) != len(k_values):
        raise ValueError("K values must be unique positive integers")
    all_metrics: list[dict[str, Any]] = []
    for task in tasks:
        print(f"Running {task}", flush=True)
        all_metrics.extend(run_task(task, output_dir / task, k_values))
    _write_metrics_tsv(output_dir / "metrics.tsv", all_metrics)
    (output_dir / "REPORT.md").write_text(_report(all_metrics, k_values), encoding="utf-8")
    _write_json(
        output_dir / "manifest.json",
        {
            "version": EXPERIMENT_VERSION,
            "tasks": tasks,
            "evaluation_split": "valid",
            "reference_split": "train",
            "k_values": list(k_values),
            "selection_policy": "report_only_no_promotion",
            "feature_contract": {
                "source": "canonical_stage06_collapsed_records",
                "partition": "assay_transfer_eligible",
                "measurement": "valid_assay_transfer_eligible_continuous_median",
                "feature_key": "pair_bucket_key",
                "value": "train_standardized_finite_scalar_value_zero_when_missing",
                "presence": "observed_indicator",
                "neighbor_aggregation": "unweighted_elementwise_mean",
                "combined_label_feature": "positive_neighbor_fraction_from_benchmark_train_Y",
            },
            "classifier": {
                "name": "sklearn.linear_model.LogisticRegression",
                "penalty": "l2",
                "C": 1.0,
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 2_000,
                "threshold": 0.5,
                "random_seed": RANDOM_SEED,
            },
            "fingerprint": {
                "type": "RDKit-Morgan",
                "radius": FP_RADIUS,
                "n_bits": FP_BITS,
                "use_chirality": False,
                "use_bond_types": True,
            },
            "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
            "software": {
                "python": platform.python_version(),
                "rdkit": rdkit.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "task_manifests": {
                task: _file_entry(output_dir / task / "manifest.json") for task in tasks
            },
        },
    )


def _file_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_metrics_tsv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = (
        "task",
        "method",
        "k",
        "n",
        "macro_f1",
        "accuracy",
        "auroc",
        "mean_neighbor_assay_coverage",
        "validation_assay_coverage",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def _report(metrics: list[dict[str, Any]], k_values: tuple[int, ...]) -> str:
    lines = [
        "# Collapsed-assay Morgan neighbor validation",
        "",
        "This validation-only diagnostic reports every configured K; it does not select a winner or read test data.",
        "",
        f"K values: `{', '.join(map(str, k_values))}`.",
        "",
        "| Task | Method | K | n | Macro-F1 | Accuracy | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['task']} | {row['method']} | {row['k'] if row['k'] is not None else 'n/a'} "
            f"| {row['n']} | {row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['auroc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The primary cohort retains every validation molecule. Missing assay values are represented by zeros after train-only standardization, with separate presence features.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASKS),
        default=list(TASKS),
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/paper/collapsed_assay_knn_v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args.output_dir.resolve(), args.tasks, tuple(args.k_values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
