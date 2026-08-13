"""Build source-agnostic pre-decision features for KNN-vs-agent routing."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from tools.chembl_tool.common.evidence_contract import (
    EVIDENCE_ROLES,
    minimal_evidence_from_row,
)
from tools.chembl_tool.common.json_utils import write_json_atomic, write_jsonl_atomic

from .contract import (
    FEATURE_SCHEMA_VERSION,
    ROUTER_FEATURE_STEM,
    TaskSpec,
    direct_batch_root,
    fold_root,
    task_root,
)
from .io import read_jsonl as _read_jsonl


ROLE_FEATURES = tuple(sorted(EVIDENCE_ROLES))
QUERY_FEATURE_COLUMNS = (
    "query_molecular_weight",
    "query_logp",
    "query_tpsa",
    "query_hbond_donors",
    "query_hbond_acceptors",
    "query_rotatable_bonds",
    "query_fraction_csp3",
    "query_formal_charge",
    "query_heavy_atom_count",
    "query_ring_count",
)
KNN_FEATURE_COLUMNS = (
    "knn_p_positive",
    "knn_weighted_p_positive",
    "knn_similarity_max",
    "knn_similarity_mean",
    "knn_similarity_kth",
    "knn_top1_top2_gap",
)
EVIDENCE_FEATURE_COLUMNS = (
    "evidence_group_count",
    "evidence_unique_molecule_count",
    "evidence_duplicate_fraction",
    "evidence_similarity_max",
    "evidence_similarity_mean",
    "evidence_similarity_min",
    *(f"role_{role}_fraction" for role in ROLE_FEATURES),
    "quality_confidence_mean",
    "quality_confidence_min",
    "quality_confidence_missing_fraction",
    "scope_present_fraction",
    "direct_human_scope_fraction",
    "uncertainty_row_fraction",
    "transferability_not_assessed_fraction",
    "source_record_count_log1p",
)
FEATURE_COLUMNS = QUERY_FEATURE_COLUMNS + KNN_FEATURE_COLUMNS + EVIDENCE_FEATURE_COLUMNS
FEATURE_PROFILES = {
    "knn_compact": KNN_FEATURE_COLUMNS,
    "query_knn": QUERY_FEATURE_COLUMNS + KNN_FEATURE_COLUMNS,
    "query_knn_evidence": FEATURE_COLUMNS,
}


def build_task_features(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    folds: Iterable[int] | None = None,
) -> dict[str, Any]:
    assignments = _read_jsonl(task_root(output_root, spec.task) / "folds.jsonl")
    selected_folds = sorted(
        set(int(fold) for fold in folds)
        if folds is not None
        else {int(row["fold"]) for row in assignments}
    )
    combined: list[dict[str, Any]] = []
    for fold in selected_folds:
        current_root = fold_root(output_root, spec.task, fold)
        query_rows = _read_jsonl(current_root / "agent_input" / "valid.jsonl")
        knn_rows = _read_jsonl(current_root / "knn_predictions.jsonl")
        agent_path = direct_batch_root(output_root, spec, fold) / "predictions.jsonl"
        agent_rows = _read_jsonl(agent_path)
        knn_by_index = {int(row["local_query_index"]): row for row in knn_rows}
        agent_by_index = {_prediction_query_index(row): row for row in agent_rows}
        expected = set(range(len(query_rows)))
        if set(knn_by_index) != expected:
            raise ValueError(f"Incomplete fold {fold} KNN predictions for {spec.task}")
        if set(agent_by_index) != expected:
            missing = sorted(expected - set(agent_by_index))
            extra = sorted(set(agent_by_index) - expected)
            raise ValueError(
                f"Incomplete fold {fold} agent predictions for {spec.task}: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        fold_features: list[dict[str, Any]] = []
        for local_index, query in enumerate(query_rows):
            knn = knn_by_index[local_index]
            agent = agent_by_index[local_index]
            label = int(query["Y"])
            if int(knn["Y"]) != label or int(agent["label"]) != label:
                raise ValueError(f"Label mismatch at {spec.task} fold {fold} index {local_index}")
            agent_prediction = int(agent["pred_label"])
            knn_prediction = int(knn["prediction"])
            retrieval_path = _resolve_artifact_path(agent["run_dir"], "retrieval.json")
            features = assemble_router_features(
                str(query["drug"]),
                knn,
                json.loads(retrieval_path.read_text(encoding="utf-8")),
            )
            outcome = paired_outcome(label, knn_prediction, agent_prediction)
            fold_features.append(
                {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "task": spec.task,
                    "fold": fold,
                    "local_query_index": local_index,
                    "train_index": int(query["oof_train_index"]),
                    "Y": label,
                    "knn_prediction": knn_prediction,
                    "agent_prediction": agent_prediction,
                    "knn_correct": knn_prediction == label,
                    "agent_correct": agent_prediction == label,
                    "paired_outcome": outcome,
                    "agent_preferred": int(outcome == "agent_only_correct"),
                    "features": features,
                    "provenance": {
                        "knn_prediction": str(current_root / "knn_predictions.jsonl"),
                        "agent_prediction": str(agent_path),
                        "retrieval": str(retrieval_path),
                    },
                }
            )
        fold_features.sort(key=lambda row: int(row["train_index"]))
        write_jsonl_atomic(current_root / f"{ROUTER_FEATURE_STEM}.jsonl", fold_features)
        combined.extend(fold_features)

    combined.sort(key=lambda row: int(row["train_index"]))
    output_dir = task_root(output_root, spec.task)
    jsonl_path = output_dir / f"{ROUTER_FEATURE_STEM}.jsonl"
    tsv_path = output_dir / f"{ROUTER_FEATURE_STEM}.tsv"
    write_jsonl_atomic(jsonl_path, combined)
    _write_tsv(tsv_path, combined)
    outcome_counts = {
        name: sum(row["paired_outcome"] == name for row in combined)
        for name in ("both_correct", "knn_only_correct", "agent_only_correct", "both_wrong")
    }
    summary = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "task": spec.task,
        "n_rows": len(combined),
        "folds": selected_folds,
        "feature_columns": list(FEATURE_COLUMNS),
        "paired_outcome_counts": outcome_counts,
        "disagreement_count": outcome_counts["knn_only_correct"] + outcome_counts["agent_only_correct"],
        "paths": {"jsonl": str(jsonl_path), "tsv": str(tsv_path)},
    }
    write_json_atomic(output_dir / f"{ROUTER_FEATURE_STEM}_summary.json", summary)
    return summary


def assemble_router_features(
    query_smiles: str,
    knn_row: Mapping[str, Any],
    retrieval: Mapping[str, Any],
) -> dict[str, float]:
    """Assemble the identical pre-decision feature vector for train or deployment."""
    molecule = molecule_features(query_smiles)
    evidence = summarize_retrieval(retrieval)
    return {
        key: (
            float(molecule[key])
            if key.startswith("query_")
            else _knn_feature_value(knn_row, key)
            if key.startswith("knn_")
            else float(evidence[key])
        )
        for key in FEATURE_COLUMNS
    }


def summarize_retrieval(payload: Mapping[str, Any]) -> dict[str, float]:
    groups = payload.get("groups") or []
    neighbors: list[Mapping[str, Any]] = []
    for group in groups:
        if isinstance(group, Mapping):
            neighbors.extend(
                item for item in (group.get("neighbors") or []) if isinstance(item, Mapping)
            )
    similarities = [_finite_float(row.get("similarity")) for row in neighbors]
    similarities = [value for value in similarities if value is not None]
    molecule_keys = [
        str(
            row.get("standard_inchi_key")
            or row.get("molecule_chembl_id")
            or row.get("canonical_smiles")
            or f"neighbor:{index}"
        )
        for index, row in enumerate(neighbors)
    ]
    evidence_rows: list[Mapping[str, Any]] = []
    for neighbor in neighbors:
        evidence_rows.extend(
            item for item in (neighbor.get("evidence_rows") or []) if isinstance(item, Mapping)
        )
    contracts = [minimal_evidence_from_row(row) for row in evidence_rows]
    roles = [
        str((contract.get("annotations") or {}).get("evidence_role") or "unspecified")
        for contract in contracts
    ]
    confidences = [
        _normalize_confidence((contract.get("quality") or {}).get("confidence"))
        for contract in contracts
    ]
    present_confidences = [value for value in confidences if value is not None]
    scopes = [(contract.get("annotations") or {}).get("scope") or {} for contract in contracts]
    uncertainties = [
        (contract.get("annotations") or {}).get("uncertainty") or [] for contract in contracts
    ]
    transferability = [
        str((contract.get("annotations") or {}).get("transferability") or "not_assessed")
        for contract in contracts
    ]
    source_counts = [
        _finite_float((contract.get("provenance") or {}).get("source_record_count"))
        for contract in contracts
    ]
    source_count_total = sum(value for value in source_counts if value is not None)
    unique_molecules = len(set(molecule_keys))
    output: dict[str, float] = {
        "evidence_group_count": float(sum(bool((group or {}).get("neighbors")) for group in groups)),
        "evidence_neighbor_count": float(len(neighbors)),
        "evidence_unique_molecule_count": float(unique_molecules),
        "evidence_duplicate_fraction": (
            1.0 - unique_molecules / len(molecule_keys) if molecule_keys else 0.0
        ),
        "evidence_similarity_max": max(similarities) if similarities else 0.0,
        "evidence_similarity_mean": sum(similarities) / len(similarities) if similarities else 0.0,
        "evidence_similarity_min": min(similarities) if similarities else 0.0,
        "quality_confidence_mean": (
            sum(present_confidences) / len(present_confidences) if present_confidences else 0.0
        ),
        "quality_confidence_min": min(present_confidences) if present_confidences else 0.0,
        "quality_confidence_missing_fraction": _fraction(
            [value is None for value in confidences]
        ),
        "scope_present_fraction": _fraction([bool(scope) for scope in scopes]),
        "direct_human_scope_fraction": _fraction(
            [role == "direct_outcome" and _scope_is_human(scope) for role, scope in zip(roles, scopes)]
        ),
        "uncertainty_row_fraction": _fraction([bool(value) for value in uncertainties]),
        "transferability_not_assessed_fraction": _fraction(
            [value == "not_assessed" for value in transferability]
        ),
        "source_record_count_log1p": math.log1p(max(source_count_total, 0.0)),
    }
    for role in ROLE_FEATURES:
        output[f"role_{role}_fraction"] = _fraction([value == role for value in roles])
    return output


def paired_outcome(label: int, knn_prediction: int, agent_prediction: int) -> str:
    knn_correct = int(knn_prediction) == int(label)
    agent_correct = int(agent_prediction) == int(label)
    if knn_correct and agent_correct:
        return "both_correct"
    if knn_correct:
        return "knn_only_correct"
    if agent_correct:
        return "agent_only_correct"
    return "both_wrong"


def molecule_features(smiles: str) -> dict[str, float]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid query SMILES: {smiles!r}")
    return {
        "query_molecular_weight": float(Descriptors.MolWt(molecule)),
        "query_logp": float(Crippen.MolLogP(molecule)),
        "query_tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "query_hbond_donors": float(Lipinski.NumHDonors(molecule)),
        "query_hbond_acceptors": float(Lipinski.NumHAcceptors(molecule)),
        "query_rotatable_bonds": float(Lipinski.NumRotatableBonds(molecule)),
        "query_fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        "query_formal_charge": float(Chem.GetFormalCharge(molecule)),
        "query_heavy_atom_count": float(molecule.GetNumHeavyAtoms()),
        "query_ring_count": float(rdMolDescriptors.CalcNumRings(molecule)),
    }


def _knn_feature_value(row: Mapping[str, Any], key: str) -> float:
    return float(row[key])


def _prediction_query_index(row: Mapping[str, Any]) -> int:
    for key in ("query_index", "index"):
        if key in row:
            return int(row[key])
    run_id = str(row.get("run_id") or "")
    marker = "_idx"
    if marker in run_id:
        return int(run_id.rsplit(marker, 1)[1])
    raise ValueError(f"Agent prediction has no query index: {row}")


def _resolve_artifact_path(run_dir: Any, filename: str) -> Path:
    path = Path(str(run_dir)) / filename
    if path.exists():
        return path
    raise FileNotFoundError(path)


def _normalize_confidence(value: Any) -> float | None:
    number = _finite_float(value)
    if number is None or number < 0:
        return None
    if number <= 1:
        return number
    if number <= 10:
        return number / 10.0
    return None


def _scope_is_human(scope: Any) -> bool:
    text = json.dumps(scope, ensure_ascii=False).lower()
    tokens = ("human", "patient", "volunteer", "adult", "children", "individual")
    return any(token in text for token in tokens)


def _fraction(flags: Sequence[bool]) -> float:
    return sum(bool(value) for value in flags) / len(flags) if flags else 0.0


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "fold",
        "local_query_index",
        "train_index",
        "Y",
        "knn_prediction",
        "agent_prediction",
        "paired_outcome",
        "agent_preferred",
        *FEATURE_COLUMNS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key not in FEATURE_COLUMNS},
                    **row["features"],
                }
            )
