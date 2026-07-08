"""Annotate Starling oral-bioavailability retrieval with transfer-model scores."""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass
from typing import Any


MODEL_REPO = "jiosephlee/starling-transfer-ssv2-srcval"
METADATA_FIELDS = [
    "molecule_name",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
]
STARLING_SOURCE = "starling-labs/Oral_Bioavailability"


@dataclass(frozen=True)
class StarlingTransferConfig:
    model_name_or_path: str = MODEL_REPO
    device: str = "auto"
    batch_size: int = 16
    max_examples_per_row: int = 6
    query_metadata_mode: str = "same_source_context"
    threshold: float = 0.5


def annotate_retrieval_with_starling_transfer(
    retrieval: dict[str, Any],
    *,
    config: StarlingTransferConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of retrieval with Starling source rows annotated by the HF model."""
    annotated = copy.deepcopy(retrieval)
    tasks = _collect_scoring_tasks(annotated, config)
    if not tasks:
        return annotated, {
            "status": "skipped",
            "reason": "no Starling numeric source_record_examples found",
            "model": config.model_name_or_path,
            "n_pairs_scored": 0,
        }

    probabilities = _score_tasks(tasks, config)
    for task, probability in zip(tasks, probabilities, strict=True):
        task["probability"] = probability
        task["prediction"] = "likely_transfer" if probability >= config.threshold else "unlikely_transfer"
    _attach_row_annotations(tasks, config)

    return annotated, {
        "status": "ok",
        "model": config.model_name_or_path,
        "device": config.device,
        "batch_size": config.batch_size,
        "query_metadata_mode": config.query_metadata_mode,
        "threshold": config.threshold,
        "n_pairs_scored": len(tasks),
        "n_rows_annotated": len({task["row_key"] for task in tasks}),
        "n_neighbors_annotated": len({task["neighbor_key"] for task in tasks}),
    }


def select_top_transfer_neighbors(
    retrieval: dict[str, Any],
    *,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the top transfer-scored neighbors per group after Starling annotation."""
    if top_k <= 0:
        return retrieval, {
            "status": "skipped",
            "reason": "top_k <= 0",
            "top_k": top_k,
        }

    selected_retrieval = copy.deepcopy(retrieval)
    groups = selected_retrieval.get("groups") or []
    total_candidates = 0
    total_selected = 0
    total_scored_candidates = 0
    group_summaries = []

    for group in groups:
        candidates = list(group.get("neighbors") or [])
        total_candidates += len(candidates)
        ranked = []
        for candidate_index, neighbor in enumerate(candidates):
            score = _neighbor_transfer_score(neighbor)
            if score["has_transfer_score"]:
                total_scored_candidates += 1
            ranked.append((neighbor, score, candidate_index))

        ranked.sort(
            key=lambda item: (
                not item[1]["has_transfer_score"],
                -_score_or_negative(item[1]["transfer_probability_max"]),
                -_score_or_negative(item[1]["transfer_probability_mean"]),
                -_score_or_negative(item[1]["transfer_probability_median"]),
                -_score_or_negative(item[1]["similarity"]),
                item[2],
            )
        )
        selected_neighbors = []
        for selection_rank, (neighbor, score, _candidate_index) in enumerate(ranked[:top_k], start=1):
            neighbor = copy.deepcopy(neighbor)
            neighbor.setdefault("structural_rank", neighbor.get("rank"))
            neighbor["transfer_selection_rank"] = selection_rank
            neighbor["transfer_selection_score"] = score
            selected_neighbors.append(neighbor)
        group["neighbors"] = selected_neighbors
        group["transfer_neighbor_selection"] = {
            "method": "rank by Starling transfer tool probability after structural retrieval",
            "candidate_neighbor_count": len(candidates),
            "scored_candidate_neighbor_count": sum(
                1 for _neighbor, score, _candidate_index in ranked if score["has_transfer_score"]
            ),
            "selected_neighbor_count": len(selected_neighbors),
            "top_k": top_k,
            "primary_sort_key": "neighbor transfer_probability_max",
            "tie_breakers": ["transfer_probability_mean", "transfer_probability_median", "similarity", "structural_rank"],
        }
        total_selected += len(selected_neighbors)
        group_summaries.append(
            {
                "group_id": group.get("group_id", ""),
                "candidate_neighbor_count": len(candidates),
                "selected_neighbor_count": len(selected_neighbors),
                "scored_candidate_neighbor_count": group["transfer_neighbor_selection"][
                    "scored_candidate_neighbor_count"
                ],
            }
        )

    coverage = selected_retrieval.setdefault("coverage", {})
    coverage["pre_transfer_selection_n_neighbors_total"] = coverage.get("n_neighbors_total", total_candidates)
    coverage["n_neighbors_total"] = total_selected
    coverage["n_groups_with_neighbors"] = sum(1 for group in groups if group.get("neighbors"))
    coverage["transfer_selection_top_k_per_group"] = top_k
    coverage["transfer_selection_candidate_neighbors_total"] = total_candidates
    coverage["transfer_selection_scored_candidate_neighbors_total"] = total_scored_candidates

    summary = {
        "status": "ok",
        "top_k": top_k,
        "n_groups": len(groups),
        "candidate_neighbors_total": total_candidates,
        "scored_candidate_neighbors_total": total_scored_candidates,
        "selected_neighbors_total": total_selected,
        "groups": group_summaries,
    }
    selected_retrieval["starling_transfer_neighbor_selection"] = summary
    return selected_retrieval, summary


def _collect_scoring_tasks(
    retrieval: dict[str, Any],
    config: StarlingTransferConfig,
) -> list[dict[str, Any]]:
    query_smiles = str((retrieval.get("query") or {}).get("canonical_smiles") or "")
    if not query_smiles:
        query_smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
    tasks: list[dict[str, Any]] = []
    for group_index, group in enumerate(retrieval.get("groups") or []):
        for neighbor_index, neighbor in enumerate(group.get("neighbors") or []):
            source_smiles = str(neighbor.get("canonical_smiles") or "")
            if not source_smiles:
                continue
            for row_index, row in enumerate(neighbor.get("evidence_rows") or []):
                if not _is_starling_row(row):
                    continue
                examples = list(row.get("source_record_examples") or [])[: config.max_examples_per_row]
                for example_index, example in enumerate(examples):
                    value = _float_or_none(example.get("oral_bioavailability_value_percent"))
                    if value is None:
                        continue
                    metadata_a = _metadata_from_example(example)
                    tasks.append(
                        {
                            "group_index": group_index,
                            "neighbor_index": neighbor_index,
                            "row_index": row_index,
                            "example_index": example_index,
                            "row": row,
                            "row_key": (group_index, neighbor_index, row_index),
                            "neighbor_key": (group_index, neighbor_index),
                            "smiles_a": source_smiles,
                            "smiles_b": query_smiles,
                            "metadata_a": metadata_a,
                            "metadata_b": _query_metadata(metadata_a, config.query_metadata_mode),
                            "source_value": value,
                            "source_index": example.get("source_index", ""),
                            "source_value_percent": value,
                            "source_metadata": metadata_a,
                        }
                    )
    return tasks


def _score_tasks(tasks: list[dict[str, Any]], config: StarlingTransferConfig) -> list[float]:
    import torch
    from transformers import AutoModel

    device = _resolve_device(config.device, torch)
    model = AutoModel.from_pretrained(config.model_name_or_path, trust_remote_code=True).eval()
    model.to(device)
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(tasks), max(1, config.batch_size)):
            batch = tasks[start : start + max(1, config.batch_size)]
            output = model(
                smiles_a=[task["smiles_a"] for task in batch],
                smiles_b=[task["smiles_b"] for task in batch],
                metadata_a=[task["metadata_a"] for task in batch],
                metadata_b=[task["metadata_b"] for task in batch],
                source_value=[task["source_value"] for task in batch],
            )
            probabilities.extend(float(value) for value in output.logits.sigmoid().detach().cpu().tolist())
    return probabilities


def _attach_row_annotations(tasks: list[dict[str, Any]], config: StarlingTransferConfig) -> None:
    by_row: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for task in tasks:
        by_row.setdefault(task["row_key"], []).append(task)
    for row_tasks in by_row.values():
        row = row_tasks[0]["row"]
        probabilities = [float(task["probability"]) for task in row_tasks]
        likely = sum(probability >= config.threshold for probability in probabilities)
        scores = [
            {
                "source_index": task["source_index"],
                "source_value_percent": round(float(task["source_value_percent"]), 4),
                "source_metadata": task["source_metadata"],
                "transfer_probability": round(float(task["probability"]), 4),
                "transfer_prediction": task["prediction"],
            }
            for task in row_tasks
        ]
        row["starling_transfer_tool"] = {
            "model": config.model_name_or_path,
            "direction": "retrieved_source_molecule_to_query",
            "query_metadata_mode": config.query_metadata_mode,
            "threshold": config.threshold,
            "n_scored_source_examples": len(row_tasks),
            "likely_transfer_count": likely,
            "unlikely_transfer_count": len(row_tasks) - likely,
            "transfer_probability_min": round(min(probabilities), 4),
            "transfer_probability_median": round(float(statistics.median(probabilities)), 4),
            "transfer_probability_mean": round(sum(probabilities) / len(probabilities), 4),
            "transfer_probability_max": round(max(probabilities), 4),
            "source_example_scores": scores,
            "llm_note": (
                "P(transfer) estimates whether the retrieved Starling source molecule's oral "
                "bioavailability behavior transfers to the query under the listed source context. "
                "Use this as auxiliary analog-transfer evidence, not as an automatic label."
            ),
        }


def _is_starling_row(row: dict[str, Any]) -> bool:
    if str(row.get("evidence_source") or "") == STARLING_SOURCE:
        return True
    if str(row.get("source_group_id") or "").startswith("Starling."):
        return True
    return bool(row.get("source_record_examples")) and str(row.get("assay_chembl_id") or "").startswith("STARLING_")


def _neighbor_transfer_score(neighbor: dict[str, Any]) -> dict[str, Any]:
    max_values = []
    mean_values = []
    median_values = []
    likely_count = 0
    scored_examples = 0
    for row in neighbor.get("evidence_rows") or []:
        annotation = row.get("starling_transfer_tool") or {}
        row_max = _float_or_none(annotation.get("transfer_probability_max"))
        row_mean = _float_or_none(annotation.get("transfer_probability_mean"))
        row_median = _float_or_none(annotation.get("transfer_probability_median"))
        if row_max is not None:
            max_values.append(row_max)
        if row_mean is not None:
            mean_values.append(row_mean)
        if row_median is not None:
            median_values.append(row_median)
        likely_count += int(annotation.get("likely_transfer_count") or 0)
        scored_examples += int(annotation.get("n_scored_source_examples") or 0)
    similarity = _float_or_none(neighbor.get("similarity"))
    return {
        "has_transfer_score": bool(max_values),
        "transfer_probability_max": round(max(max_values), 4) if max_values else None,
        "transfer_probability_mean": round(sum(mean_values) / len(mean_values), 4) if mean_values else None,
        "transfer_probability_median": round(float(statistics.median(median_values)), 4) if median_values else None,
        "likely_transfer_count": likely_count,
        "n_scored_source_examples": scored_examples,
        "similarity": similarity,
    }


def _score_or_negative(value: Any) -> float:
    numeric = _float_or_none(value)
    return numeric if numeric is not None else -1.0


def _metadata_from_example(example: dict[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for field in METADATA_FIELDS:
        value = example.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            metadata[field] = text
    return metadata


def _query_metadata(source_metadata: dict[str, str], mode: str) -> dict[str, str]:
    if mode == "missing":
        return {}
    if mode == "same_source_context":
        return {field: value for field, value in source_metadata.items() if field != "molecule_name"}
    raise ValueError(f"Unsupported query_metadata_mode: {mode}")


def _resolve_device(device: str, torch_module: Any) -> str:
    if device != "auto":
        return device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
