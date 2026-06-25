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
