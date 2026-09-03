"""Load a frozen retrieval artifact for controlled paired experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predict.retrieval.policies import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
    selector_metadata,
)


def _validated_selector_metadata(
    metadata: Any,
    *,
    location: str,
) -> dict[str, Any]:
    if (
        not isinstance(metadata, dict)
        or metadata.get("name") not in NEIGHBOR_SELECTORS
        or not isinstance(metadata.get("version"), str)
        or not metadata["version"]
    ):
        raise ValueError(
            "Retrieval replay neighbor-selector provenance is malformed at "
            f"{location}: {metadata!r}"
        )
    return metadata


def _selector_from_retrieval_payload(retrieval: dict[str, Any]) -> dict[str, Any]:
    """Read selector provenance from current or legacy retrieval payloads."""

    observed: list[tuple[str, dict[str, Any]]] = []
    for section_name in ("retrieval_policy", "experiment"):
        section = retrieval.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(
                "Retrieval replay selector-provenance section is malformed: "
                f"{section_name}={section!r}"
            )
        metadata = section.get("neighbor_selector")
        if metadata is not None:
            observed.append(
                (
                    f"{section_name}.neighbor_selector",
                    _validated_selector_metadata(metadata, location=section_name),
                )
            )

    if not observed:
        # Artifacts predating selector provenance used similarity ordering.
        return selector_metadata(SIMILARITY_SELECTOR)
    if len(observed) == 2 and observed[0][1] != observed[1][1]:
        raise ValueError(
            "Retrieval replay neighbor-selector provenance conflicts between "
            f"{observed[0][0]}={observed[0][1]!r} and "
            f"{observed[1][0]}={observed[1][1]!r}"
        )
    return observed[0][1]


def load_retrieval_replay(
    source_run_dir: str,
    query_smiles: str,
    *,
    expected_neighbor_selector: str,
    expected_reranker_provenance: dict[str, Any] | None = None,
    expected_assay_transfer_selection_policy: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a frozen retrieval payload after validating query and policy provenance."""

    if not source_run_dir:
        return None
    source_path = Path(source_run_dir) / "retrieval.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Retrieval replay artifact does not exist: {source_path}")
    retrieval = json.loads(source_path.read_text(encoding="utf-8"))
    if retrieval.get("status") != "ok":
        raise ValueError(f"Retrieval replay artifact is not successful: {source_path}")
    source_smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
    if source_smiles and source_smiles != query_smiles:
        raise ValueError(
            "Retrieval replay query mismatch: "
            f"expected {query_smiles!r}, found {source_smiles!r} in {source_path}"
        )
    expected_selector = selector_metadata(expected_neighbor_selector)
    observed_selector = _selector_from_retrieval_payload(retrieval)
    selector_mismatches = {
        key: {
            "expected": expected_selector.get(key),
            "observed": observed_selector.get(key),
        }
        for key in ("name", "version")
        if expected_selector.get(key) != observed_selector.get(key)
    }
    if selector_mismatches:
        raise ValueError(
            "Retrieval replay neighbor-selector provenance mismatch: "
            f"{selector_mismatches}"
        )
    if expected_reranker_provenance is not None:
        observed = (retrieval.get("experiment") or {}).get("retrieval_reranker") or {"name": "none"}
        expected = expected_reranker_provenance
        keys = (
            "name",
            "model",
            "model_revision",
            "scoring_contract_version",
            "template_hash",
            "catalog_version",
        )
        mismatches = {
            key: {"expected": expected.get(key), "observed": observed.get(key)}
            for key in keys
            if expected.get(key) != observed.get(key)
        }
        if mismatches:
            raise ValueError(f"Retrieval replay reranker provenance mismatch: {mismatches}")
    if expected_assay_transfer_selection_policy is not None:
        observed_policy = ((retrieval.get("experiment") or {}).get("assay_transfer_selection_policy") or {})
        observed_diversity = observed_policy.get("diversity") or {}
        expected_diversity = expected_assay_transfer_selection_policy.get("diversity") or {}
        keys = ("version", "mode", "score_slack")
        mismatches = {
            key: {"expected": expected_diversity.get(key), "observed": observed_diversity.get(key)}
            for key in keys
            if expected_diversity.get(key) != observed_diversity.get(key)
        }
        if expected_assay_transfer_selection_policy.get("min_score") != observed_policy.get("min_score"):
            mismatches["min_score"] = {
                "expected": expected_assay_transfer_selection_policy.get("min_score"),
                "observed": observed_policy.get("min_score"),
            }
        if mismatches:
            raise ValueError(
                "Retrieval replay assay-transfer selection-policy mismatch: "
                f"{mismatches}"
            )
    return retrieval
