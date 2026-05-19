"""Exact ChEMBL molecule context and shared-assay enrichment for Skin_Reaction reasoning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.chembl_tool.common.task_workflows.chembl_exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    QUERY_ACTIVITY_FIELDS,
    attach_shared_assay_context,
    exact_evidence_rows,
    load_query_activities_for_assays,
    lookup_exact_molecule,
)
from tools.chembl_tool.common.task_workflows.chembl_exact_context import (
    enrich_retrieval_with_chembl_context as enrich_common_retrieval_with_chembl_context,
)


def enrich_retrieval_with_chembl_context(
    retrieval: dict[str, Any],
    index: dict[str, Any],
    *,
    chembl_sqlite: str | Path = DEFAULT_CHEMBL_SQLITE,
    max_exact_skin_reaction_rows: int = 30,
    max_shared_per_neighbor: int = 8,
) -> dict[str, Any]:
    return enrich_common_retrieval_with_chembl_context(
        retrieval,
        index,
        relevant_evidence_key="skin_reaction_relevant_evidence_rows",
        context_label="Skin_Reaction",
        chembl_sqlite=chembl_sqlite,
        max_exact_rows=max_exact_skin_reaction_rows,
        max_shared_per_neighbor=max_shared_per_neighbor,
    )


def exact_skin_reaction_evidence_rows(
    index: dict[str, Any],
    molecule_chembl_id: str,
    *,
    max_rows: int = 30,
) -> list[dict[str, Any]]:
    return exact_evidence_rows(index, molecule_chembl_id, max_rows=max_rows)
