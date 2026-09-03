"""Source-independent interfaces for optional molecule-level retrieval reranking."""

from __future__ import annotations

from typing import Any, Protocol


class RetrievalReranker(Protocol):
    """Rerank an identity-filtered structural candidate pool for one evidence family."""

    name: str

    def rerank(
        self,
        *,
        query_smiles: str,
        group_id: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return every candidate with deterministic audit-only reranking metadata."""

    def rerank_records(
        self,
        *,
        query_smiles: str,
        group_id: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return one item per scored record (record-level top-K), sorted by score."""

    def provenance(self) -> dict[str, Any]:
        """Return immutable configuration needed to validate retrieval replay."""
