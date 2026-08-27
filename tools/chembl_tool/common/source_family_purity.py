"""Shared types for retrieval-only source-family reassignment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyMove:
    """One auditable source-row reassignment."""

    new_group: str
    reason: str
