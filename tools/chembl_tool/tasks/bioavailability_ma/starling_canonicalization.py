"""Thin Bioavailability façade over the shared v7 record contract.

The established endpoint, measurement/unit, Fg, and auxiliary modules remain
the scientific implementations.  This module owns only the persisted v7
boundary so those rules are not duplicated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .starling_schema import RECORD_CONTRACT


def clean_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return RECORD_CONTRACT.clean_projection(record)


def canonicalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return RECORD_CONTRACT.canonical_projection(record)


__all__ = ["canonicalize_record", "clean_record"]
