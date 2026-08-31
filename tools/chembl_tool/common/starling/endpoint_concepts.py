"""Load exhaustive, source-specific endpoint concept maps."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from tools.chembl_tool.common.starling.normalization.cleaning import (
    clean_text,
    endpoint_inventory_hash,
)


_CONCEPT_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def load_endpoint_concept_maps(
    paths: Sequence[Path],
    *,
    version: str,
    expected_inventories: Mapping[str, Mapping[str, object]],
    canonical_resolver: Callable[[str, str], str] | None = None,
) -> dict[str, dict[tuple[str, str], str]]:
    """Load complete reviewed maps keyed by raw and canonical endpoint."""
    maps: dict[str, dict[tuple[str, str], str]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_id = str(payload.get("source_id") or "")
        expected = expected_inventories.get(source_id)
        input_contract = payload.get("input_field") == "endpoint_name" or payload.get(
            "input_fields"
        ) == ["endpoint_name", "canonical_endpoint_name"]
        if (
            payload.get("version") != version
            or not input_contract
            or payload.get("output_field") != "canonical_endpoint_concept"
            or expected is None
            or payload.get("inventory") != expected
        ):
            raise ValueError(f"invalid endpoint-concept header: {path}")

        mapping: dict[tuple[str, str], str] = {}
        raw_endpoints: set[str] = set()
        for row in payload.get("mappings", []):
            endpoint = clean_text(row.get("endpoint_name")) or ""
            canonical = str(row.get("canonical_endpoint_name") or "")
            key = (endpoint, canonical)
            if not canonical or key in mapping:
                raise ValueError(f"invalid endpoint-concept key: {source_id}/{key}")
            if canonical_resolver is not None and canonical_resolver(
                source_id, endpoint
            ) != canonical:
                raise ValueError(
                    f"stale canonical endpoint in endpoint-concept map: "
                    f"{source_id}/{endpoint}"
                )
            concept = str(row.get("canonical_endpoint_concept") or "")
            if _CONCEPT_RE.fullmatch(concept) is None:
                raise ValueError(f"invalid endpoint concept: {source_id}/{key}")
            if not str(row.get("review_reason") or "").strip():
                raise ValueError(f"endpoint concept lacks review reason: {source_id}/{key}")
            mapping[key] = concept
            raw_endpoints.add(endpoint)

        actual = {
            "count": len(raw_endpoints),
            "sha256": endpoint_inventory_hash(raw_endpoints),
        }
        if actual != expected:
            raise ValueError(
                f"endpoint-concept inventory drift for {source_id}: "
                f"expected {expected}, found {actual}"
            )
        pair_tokens = [
            json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            for key in mapping
        ]
        observed_pairs = payload.get("observed_pairs")
        if observed_pairs is not None and observed_pairs != {
            "count": len(mapping),
            "sha256": endpoint_inventory_hash(pair_tokens),
        }:
            raise ValueError(f"endpoint-concept pair inventory drift: {source_id}")
        maps[source_id] = mapping

    if set(maps) != set(expected_inventories):
        raise ValueError("endpoint-concept source inventory is incomplete")
    return maps


__all__ = ["load_endpoint_concept_maps"]
