"""Reusable molecule-properties feature materialization for offline audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from tools.chembl_tool.common.json_utils import write_jsonl_atomic


def load_or_fetch_profiles(
    records: Iterable[tuple[str, str]],
    *,
    path: Path,
    required_features: set[str],
    tool_service_url: str,
    batch_size: int,
    fallback: Callable[[str], dict[str, float | None]] | None = None,
) -> tuple[dict[str, dict[str, float | None]], dict[str, int]]:
    expected = list(records)
    existing = {row["key"]: row for row in _read_jsonl(path)} if path.exists() else {}
    missing = [(key, smiles) for key, smiles in expected if key not in existing]
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        responses = _invoke_batch(tool_service_url, [smiles for _, smiles in chunk])
        for (key, smiles), response in zip(chunk, responses, strict=True):
            if response.get("status") != "ok":
                if fallback is None:
                    raise RuntimeError(f"molecule_properties failed for {key}: {response}")
                features = fallback(smiles)
                status = "rdkit_fallback_pka_unavailable"
                error = (response.get("errors") or [{}])[0]
            else:
                features = {
                    str(item["feature_name"]): item.get("feature_value")
                    for item in (response.get("output") or {}).get("features") or []
                }
                status, error = "ok", None
            if not required_features.issubset(features):
                raise ValueError(f"molecule_properties missing features for {key}")
            existing[key] = {
                "key": key,
                "smiles": smiles,
                "status": status,
                "error": error,
                "features": {name: features.get(name) for name in sorted(required_features)},
            }
    rows = [existing[key] for key, _ in expected]
    write_jsonl_atomic(path, rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "legacy_unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {row["key"]: row["features"] for row in rows}, status_counts


def _invoke_batch(url: str, smiles: list[str]) -> list[dict[str, Any]]:
    response = requests.post(
        f"{url.rstrip('/')}/tools/batch",
        json={
            "requests": [
                {
                    "tool_name": "molecule_properties",
                    "version": "v1",
                    "input": {"query_smiles": value, "logd_ph": 7.4},
                    "options": {"timeout_s": 300, "return_debug": False},
                }
                for value in smiles
            ]
        },
        timeout=300,
    )
    response.raise_for_status()
    results = response.json().get("responses") or []
    if len(results) != len(smiles):
        raise ValueError(f"Batch response count mismatch: {len(results)} != {len(smiles)}")
    return results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
