"""Compare frozen PubChem name resolutions with source-supplied structures.

This module never repairs a source structure and never treats unresolved names
as matches. Network acquisition is separate from deterministic source builds.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity

RESPONSES = Path(
    "data/starling_data/ames/identity_review_v1/pubchem_name_responses.jsonl"
)


def load_name_resolutions(path: Path = RESPONSES) -> dict[str, dict]:
    result = {}
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        name = raw["name"]
        if name in result:
            raise ValueError(f"Duplicate PubChem name response: {name}")
        attempts = raw.get("request_attempts") or [raw]
        properties = [
            item
            for attempt in attempts
            if attempt.get("ok")
            for item in (
                (attempt.get("response") or {}).get("PropertyTable") or {}
            ).get("Properties", [])
        ]
        parents = set()
        for item in properties:
            identity = normalize_molecule_identity(str(item.get("SMILES") or ""))
            if identity.status == "ok" and identity.parent_inchi_key:
                parents.add(identity.parent_inchi_key)
        result[name] = {
            "parent_keys": sorted(parents),
            "cids": sorted({r["CID"] for r in properties}),
            "n_request_attempts": len(attempts),
            "request_ok": raw["ok"],
            "url": raw["url"],
            "failure_kind": (
                ""
                if raw["ok"]
                else "name_not_found"
                if "404 Client Error" in (raw.get("error") or {}).get("message", "")
                else "identity_request_unresolved"
            ),
        }
    return result


def identity_status(name: str, parent: str, resolutions: dict[str, dict]) -> str:
    if name not in resolutions:
        return "not_queried"
    keys = resolutions[name]["parent_keys"]
    if not keys:
        return resolutions[name].get("failure_kind") or "unresolved_name_or_request"
    if len(keys) != 1:
        return "ambiguous_name_resolution"
    if parent == keys[0]:
        return "verified_parent_match"
    if parent.split("-")[0] == keys[0].split("-")[0]:
        return "stereochemistry_or_isotope_unresolved"
    return "name_structure_mismatch"
