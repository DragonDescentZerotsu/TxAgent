"""Direct-card and joint-record search policies, using explicit scientific grids."""

from dataclasses import fields
import json
import math
from pathlib import Path
from .objectives import DirectProfile, IndirectProfile

DIRECT_BUDGET = 10
INDIRECT_BUDGET = 50
GRIDS = Path(__file__).with_name("grids")


def load_grid(name: str | Path, *, kind: str) -> tuple[list, str]:
    path = Path(name)
    if not path.is_file():
        path = GRIDS / f"{name}.json"
    document = json.loads(path.read_text())
    if document.get("kind") != kind or document.get("diversity") not in {"log", "sqrt"}:
        raise ValueError("grid kind or diversity differs from the search policy")
    cls = DirectProfile if kind == "direct" else IndirectProfile
    allowed = {field.name for field in fields(cls)}
    profiles = []
    seen = set()
    for row in document["profiles"]:
        if set(row) - allowed or not row.get("name") or row["name"] in seen:
            raise ValueError("grid repeats names or contains unknown profile fields")
        if any(
            (
                not math.isfinite(value) or value < 0
                for key, value in row.items()
                if key not in {"name", "aliases", "dili_geomean"}
            )
        ):
            raise ValueError(
                "grid objective coefficients must be finite and nonnegative"
            )
        row = dict(row)
        if "aliases" in row:
            row["aliases"] = tuple(row["aliases"])
        profiles.append(cls(**row))
        seen.add(row["name"])
    if not profiles:
        raise ValueError("grid contains no configurations")
    return (profiles, document["diversity"])


def validate_indirect(
    profile: IndirectProfile,
    *,
    task: str,
    diversity: str,
    dili_scores: Path | None,
    direct_manifest: Path | None,
) -> None:
    if profile.dili_geomean:
        if (
            task != "dili"
            or dili_scores is None
            or direct_manifest is None
            or (diversity != "sqrt")
        ):
            raise ValueError(
                "DILI geometric relevance requires DILI scores, a frozen direct panel and square-root diversity"
            )
        if (
            len(
                {
                    profile.molecular,
                    profile.semantic_diversity,
                    profile.level_diversity,
                    profile.record_diversity,
                }
            )
            != 1
        ):
            raise ValueError("DILI geometric diversity coefficients must be tied")
    elif dili_scores is not None:
        raise ValueError("DILI score inputs require an explicit geometric objective")
