"""Correctness audit for the unit normalizer over all on-disk Starling unit columns.

Not a pytest module -- it streams the large ``extractions.parquet`` files (~640k rows)
and checks that the normalizer never *incorrectly* normalizes: it writes a reviewable
CSV of every distinct raw form and runs automated over-merge / mis-dimension detectors.

Run from the repo root::

    conda run --no-capture-output -n txagent-glm python tests/chembl_tool/common/units_audit.py
    conda run --no-capture-output -n txagent-glm python tests/chembl_tool/common/units_audit.py --write-golden

Detectors:
  * cross-dimension collision -- any canonical key mapping to >1 distinct dimension is a
    parser bug (MUST be zero).
  * beyond-typography merge -- distinct raw forms that collapse to one cleaned string but
    differ by more than pure typography/case; listed for human confirmation that the merge
    is meaning-preserving.
  * under-normalization -- residual unknown tokens ranked by frequency, split unit-like vs
    free-text, so genuinely-missed units stand out from free-text noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.chembl_tool.common.units import canonicalize_unit, clean_unit  # noqa: E402

DATA_ROOT = ROOT / "data" / "starling_data"
GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "units_golden.json"
AUDIT_CSV = Path("/tmp/txagent_units_audit.csv")

# (label, parquet subpath, unit column) for every on-disk Starling unit column.
SOURCES = [
    ("oral_exposure", "bioavailability_ma/Oral_AUC-Cmax_Exposure", "parameter_units"),
    ("fa", "bioavailability_ma/Fa", "reported_units"),
    ("fh", "bioavailability_ma/Fh", "reported_units"),
    ("bbb_passive_permeability", "bbb_martins/passive_permeability", "metric_units"),
    ("skin_exposure", "skin_reaction/skin_exposure", "result_unit"),
    ("skin_sensitization_aop", "skin_reaction/sensitization_aop", "result_unit"),
]

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-", "‒": "-", "−": "-"})


def _reference_key(raw: str) -> str:
    """A minimal typography/case-insensitive key.

    Two raw forms sharing this key differ only by trivia (case, whitespace, unicode
    mu/dash/superscript variants, ×/x, ^/{} exponent notation). A merge whose raw forms
    do NOT all share one reference key means ``clean_unit`` applied a semantic alias -- the
    place a real over-merge would hide, so those groups are surfaced for review.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.translate(_SUPERSCRIPTS).translate(_DASHES).casefold()
    text = text.replace("μ", "u").replace("µ", "u").replace("×", "x")
    text = re.sub(r"[\s^{}]+", "", text)
    return text


# English words that appear in qualified unit strings but are not units themselves.
_FREE_TEXT_WORDS = {
    "of", "dose", "skin", "tissue", "liver", "log", "total", "weight", "score", "week",
    "out", "per", "applied", "administered", "control", "initial", "absorbed", "patients",
    "subjects", "cases", "reactions", "positive", "million", "radioactivity", "microsomal",
}


def _is_unit_like(token: str) -> bool:
    """Heuristic: an unresolved token that plausibly IS a unit (vs free-text)."""
    stripped = token.strip("()[]")
    if stripped.casefold() in _FREE_TEXT_WORDS:
        return False
    return bool(re.fullmatch(r"[A-Za-zµ%]{1,6}(?:\^?-?\d+)?", stripped))


def _collect() -> tuple[Counter, dict, dict]:
    """Return (raw form -> total count), (raw -> set of source labels), (label -> row count)."""
    raw_counts: Counter[str] = Counter()
    raw_sources: dict[str, set[str]] = defaultdict(set)
    source_rows: dict[str, int] = {}
    for label, subpath, column in SOURCES:
        path = DATA_ROOT / subpath / "extractions.parquet"
        if not path.exists():
            print(f"  [skip] {label}: {path} not found")
            continue
        frame = pd.read_parquet(path, columns=[column])
        source_rows[label] = len(frame)
        for value in frame[column].tolist():
            raw = str(value)
            raw_counts[raw] += 1
            raw_sources[raw].add(label)
    return raw_counts, raw_sources, source_rows


def _audit() -> dict:
    raw_counts, raw_sources, source_rows = _collect()

    rows = []  # per distinct raw form
    canonical_dims: dict[str, set] = defaultdict(set)   # canonical -> {dimension}
    cleaned_to_raws: dict[str, set] = defaultdict(set)  # cleaned -> {raw}
    unknown_counts: Counter[str] = Counter()

    for raw, count in raw_counts.items():
        result = canonicalize_unit(raw)
        cleaned = result.cleaned
        rows.append(
            {
                "raw": raw,
                "count": count,
                "sources": ",".join(sorted(raw_sources[raw])),
                "cleaned": cleaned,
                "canonical": result.canonical,
                "dimension": json.dumps(result.dimension),
                "scale": result.scale,
                "transform": result.transform,
                "unknown_tokens": ",".join(result.unknown_tokens),
            }
        )
        if cleaned:
            canonical_dims[result.canonical].add(result.dimension)
            cleaned_to_raws[cleaned].add(raw)
            for token in result.unknown_tokens:
                unknown_counts[token] += count

    rows.sort(key=lambda r: r["count"], reverse=True)

    # Detector 1: cross-dimension collisions (must be empty).
    collisions = {c: dims for c, dims in canonical_dims.items() if len(dims) > 1}

    # Detector 2: beyond-typography merges (distinct reference keys inside one cleaned group).
    beyond_typography = []
    for cleaned, raws in cleaned_to_raws.items():
        if len(raws) < 2:
            continue
        ref_keys = {_reference_key(r) for r in raws}
        if len(ref_keys) > 1:
            beyond_typography.append(
                {
                    "cleaned": cleaned,
                    "count": sum(raw_counts[r] for r in raws),
                    "raws": sorted(raws, key=lambda r: raw_counts[r], reverse=True),
                }
            )
    beyond_typography.sort(key=lambda g: g["count"], reverse=True)

    return {
        "rows": rows,
        "source_rows": source_rows,
        "raw_counts": raw_counts,
        "raw_sources": raw_sources,
        "collisions": collisions,
        "beyond_typography": beyond_typography,
        "unknown_counts": unknown_counts,
    }


def _print_report(audit: dict) -> None:
    rows = audit["rows"]
    print("Unit normalization correctness audit")
    print(f"data root: {DATA_ROOT}")
    print(f"distinct raw forms across all sources: {len(rows):,}")

    print("\nPer-source distinct-form reduction:")
    for label, _, _ in SOURCES:
        if label not in audit["source_rows"]:
            continue
        source_raws = [r for r in rows if label in r["sources"].split(",")]
        raw_distinct = len(source_raws)
        cleaned_distinct = len({r["cleaned"] for r in source_raws if r["cleaned"]})
        print(f"  {label:26} rows={audit['source_rows'][label]:>7,}  "
              f"raw={raw_distinct:>5}  cleaned={cleaned_distinct:>5}")

    collisions = audit["collisions"]
    print(f"\n[detector 1] cross-dimension collisions: {len(collisions)} (must be 0)")
    for canonical, dims in list(collisions.items())[:20]:
        print(f"    COLLISION canonical={canonical!r}: {dims}")

    beyond = audit["beyond_typography"]
    print(f"\n[detector 2] beyond-typography merge groups to review: {len(beyond)}")
    print("    (each is clean_unit applying a semantic alias; confirm meaning-preserving)")
    for group in beyond[:40]:
        sample = ", ".join(repr(r) for r in group["raws"][:6])
        print(f"    cleaned={group['cleaned']!r:22} (n={group['count']:>6})  <- {sample}")

    unknown = audit["unknown_counts"]
    unit_like = [(t, n) for t, n in unknown.most_common() if _is_unit_like(t)]
    free_text = [(t, n) for t, n in unknown.most_common() if not _is_unit_like(t)]
    print(f"\n[detector 3] residual unknown tokens: {len(unknown)} distinct")
    print("    unit-like (potential missed units — investigate top ones):")
    for token, n in unit_like[:25]:
        print(f"      {n:>7}  {token!r}")
    print("    free-text (expected to stay flagged):")
    for token, n in free_text[:15]:
        print(f"      {n:>7}  {token!r}")

    print(f"\nFull per-form audit written to: {AUDIT_CSV}")


def _write_csv(audit: dict) -> None:
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["raw", "count", "sources", "cleaned", "canonical",
              "dimension", "scale", "transform", "unknown_tokens"]
    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit["rows"])


def _write_golden(audit: dict, per_source: int = 150) -> None:
    """Freeze the top-N most-frequent real raw forms per source as a regression baseline."""
    raw_counts = audit["raw_counts"]
    raw_sources = audit["raw_sources"]
    chosen: set[str] = set()
    for label, _, _ in SOURCES:
        source_forms = [r for r in raw_counts if label in raw_sources[r]]
        source_forms.sort(key=lambda r: raw_counts[r], reverse=True)
        chosen.update(source_forms[:per_source])

    golden = {}
    for raw in sorted(chosen):
        if clean_unit(raw) is None:
            continue  # skip null-like forms (nan, "", ...)
        result = canonicalize_unit(raw)
        golden[raw] = {
            "cleaned": result.cleaned,
            "canonical": result.canonical,
            "dimension": result.dimension,
            "scale": result.scale,
            "transform": result.transform,
            "unknown_tokens": list(result.unknown_tokens),
        }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(golden)} golden entries to {GOLDEN_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-golden", action="store_true",
                        help="regenerate the golden regression fixture from top real forms")
    args = parser.parse_args()

    audit = _audit()
    _write_csv(audit)
    _print_report(audit)
    if args.write_golden:
        _write_golden(audit)


if __name__ == "__main__":
    main()
