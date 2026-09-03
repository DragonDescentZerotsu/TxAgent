"""Freeze a per-endpoint unit profile from rows resolved without a model.

Measurement extraction reads one row at a time, which leaves it arguing the
``x10^n`` convention from physical plausibility: handed ``21.4`` against a unit of
``x 10^6 cm/s`` it must reason about whether ``2.14e+07 cm/s`` could be a real
permeability.  A unit is an endpoint property far more than a row property -- of the
endpoints with at least :data:`MIN_PROFILE_ROWS` deterministically-resolved rows,
about half use one unit only and five in six have a dominant unit -- so the same
question can be settled empirically instead: ``apparent_permeability`` reports
``cm/s`` with a median of ``1.4e-05``, twelve orders below the literal reading.

**Only rows resolved without a model contribute.**  A current Stage-01 build
persists its route, so only ``routed_accept`` contributes:

``routed_accept``
    The router settled the row deterministically -- a bare positive number against an
    obvious unit.  Clean by construction.
``resolved_without_notation``
    The retiring parser produced a finite scalar and the unit carried no
    ``unit_notation_factor``. This remains a read-only compatibility path for
    older artifacts that predate persisted routing; it is not used to build a
    current Stage-01 profile.

The notation-factor exclusion is the point of the second predicate, not an
afterthought.  The parser's one known systematic error is exactly the ``x10^n``
convention this profile exists to inform, so admitting those rows would let a wrong
convention become the anchor that teaches it.  Excluding them costs almost nothing
in coverage.

Two limits are reported rather than smoothed over, because both bound how much the
profile can be trusted:

1. **The profile is inherited from the parser being retired.**  Most contributing
   rows are themselves extraction candidates that the parser happened to resolve.
   The notation filter removes the one known systematic error; an unknown one would
   be carried forward.  Every entry therefore states its own ``n`` so a 16-row
   anchor is not read as an 11,418-row one, and a downstream audit compares
   extraction output against the profile per endpoint.
2. **Profile strength is not proportional to extraction volume.**  The largest
   endpoints by candidate count are among the thinnest here, so coverage is reported
   broken out by strength rather than as a single percentage.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v7.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    ROUTE_BUCKETS,
    RouteDecision,
    SourceRoutingRules,
    route,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v1.normalization.measurements import parse_point_measurement
from tools.chembl_tool.common.units import canonicalize_unit
from data.processing.evidence_library.versions.v7.task_registry import import_task_module


PROFILE_VERSION = "starling_endpoint_unit_profile.v2"
SUPPORTED_TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction", "ames")
MAX_EXAMPLE_CANDIDATES_PER_UNIT = 4
MAX_SUPPORT_TEXT_CHARS = 300

#: An endpoint needs at least this many resolved rows before its profile is shown to
#: a model.  Below it the "dominant unit" is an artefact of the sample, not a fact
#: about the endpoint.
MIN_PROFILE_ROWS = 5

#: A unit needs this many rows within an endpoint to be listed.  Prevents a single
#: mis-parsed row from appearing beside a 10,000-row unit as though comparable.
MIN_UNIT_ROWS = 2

#: Encoder output, not a measured unit: a +/-1 categorical outcome carries no
#: magnitude information and would corrupt every median it entered.
ENCODER_UNIT = "binary_outcome_class"

#: Reported strength bands.  Named rather than inlined so the audit, the tests and
#: the manifest cannot drift apart.
STRENGTH_BANDS: tuple[tuple[str, int, int], ...] = (
    ("none", 0, 0),
    ("too_thin", 1, MIN_PROFILE_ROWS - 1),
    ("thin", MIN_PROFILE_ROWS, 24),
    ("moderate", 25, 199),
    ("strong", 200, 2**62),
)

CONTRIBUTION_KINDS = ("routed_accept", "resolved_without_notation")


def strength_band(n: int) -> str:
    for name, low, high in STRENGTH_BANDS:
        if low <= n <= high:
            return name
    raise ValueError(f"no strength band covers n={n}")


def _endpoint_key(
    record: dict[str, Any],
    endpoint_resolver: Callable[[str, Any], str] | None,
    endpoint_record_resolver: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    """Return the pre-encoder endpoint that owns the measurement."""
    if endpoint_record_resolver is not None:
        return endpoint_record_resolver(record)
    if endpoint_resolver is not None:
        return endpoint_resolver(
            str(record.get("source_id") or ""), record.get("endpoint_name")
        )
    published = record.get("canonical_endpoint_name")
    if not published:
        raise ValueError(
            "records lack canonical_endpoint_name; build the profile from the "
            "Stage 01 cleaned records after measurement routing"
        )
    return str(published)


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def contribution(
    record: dict[str, Any],
    rules: SourceRoutingRules,
    *,
    task: str | None,
) -> tuple[str, str, float] | None:
    """Classify one row as a profile contributor, or ``None``.

    Returns ``(kind, canonical_unit, value)``.  ``resolved_without_notation`` is
    preferred over ``routed_accept`` when a row qualifies as both, because the
    published canonical unit is the one downstream consumers see.
    """
    persisted_route = record.get("measurement_resolution_route")
    if persisted_route is not None and str(persisted_route) not in {
        *ROUTE_BUCKETS,
        "categorical",
    }:
        raise ValueError(f"unsupported persisted measurement route {persisted_route!r}")
    if persisted_route is None:
        notation = _float(record.get("unit_notation_factor"))
        scalar = _float(record.get("finite_scalar_value"))
        unit = record.get("canonical_unit_text")
        if scalar is not None and notation is None and unit and unit != ENCODER_UNIT:
            return ("resolved_without_notation", str(unit), scalar)
    elif persisted_route != "accept":
        return None

    if persisted_route == "accept" and record.get(
        "measurement_resolution_exact_measurement"
    ) is not None:
        decision = RouteDecision(
            "accept",
            str(record.get("measurement_resolution_rule_id") or "source_exact"),
            str(record["measurement_resolution_exact_measurement"]),
            str(record.get("measurement_resolution_exact_unit") or ""),
            bool(record.get("measurement_resolution_exact_unit_is_canonical")),
        )
    else:
        decision = route(record, rules, task=task)
    if persisted_route == "accept" and decision.bucket != "accept":
        raise ValueError("persisted accept disagrees with legacy acceptance rule")
    if decision.bucket != "accept":
        return None
    value = parse_point_measurement(decision.measurement_text).value
    if value is None:
        return None
    # The router already proved this unit resolves with no unknown tokens and no
    # scale factor, so canonicalization here is a spelling step, not a judgement.
    resolved = canonicalize_unit(decision.unit_text, task=task)
    canonical = resolved.canonical or str(decision.unit_text or "").strip()
    if not canonical or canonical == ENCODER_UNIT:
        return None
    return ("routed_accept", canonical, value)


def _summarize(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    n = len(ordered)

    def q(fraction: float) -> float:
        if n == 1:
            return ordered[0]
        index = min(n - 1, max(0, int(round(fraction * (n - 1)))))
        return ordered[index]

    return {
        "n": n,
        "median": statistics.median(ordered),
        "p10": q(0.10),
        "p90": q(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def build_profile(
    records_path: Path,
    rules_by_source: dict[str, SourceRoutingRules],
    *,
    task: str | None = None,
    endpoint_resolver: Callable[[str, Any], str] | None = None,
    endpoint_record_resolver: Callable[[dict[str, Any]], str] | None = None,
    min_profile_rows: int = MIN_PROFILE_ROWS,
    min_unit_rows: int = MIN_UNIT_ROWS,
) -> dict[str, Any]:
    available = set(pq.read_schema(records_path).names)
    wanted = {
        "source_id",
        "cleaned_record_id",
        "endpoint_name",
        "canonical_endpoint_name",
        "canonical_unit_text",
        "finite_scalar_value",
        "unit_notation_factor",
        "measurement_text",
        "unit_text",
        "support_text",
        "measurement_resolution_route",
        "measurement_resolution_rule_id",
        "measurement_resolution_exact_measurement",
        "measurement_resolution_exact_unit",
        "measurement_resolution_exact_unit_is_canonical",
    }
    for rules in rules_by_source.values():
        wanted.add(rules.measurement_field)
        if rules.unit_field:
            wanted.add(rules.unit_field)
    columns = sorted(wanted & available)
    missing = sorted(wanted - available)

    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    examples: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    kinds: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: dict.fromkeys(CONTRIBUTION_KINDS, 0)
    )
    candidates: dict[tuple[str, str], int] = defaultdict(int)
    rows_seen = 0
    unknown_sources: dict[str, int] = defaultdict(int)

    parquet = pq.ParquetFile(records_path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        for record in batch.to_pylist():
            rows_seen += 1
            source_id = str(record.get("source_id") or "")
            rules = rules_by_source.get(source_id)
            if rules is None:
                unknown_sources[source_id] += 1
                continue
            key = (
                source_id,
                _endpoint_key(record, endpoint_resolver, endpoint_record_resolver),
            )
            persisted_route = record.get("measurement_resolution_route")
            route_bucket = (
                str(persisted_route)
                if persisted_route is not None
                else route(record, rules, task=task).bucket
            )
            if route_bucket == "extract":
                candidates[key] += 1
            found = contribution(record, rules, task=task)
            if found is None:
                continue
            kind, unit, value = found
            values[key][unit].append(value)
            examples[key][unit].append(
                {
                    "cleaned_record_id": str(record.get("cleaned_record_id") or ""),
                    "measurement_text": record.get("measurement_text"),
                    "unit_text": record.get("unit_text"),
                    "folded_value": value,
                    "canonical_unit": unit,
                    "support_text": str(record.get("support_text") or "").strip()[
                        :MAX_SUPPORT_TEXT_CHARS
                    ],
                }
            )
            kinds[key][kind] += 1

    endpoints: dict[str, Any] = {}
    resolved_rows = {
        key: sum(map(len, per_unit.values())) for key, per_unit in values.items()
    }
    for key in sorted(values):
        per_unit = values[key]
        total = resolved_rows[key]
        if total < min_profile_rows:
            continue
        listed = {
            unit: _summarize(items)
            for unit, items in per_unit.items()
            if len(items) >= min_unit_rows
        }
        if not listed:
            continue
        ordered = sorted(listed.items(), key=lambda item: (-item[1]["n"], item[0]))
        source_id, endpoint = key
        units = []
        for unit, stats in ordered[:4]:
            ranked_examples = sorted(
                examples[key][unit],
                key=lambda item: (
                    abs(float(item["folded_value"]) - float(stats["median"])),
                    item["cleaned_record_id"],
                ),
            )
            summary = {
                "unit": unit,
                **stats,
                "example_candidates": ranked_examples[
                    :MAX_EXAMPLE_CANDIDATES_PER_UNIT
                ],
            }
            units.append(summary)
        endpoints[f"{source_id}|{endpoint}"] = {
            "source_id": source_id,
            "canonical_endpoint_name": endpoint,
            "resolved_rows": total,
            "listed_rows": sum(item["n"] for _, item in ordered),
            "listed_unit_count": len(ordered),
            "distinct_units": len(per_unit),
            "dominant_unit": ordered[0][0],
            "dominant_share": round(ordered[0][1]["n"] / total, 6),
            "contribution_kinds": dict(kinds[key]),
            "extraction_candidates": candidates.get(key, 0),
            "units": units,
        }

    coverage = dict.fromkeys((band for band, _, _ in STRENGTH_BANDS), 0)
    for key, count in candidates.items():
        endpoint_key = f"{key[0]}|{key[1]}"
        if endpoint_key in endpoints:
            band = strength_band(resolved_rows[key])
        elif 0 < resolved_rows.get(key, 0) < min_profile_rows:
            band = "too_thin"
        else:
            band = "none"
        coverage[band] += count
    total_candidates = sum(candidates.values())

    return {
        "profile_version": PROFILE_VERSION,
        "routing_version": MEASUREMENT_ROUTING_VERSION,
        "task": task,
        "records_path": str(records_path),
        "records_sha256": file_sha256(records_path),
        "min_profile_rows": min_profile_rows,
        "min_unit_rows": min_unit_rows,
        "encoder_unit_excluded": ENCODER_UNIT,
        "notation_factor_rows_excluded": True,
        "columns_requested_but_absent": missing,
        "rows_from_undeclared_sources": dict(unknown_sources),
        "totals": {
            "rows": rows_seen,
            "endpoints_profiled": len(endpoints),
            "extraction_candidates": total_candidates,
            "candidates_by_profile_strength": coverage,
            "candidate_share_with_a_profile": (
                round(
                    sum(
                        count
                        for band, count in coverage.items()
                        if band not in {"none", "too_thin"}
                    )
                    / total_candidates,
                    6,
                )
                if total_candidates
                else 0.0
            ),
            "single_unit_endpoint_share": (
                round(
                    sum(
                        1
                        for item in endpoints.values()
                        if item["distinct_units"] == 1
                    )
                    / len(endpoints),
                    6,
                )
                if endpoints
                else 0.0
            ),
            "dominant_unit_at_least_80pct_share": (
                round(
                    sum(
                        1
                        for item in endpoints.values()
                        if item["dominant_share"] >= 0.8
                    )
                    / len(endpoints),
                    6,
                )
                if endpoints
                else 0.0
            ),
        },
        "endpoints": endpoints,
    }


def profile_lookup(profile: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index a built profile by ``(source_id, canonical_endpoint_name)``."""
    return {
        (item["source_id"], item["canonical_endpoint_name"]): item
        for item in profile["endpoints"].values()
    }


def render_profile_block(
    entry: dict[str, Any],
    *,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
    max_units: int = 4,
    max_examples: int = 3,
) -> str:
    """Format one endpoint's profile as the read-only block shown to the model.

    ``n`` is stated per unit and for the endpoint as a whole, because a 16-row
    anchor and an 11,418-row anchor deserve very different weight and the model
    cannot tell them apart otherwise.
    """
    lines = [
        f"[endpoint: {entry['canonical_endpoint_name']} "
        f"- {entry['resolved_rows']} rows resolved deterministically]"
    ]
    for unit in entry["units"][:max_units]:
        lines.append(
            f"  {unit['unit']}  n={unit['n']}  median {unit['median']:.3g}"
            f"  p10 {unit['p10']:.3g}  p90 {unit['p90']:.3g}"
        )
    listed_unit_count = int(entry.get("listed_unit_count") or len(entry["units"]))
    if listed_unit_count > max_units:
        lines.append(f"  ... and {listed_unit_count - max_units} rarer units")
    selected: list[dict[str, Any]] = []
    for unit in entry["units"][:max_units]:
        example = next(
            (
                item
                for item in unit.get("example_candidates", ())
                if item["cleaned_record_id"] not in exclude_ids
            ),
            None,
        )
        if example is not None:
            selected.append(example)
        if len(selected) == max_examples:
            break
    if selected:
        lines.append("  representative parsed rows:")
        for item in selected:
            lines.append(
                "    - raw measurement "
                f"{item['measurement_text']!r}; raw unit {item['unit_text']!r}; "
                f"folded {item['folded_value']:.3g} {item['canonical_unit']}; "
                f"support {item['support_text']!r}"
            )
    return "\n".join(lines)


def _load_task(
    task_id: str,
) -> tuple[
    dict[str, SourceRoutingRules],
    Path,
    Callable[[str, Any], str],
    Callable[[dict[str, Any]], str] | None,
]:
    cfg = import_task_module(task_id, "starling_measurement_resolution")
    return (
        cfg.source_routing_rules(),
        cfg.DEFAULT_CANONICAL_RECORDS,
        cfg.canonical_endpoint_name,
        getattr(cfg, "canonical_endpoint_record", None),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, choices=SUPPORTED_TASKS
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Stage 01 cleaned records parquet carrying the persisted routing and "
        "canonical endpoint contract",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", action="store_true", help="print a coverage summary")
    parser.add_argument(
        "--show", type=int, default=0, help="print the N largest endpoint profiles"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rules, default_records, endpoint_resolver, endpoint_record_resolver = _load_task(
        args.task
    )
    records_path = args.records or default_records
    if not records_path.is_file():
        raise SystemExit(f"records not found: {records_path}")
    profile = build_profile(
        records_path,
        rules,
        task=args.task,
        endpoint_resolver=endpoint_resolver,
        endpoint_record_resolver=endpoint_record_resolver,
    )
    totals = profile["totals"]
    if profile["rows_from_undeclared_sources"]:
        raise SystemExit(
            f"undeclared sources present: {profile['rows_from_undeclared_sources']}"
        )

    if args.report or not args.output:
        print(f"rows read                : {totals['rows']:,}")
        print(f"endpoints profiled       : {totals['endpoints_profiled']:,}")
        print(f"  single unit only       : {totals['single_unit_endpoint_share']:.1%}")
        print(f"  dominant unit >=80%    : {totals['dominant_unit_at_least_80pct_share']:.1%}")
        print(f"extraction candidates    : {totals['extraction_candidates']:,}")
        for band, _, _ in reversed(STRENGTH_BANDS):
            count = totals["candidates_by_profile_strength"][band]
            share = (
                count / totals["extraction_candidates"]
                if totals["extraction_candidates"]
                else 0
            )
            print(f"  {band:10} {count:>9,}  ({share:.1%})")
        print(f"candidates with a profile: {totals['candidate_share_with_a_profile']:.1%}")

    if args.show:
        ranked = sorted(
            profile["endpoints"].values(),
            key=lambda item: -item["extraction_candidates"],
        )
        print()
        for entry in ranked[: args.show]:
            print(
                f"# {entry['source_id']}  candidates={entry['extraction_candidates']:,}"
            )
            print(render_profile_block(entry))
            print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
