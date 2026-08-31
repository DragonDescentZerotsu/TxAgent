"""Prepare and compose exact-unit reviews for the three normalized-v7 tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")
DEFAULT_REVIEW_ROOT = (
    ROOT / "outputs/chembl_tool/audits/exact_measurement_unit_review_v3"
)


def clean_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def _paths(task: str) -> dict[str, Path]:
    task_root = ROOT / "tools/chembl_tool/tasks" / task
    return {
        "map": task_root
        / "data_processing/canonicalization_v7/exact_measurement_unit_map.v2.json",
        "resolution": task_root
        / "data_processing/measurement_resolution_v3/measurement_resolution.parquet",
        "cleaned": ROOT
        / "outputs/chembl_tool/tasks"
        / task
        / "evidence_library/starling_normalized_v7/01_cleaned/records.parquet",
        "canonical": ROOT
        / "outputs/chembl_tool/tasks"
        / task
        / "evidence_library/starling_normalized_v7/02_canonicalized/records.parquet",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rule_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["action"],
        entry.get("canonical_unit"),
        str(entry.get("scale")) if entry.get("scale") is not None else None,
        entry.get("domain", "any"),
    )


def _expanded_rules(
    payloads: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rules: dict[tuple[str, str, str], dict[str, Any]] = {}
    for task, payload in payloads.items():
        for entry in payload["entries"]:
            for endpoint in entry["canonical_endpoints"]:
                key = (task, endpoint, entry["input_unit"])
                if key in rules:
                    raise ValueError(f"duplicate exact-unit key: {key!r}")
                rules[key] = {
                    "action": entry["action"],
                    "canonical_unit": entry.get("canonical_unit"),
                    "scale": entry.get("scale"),
                    "domain": entry.get("domain", "any"),
                }
    return rules


def find_whitespace_alias(
    key: tuple[str, str, str],
    rules: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    task, endpoint, unit = key
    signatures = {
        _rule_signature(rule): rule
        for (rule_task, rule_endpoint, rule_unit), rule in rules.items()
        if rule_task == task
        and rule_endpoint == endpoint
        and clean_whitespace(rule_unit) == clean_whitespace(unit)
        and rule["action"] == "map"
    }
    return next(iter(signatures.values())) if len(signatures) == 1 else None


def _active_missing(
    task: str,
    rules: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    paths = _paths(task)
    cleaned = pq.read_table(
        paths["cleaned"],
        columns=[
            "cleaned_record_id",
            "canonical_endpoint_name",
            "measurement_resolution_route",
            "measurement_resolution_exact_measurement",
            "measurement_resolution_exact_unit",
            "measurement_resolution_exact_unit_is_canonical",
            "measurement_text",
            "unit_text",
        ],
    ).to_pydict()
    cleaned_by_id = {
        str(values[0] or ""): dict(zip(cleaned, values, strict=True))
        for values in zip(*cleaned.values(), strict=True)
    }
    missing: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for record_id, row in cleaned_by_id.items():
        if row["measurement_resolution_route"] != "accept" or row[
            "measurement_resolution_exact_unit_is_canonical"
        ]:
            continue
        endpoint = str(row["canonical_endpoint_name"] or "")
        unit = str(
            row["measurement_resolution_exact_unit"] or row["unit_text"] or ""
        ).strip()
        key = (task, endpoint, unit)
        if key not in rules:
            missing[key].append(
                {
                    "cleaned_record_id": record_id,
                    "measurement": str(
                        row["measurement_resolution_exact_measurement"]
                        or row["measurement_text"]
                        or ""
                    ),
                    "unit": unit,
                }
            )
    table = pq.read_table(
        paths["resolution"],
        columns=["cleaned_record_id", "status", "measurements_json"],
    ).to_pydict()
    for record_id, status, measurements_json in zip(
        table["cleaned_record_id"],
        table["status"],
        table["measurements_json"],
        strict=True,
    ):
        if status != "ok":
            continue
        record_id = str(record_id or "")
        row = cleaned_by_id.get(record_id)
        if row is None or row["measurement_resolution_route"] != "extract":
            continue
        endpoint = str(row["canonical_endpoint_name"] or "")
        for quantity in json.loads(measurements_json or "[]"):
            unit = str(quantity.get("unit") or "").strip()
            key = (task, endpoint, unit)
            if key not in rules:
                missing[key].append(
                    {
                        "cleaned_record_id": record_id,
                        "measurement": str(quantity.get("measurement") or ""),
                        "unit": unit,
                    }
                )
    return missing


def _existing_excludes(
    payloads: dict[str, dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, str]]],
    dict[tuple[str, str, str], int],
]:
    examples: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for task, payload in payloads.items():
        for entry in payload["entries"]:
            if entry["action"] != "exclude":
                continue
            for endpoint in entry["canonical_endpoints"]:
                key = (task, endpoint, entry["input_unit"])
                examples[key] = []
                counts[key] = 0
        table = pq.read_table(
            _paths(task)["canonical"],
            columns=[
                "canonical_endpoint_name",
                "measurement_resolution_input_measurement",
                "measurement_resolution_input_unit",
                "measurement_resolution_parent_cleaned_record_id",
                "cleaned_record_id",
                "measurement_unit_mapping_action",
            ],
        ).to_pylist()
        for row in table:
            if row["measurement_unit_mapping_action"] != "exclude":
                continue
            key = (
                task,
                str(row["canonical_endpoint_name"] or ""),
                str(row["measurement_resolution_input_unit"] or ""),
            )
            if key not in examples:
                raise ValueError(f"active exclusion is absent from the map: {key!r}")
            counts[key] += 1
            if len(examples[key]) < 3:
                examples[key].append(
                    {
                        "cleaned_record_id": str(
                            row["measurement_resolution_parent_cleaned_record_id"]
                            or row["cleaned_record_id"]
                            or ""
                        ),
                        "measurement": str(
                            row["measurement_resolution_input_measurement"] or ""
                        ),
                        "unit": key[2],
                    }
                )
    return examples, counts


def _source_rows(task: str, record_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    parquet = pq.ParquetFile(_paths(task)["cleaned"])
    for batch in parquet.iter_batches(batch_size=16_384):
        for row in batch.to_pylist():
            record_id = str(row.get("cleaned_record_id") or "")
            if record_id in record_ids:
                rows[record_id] = row
    missing = record_ids - rows.keys()
    if missing:
        raise ValueError(f"{task}: cleaned rows missing for {sorted(missing)[:3]}")
    return rows


def fuzzy_matches(
    queries: list[str], candidates: list[str], top_k: int
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic character-ngram cosine candidates."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(1, 4), lowercase=True, norm="l2"
    )
    candidate_matrix = vectorizer.fit_transform(candidates)
    similarities = cosine_similarity(
        vectorizer.transform(queries), candidate_matrix, dense_output=True
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for query, scores in zip(queries, similarities, strict=True):
        ranked = sorted(
            zip(scores, candidates, strict=True), key=lambda item: (-item[0], item[1])
        )[:top_k]
        result[query] = [
            {"unit": candidate, "score": round(float(score), 8)}
            for score, candidate in ranked
        ]
    return result


def _candidate_rules(
    units: set[str], payloads: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = []
    for task, payload in payloads.items():
        for entry in payload["entries"]:
            if entry["input_unit"] in units or entry.get("canonical_unit") in units:
                selected.append(entry)
    return sorted(
        selected,
        key=lambda entry: (
            entry["input_unit"],
            entry.get("canonical_unit") or "",
            entry["task"],
            entry["canonical_endpoints"],
        ),
    )


def prepare(
    review_root: Path,
    *,
    batch_size: int = 50,
    top_k: int = 100,
    scope: str = "missing",
) -> None:
    payloads = {
        task: json.loads(_paths(task)["map"].read_text(encoding="utf-8"))
        for task in TASKS
    }
    rules = _expanded_rules(payloads)
    aliases: dict[tuple[str, str, str], dict[str, Any]] = {}
    if scope == "excluded":
        review, review_counts = _existing_excludes(payloads)
        missing = review
    else:
        missing: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for task in TASKS:
            missing.update(_active_missing(task, rules))
        review = {}
        for key, occurrences in missing.items():
            alias = find_whitespace_alias(key, rules)
            if alias is None:
                review[key] = occurrences
            else:
                aliases[key] = alias
        review_counts = {key: len(occurrences) for key, occurrences in review.items()}

    record_ids = {
        task: {
            occurrence["cleaned_record_id"]
            for key, occurrences in review.items()
            if key[0] == task
            for occurrence in occurrences
        }
        for task in TASKS
    }
    full_rows = {task: _source_rows(task, record_ids[task]) for task in TASKS}
    current_units = sorted(
        {
            value
            for payload in payloads.values()
            for entry in payload["entries"]
            for value in (entry["input_unit"], entry.get("canonical_unit"))
            if value
        }
    )
    input_units = sorted({key[2] for key in review})
    matches = fuzzy_matches(input_units, current_units, top_k)

    review_root.mkdir(parents=True, exist_ok=True)
    packet_dir = review_root / "packets"
    decision_dir = review_root / "decisions"
    packet_dir.mkdir(exist_ok=True)
    decision_dir.mkdir(exist_ok=True)
    batches = []
    for offset in range(0, len(input_units), batch_size):
        units = input_units[offset : offset + batch_size]
        batch_id = f"batch_{offset // batch_size:03d}"
        candidate_units = {
            candidate["unit"] for unit in units for candidate in matches[unit]
        }
        items = []
        for unit in units:
            keys = []
            for key in sorted(key for key in review if key[2] == unit):
                task, endpoint, _ = key
                occurrences = []
                for occurrence in review[key]:
                    record_id = occurrence["cleaned_record_id"]
                    occurrences.append(
                        {**occurrence, "source_row": full_rows[task][record_id]}
                    )
                keys.append(
                    {
                        "task": task,
                        "canonical_endpoint": endpoint,
                        "quantity_count": review_counts[key],
                        "current_rule": rules.get(key),
                        "affected_rows": occurrences,
                    }
                )
            items.append(
                {
                    "input_unit": unit,
                    "keys": keys,
                    "top_candidates": matches[unit],
                }
            )
        packet = {
            "batch_id": batch_id,
            "instructions": {
                "integrate": "Choose an existing canonical unit and positive fixed-point scale.",
                "preserve_new": "Use the exact input spelling as canonical_unit and scale 1.",
                "domain": "Do not assign domain; all new reviewed rules default to any.",
                "exclusions": "Do not exclude reviewed keys.",
            },
            "items": items,
            "candidate_rules": _candidate_rules(candidate_units, payloads),
        }
        packet_path = packet_dir / f"{batch_id}.json"
        packet_path.write_text(
            json.dumps(packet, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        batches.append(
            {
                "batch_id": batch_id,
                "packet": str(packet_path.relative_to(ROOT)),
                "decisions": str(
                    (decision_dir / f"{batch_id}.jsonl").relative_to(ROOT)
                ),
                "input_unit_count": len(units),
                "exact_key_count": sum(
                    1 for key in review if key[2] in set(units)
                ),
            }
        )

    alias_rows = [
        {
            "task": key[0],
            "canonical_endpoint": key[1],
            "input_unit": key[2],
            **rule,
        }
        for key, rule in sorted(aliases.items())
    ]
    (review_root / "whitespace_aliases.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in alias_rows),
        encoding="utf-8",
    )
    manifest = {
        "version": "exact_measurement_unit_review.v4",
        "review_scope": scope,
        "tasks": list(TASKS),
        "batch_size": batch_size,
        "fuzzy_top_k": top_k,
        "missing_exact_keys": len(missing) if scope == "missing" else 0,
        "excluded_exact_keys": len(review) if scope == "excluded" else 0,
        "whitespace_alias_keys": len(aliases),
        "review_exact_keys": len(review),
        "review_input_units": len(input_units),
        "active_review_rows": sum(review_counts.values()),
        "batches": batches,
        "inputs": {
            task: {
                name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
                for name, path in _paths(task).items()
            }
            for task in TASKS
        },
    }
    (review_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_scale(value: Any) -> str:
    text = str(value)
    try:
        scale = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"invalid scale: {value!r}") from error
    if not scale.is_finite() or scale <= 0 or "e" in text.casefold():
        raise ValueError(f"scale must be positive fixed-point: {value!r}")
    return text


def validate_decisions(
    expected_keys: set[tuple[str, str, str]],
    decisions: Iterable[dict[str, Any]],
    canonical_pool: set[str],
) -> list[dict[str, Any]]:
    validated = []
    seen: set[tuple[str, str, str]] = set()
    for decision in decisions:
        key = (
            str(decision.get("task") or ""),
            str(decision.get("canonical_endpoint") or ""),
            str(decision.get("input_unit") or ""),
        )
        if key not in expected_keys or key in seen:
            raise ValueError(f"unexpected or duplicate decision: {key!r}")
        kind = decision.get("decision")
        canonical_unit = str(decision.get("canonical_unit") or "")
        scale = _validate_scale(decision.get("scale"))
        if kind == "preserve_new":
            if canonical_unit != key[2] or scale != "1":
                raise ValueError(f"invalid preserve_new decision: {key!r}")
        elif kind == "integrate":
            if canonical_unit not in canonical_pool:
                raise ValueError(f"integration target is not in current pool: {key!r}")
        else:
            raise ValueError(f"invalid decision type for {key!r}: {kind!r}")
        if not str(decision.get("rationale") or "").strip():
            raise ValueError(f"decision has no rationale: {key!r}")
        seen.add(key)
        validated.append(
            {
                **decision,
                "task": key[0],
                "canonical_endpoint": key[1],
                "input_unit": key[2],
                "canonical_unit": canonical_unit,
                "scale": scale,
            }
        )
    missing = expected_keys - seen
    if missing:
        raise ValueError(f"{len(missing)} review decisions missing; first={min(missing)!r}")
    return validated


def _merge_entries(
    payload: dict[str, Any], additions: list[dict[str, Any]]
) -> dict[str, Any]:
    entries = [dict(entry) for entry in payload["entries"]]
    by_signature = {
        (
            entry["task"],
            entry["input_unit"],
            *_rule_signature(entry),
        ): entry
        for entry in entries
    }
    for addition in additions:
        signature = (
            addition["task"],
            addition["input_unit"],
            addition["action"],
            addition.get("canonical_unit"),
            addition.get("scale"),
            addition.get("domain", "any"),
        )
        entry = by_signature.get(signature)
        if entry is None:
            entry = {
                "task": addition["task"],
                "canonical_endpoints": [],
                "input_unit": addition["input_unit"],
                "action": addition["action"],
                "canonical_unit": addition.get("canonical_unit"),
                "scale": addition.get("scale"),
            }
            if addition.get("domain", "any") != "any":
                entry["domain"] = addition["domain"]
            entries.append(entry)
            by_signature[signature] = entry
        entry["canonical_endpoints"] = sorted(
            set(entry["canonical_endpoints"]) | {addition["canonical_endpoint"]}
        )
    entries.sort(
        key=lambda entry: (
            entry["input_unit"],
            entry.get("canonical_unit") or "",
            entry.get("scale") or "",
            entry.get("domain", "any"),
            entry["canonical_endpoints"],
        )
    )
    return {**payload, "entries": entries}


def _remove_exact_keys(
    payload: dict[str, Any], keys: set[tuple[str, str, str]]
) -> dict[str, Any]:
    entries = []
    for entry in payload["entries"]:
        endpoints = [
            endpoint
            for endpoint in entry["canonical_endpoints"]
            if (entry["task"], endpoint, entry["input_unit"]) not in keys
        ]
        if endpoints:
            entries.append({**entry, "canonical_endpoints": endpoints})
    return {**payload, "entries": entries}


def compose(review_root: Path, *, write_maps: bool = False) -> None:
    manifest = json.loads((review_root / "manifest.json").read_text(encoding="utf-8"))
    for task, inputs in manifest["inputs"].items():
        for name, frozen in inputs.items():
            path = ROOT / frozen["path"]
            if _sha256(path) != frozen["sha256"]:
                raise ValueError(f"review input changed after packet generation: {task}/{name}")
    payloads = {
        task: json.loads(_paths(task)["map"].read_text(encoding="utf-8"))
        for task in TASKS
    }
    rules = _expanded_rules(payloads)
    expected: set[tuple[str, str, str]] = set()
    decisions = []
    for batch in manifest["batches"]:
        packet = json.loads((ROOT / batch["packet"]).read_text(encoding="utf-8"))
        expected.update(
            (key["task"], key["canonical_endpoint"], item["input_unit"])
            for item in packet["items"]
            for key in item["keys"]
        )
        decision_path = ROOT / batch["decisions"]
        if not decision_path.exists():
            raise ValueError(f"missing decision shard: {decision_path}")
        decisions.extend(_read_jsonl(decision_path))
    canonical_pool = {
        str(rule["canonical_unit"])
        for rule in rules.values()
        if rule["action"] == "map" and rule.get("canonical_unit")
    }
    validated = validate_decisions(expected, decisions, canonical_pool)
    additions = [
        {
            "task": decision["task"],
            "canonical_endpoint": decision["canonical_endpoint"],
            "input_unit": decision["input_unit"],
            "action": "map",
            "canonical_unit": decision["canonical_unit"],
            "scale": decision["scale"],
        }
        for decision in validated
    ]
    additions.extend(_read_jsonl(review_root / "whitespace_aliases.jsonl"))

    proposal_dir = review_root / "proposed_maps"
    proposal_dir.mkdir(exist_ok=True)
    report = {"reviewed_keys": len(validated), "tasks": {}}
    for task in TASKS:
        base = payloads[task]
        if manifest.get("review_scope") == "excluded":
            base = _remove_exact_keys(
                base, {key for key in expected if key[0] == task}
            )
        merged = _merge_entries(
            base, [row for row in additions if row["task"] == task]
        )
        target = _paths(task)["map"] if write_maps else proposal_dir / f"{task}.json"
        target.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report["tasks"][task] = {
            "added_exact_keys": sum(row["task"] == task for row in additions),
            "replaced_exact_keys": sum(
                key[0] == task for key in expected
            )
            if manifest.get("review_scope") == "excluded"
            else 0,
            "output": str(target.relative_to(ROOT)),
            "sha256": _sha256(target),
        }
    (review_root / "composition_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "compose"))
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--scope", choices=("missing", "excluded"), default="missing")
    parser.add_argument("--write-maps", action="store_true")
    args = parser.parse_args(argv)
    review_root = (
        args.review_root
        if args.review_root.is_absolute()
        else ROOT / args.review_root
    )
    if args.command == "prepare":
        prepare(
            review_root,
            batch_size=args.batch_size,
            top_k=args.top_k,
            scope=args.scope,
        )
    else:
        compose(review_root, write_maps=args.write_maps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
