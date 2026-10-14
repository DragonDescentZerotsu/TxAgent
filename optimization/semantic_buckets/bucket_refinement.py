"""Partitioned two-pass family refinement with independent large-family audits."""

from __future__ import annotations
from . import execution
from . import storage
from collections import defaultdict
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
from typing import Any
import argparse
import fcntl
import duckdb
from jinja2 import Environment, StrictUndefined
import pandas as pd
from . import bucket_cards
from .bucket_cards import TASKS
from .inputs import REPOSITORY_ROOT, resolve_semantic_bucket_artifacts
from predict.utils.json import sha256_file
from predict.utils.json import write_json_atomic

VERSION = "semantic_refinement.partitioned_two_pass.v1"
MODEL = "openai/gpt-6-luna"
MAX_BATCH = 100
MAX_PROMPT_CHARS = 100000
PROFILE = (
    REPOSITORY_ROOT
    / "predict/api_client/providers/gpt6_luna_openrouter_standard_medium_64.json"
)
PROMPTS = Path(__file__).with_name("prompts") / "bucket_refinement"
PROMPT = PROMPTS / "pass2.jinja"
SKIN_ATOMS = Path(__file__).with_name("assets") / "skin_reaction/input_atoms.parquet"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _sources(task: str) -> dict[str, Path]:
    if task not in {"dili", "skin_reaction"}:
        raise ValueError(f"unsupported task: {task}")
    selected = resolve_semantic_bucket_artifacts(task)
    if not all(
        (
            selected.record_semantic_bucket_map,
            selected.semantic_bucket_weights,
            selected.central_ledger,
        )
    ):
        raise ValueError(f"{task}: selected release lacks required semantic inputs")
    atoms = (
        selected.semantic_map.parent / "input_atoms.parquet"
        if task == "dili"
        else SKIN_ATOMS
    )
    map_manifest = json.loads(selected.semantic_map_manifest.read_text())
    if (
        map_manifest.get("input_atoms_sha256")
        and sha256_file(atoms) != map_manifest["input_atoms_sha256"]
    ):
        raise ValueError("intrinsic atoms differ from the reviewed semantic map")
    evidence = REPOSITORY_ROOT / "data/evidence_libraries" / task / selected.release
    result = {
        "release_manifest": selected.manifest,
        "semantic_map": selected.semantic_map,
        "record_map": selected.record_semantic_bucket_map,
        "weights": selected.semantic_bucket_weights,
        "atoms": atoms,
        "retrieval_uids": selected.central_ledger.parent / "retrieval_uids.parquet",
        "canonical_records": evidence / "02_canonicalized/records.parquet",
        "merge_prompt": PROMPTS / "merge.jinja",
        "audit_prompt": PROMPTS / "audit.jinja",
        "provider_profile": PROFILE,
        "workflow_code": Path(__file__),
    }
    if any((not path.is_file() for path in result.values())):
        raise FileNotFoundError(f"{task}: missing selected refinement input")
    return result


def _families(sources: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    db = duckdb.connect()
    records = str(sources["record_map"])
    counts = db.execute(
        """WITH sizes AS (
        SELECT level,semantic_bucket_id,count(*) AS record_count
        FROM read_parquet(?) GROUP BY 1,2
    ), world AS (
        SELECT DISTINCT m.level,m.semantic_bucket_id FROM read_parquet(?) m
        JOIN read_parquet(?) r USING(source_row_uid,level)
    ) SELECT s.* FROM sizes s JOIN world w USING(level,semantic_bucket_id)""",
        [records, records, str(sources["retrieval_uids"])],
    ).df()
    db.close()
    atoms = pd.read_parquet(sources["atoms"], columns=["atom_id", "values_json"])
    mapping = pd.read_parquet(sources["semantic_map"])
    if atoms.atom_id.duplicated().any() or mapping.atom_id.duplicated().any():
        raise ValueError("selected atom mapping repeats an atom")
    joined = mapping.merge(atoms, on="atom_id", validate="one_to_one")
    joined = joined.merge(
        counts, on=["level", "semantic_bucket_id"], validate="many_to_one"
    )
    joined["endpoint"] = joined.values_json.map(
        lambda raw: str(
            json.loads(raw).get("canonical_endpoint_concept") or "__unknown__"
        )
    )
    families = (
        joined.groupby(["level", "source_id", "semantic_bucket_id"], sort=True)
        .agg(
            record_count=("record_count", "first"),
            atom_count=("atom_id", "size"),
            endpoint_count=("endpoint", "nunique"),
        )
        .reset_index()
    )
    if len(families) != len(counts):
        raise ValueError("selected family scope is not source-local")
    return (joined, families)


def _samples(sources: dict[str, Path]) -> dict[str, list[dict]]:
    db = duckdb.connect()
    schema = {
        row[0]
        for row in db.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)",
            [str(sources["canonical_records"])],
        ).fetchall()
    }
    fields = [
        name
        for name in (
            "source_name",
            "endpoint_name",
            "canonical_endpoint_concept",
            "canonical_assay_context",
            "canonical_assay_or_test",
            "assay_and_readout",
            "canonical_measurement_text",
            "canonical_unit_text",
            "result_label",
            "support_text",
        )
        if name in schema
    ]
    columns = ",".join(
        (f"substr(cast(c.{name} AS VARCHAR),1,240) AS {name}" for name in fields)
    )
    query = f"WITH within_atom AS (\n        SELECT semantic_bucket_id,source_row_uid,atom_id,\n        row_number() OVER(PARTITION BY semantic_bucket_id,atom_id\n                          ORDER BY hash(source_row_uid)) AS atom_rank\n        FROM read_parquet(?)\n    ), within_family AS (\n        SELECT *,row_number() OVER(PARTITION BY semantic_bucket_id\n            ORDER BY atom_rank,hash(atom_id),hash(source_row_uid)) AS family_rank\n        FROM within_atom\n    ) SELECT p.semantic_bucket_id,p.atom_id,p.source_row_uid,{columns}\n      FROM within_family p JOIN read_parquet(?) c USING(source_row_uid)\n      WHERE p.family_rank<=10 ORDER BY p.semantic_bucket_id,p.family_rank"
    frame = db.execute(
        query, [str(sources["record_map"]), str(sources["canonical_records"])]
    ).df()
    db.close()
    result: dict[str, list[dict]] = defaultdict(list)
    for row in frame.to_dict("records"):
        bucket = str(row.pop("semantic_bucket_id"))
        result[bucket].append(
            {key: value for key, value in row.items() if pd.notna(value)}
        )
    return result


def _cards(
    task: str,
    atoms: pd.DataFrame,
    families: pd.DataFrame,
    samples: dict[str, list[dict]],
) -> tuple[dict, dict]:
    labels = TASKS[task]["source_labels"]
    family_index = families.set_index(["level", "source_id", "semantic_bucket_id"])
    payloads, groups = ({}, {})
    for (level, source, bucket), rows in atoms.groupby(
        ["level", "source_id", "semantic_bucket_id"], sort=True
    ):
        family = family_index.loc[level, source, bucket]
        examples = samples.get(str(bucket), [])
        if not examples:
            raise ValueError(f"{task}/{bucket}: no representative records")
        payloads[str(bucket)] = (
            f"Physical records: {int(family.record_count)}; intrinsic atoms: {len(rows)}.",
            {
                "identity": {
                    "source_components": [
                        bucket_cards._component(source, rows, source_labels=labels)
                    ]
                },
                "sample_records": [
                    {"source_id": source, **record} for record in examples
                ],
            },
        )
        values = sorted(
            {
                str(json.loads(raw).get("canonical_assay_context") or "__unknown__")
                for raw in rows.values_json
            }
        )
        groups[str(bucket)] = {
            "level": level,
            "source_id": source,
            "endpoints": sorted(set(rows.endpoint)),
            "values": values,
            "record_count": int(family.record_count),
        }
    return (payloads, groups)


def _render(
    task: str, kind: str, batch: list[str], payloads: dict, groups: dict, endpoint: str
) -> str:
    first = groups[batch[0]]
    count = 5 if kind == "merge" else 10
    shown = "\n\n".join(
        (
            f"ID {i}\n{payloads[bucket][0]}\n"
            + bucket_cards._card(
                "Scientific profile",
                {
                    **payloads[bucket][1],
                    "sample_records": payloads[bucket][1]["sample_records"][:count],
                },
                source_labels=TASKS[task]["source_labels"],
            )
            for i, bucket in enumerate(batch)
        )
    )
    template = Environment(undefined=StrictUndefined).from_string(
        (PROMPTS / f"{kind}.jinja").read_text(encoding="utf-8")
    )
    return template.render(
        task=task,
        level=first["level"],
        source=first["source_id"],
        endpoint=endpoint,
        cards=shown,
    ).strip()


def _fit_batches(
    task: str, kind: str, batch: list[str], payloads: dict, groups: dict, endpoint: str
) -> list[dict]:
    if not batch:
        return []
    prompt = _render(task, kind, batch, payloads, groups, endpoint)
    if len(prompt) > MAX_PROMPT_CHARS or len(batch) > MAX_BATCH:
        if len(batch) < 2:
            raise ValueError(f"{task}/{kind}/{batch[0]}: card exceeds prompt budget")
        middle = len(batch) // 2
        return _fit_batches(
            task, kind, batch[:middle], payloads, groups, endpoint
        ) + _fit_batches(task, kind, batch[middle:], payloads, groups, endpoint)
    if kind == "merge" and len(batch) < 2:
        return []
    scope = {
        "level": groups[batch[0]]["level"],
        "source_id": groups[batch[0]]["source_id"],
        "endpoint": endpoint,
        "provider_family": "luna_openai_standard_medium",
    }
    return [
        {
            "request_id": _digest([VERSION, task, kind, scope, batch]),
            "task": task,
            "pass_number": 1 if kind == "merge" else 2,
            "scope": scope,
            "items": batch,
            "prompt": prompt,
        }
    ]


def _validate_response(raw: str, row: dict) -> dict:
    result = json.loads(raw)
    items = json.loads(row["items_json"])
    if row["pass_number"] in {1, 2}:
        return validate_review_merge(raw, len(items))
    if set(result) != {"reviews"} or not isinstance(result["reviews"], list):
        raise ValueError("large-family review must contain reviews")
    if len(result["reviews"]) != len(items):
        raise ValueError("large-family review omitted a family")
    seen = set()
    for review in result["reviews"]:
        if (
            not isinstance(review, dict)
            or not {"id", "risk", "rationale"}.issubset(review)
            or (not isinstance(review["id"], int))
            or (not 0 <= review["id"] < len(items))
            or (review["id"] in seen)
            or (review["risk"] not in {"low", "medium", "high"})
            or (not isinstance(review["rationale"], str))
            or (not review["rationale"].strip())
        ):
            raise ValueError("invalid large-family risk review")
        activities = review.get("suspected_activities", [])
        if not isinstance(activities, list) or not all(
            (isinstance(value, str) for value in activities)
        ):
            raise ValueError("invalid suspected activities")
        seen.add(review["id"])
    return result


def _scope_rows(groups: dict) -> pd.DataFrame:
    rows = []
    for bucket, group in groups.items():
        if group["record_count"] >= 100:
            continue
        endpoints = group["endpoints"]
        endpoint = endpoints[0] if len(endpoints) == 1 else "__mixed_endpoint__"
        rows.append(
            {
                "semantic_bucket_id": bucket,
                "level": group["level"],
                "source_id": group["source_id"],
                "endpoint": endpoint,
                "label": group["values"][0],
                "record_count": group["record_count"],
            }
        )
    result = pd.DataFrame(
        rows,
        columns=[
            "semantic_bucket_id",
            "level",
            "source_id",
            "endpoint",
            "label",
            "record_count",
        ],
    ).sort_values(["level", "source_id", "endpoint", "semantic_bucket_id"])
    if result.semantic_bucket_id.duplicated().any():
        raise ValueError("small family is assigned to multiple review patches")
    return result


def _requests(
    task: str, scopes: pd.DataFrame, payloads: dict, groups: dict
) -> list[dict]:
    requests = []
    for (level, source, endpoint), frame in scopes.groupby(
        ["level", "source_id", "endpoint"], sort=True
    ):
        members = [
            {
                "parent_id": json.dumps([level, source, endpoint]),
                "group_id": bucket,
                "label": groups[bucket]["values"][0],
                "values": groups[bucket]["values"],
            }
            for bucket in frame.semantic_bucket_id.astype(str)
        ]
        for batch in batch_groups(members, batch_size=MAX_BATCH):
            ids = [item["group_id"] for item in batch]
            for row in _fit_batches(task, "merge", ids, payloads, groups, endpoint):
                row["scope"] = {**row["scope"], "stage": "merge_pass1"}
                row["request_id"] = _digest(
                    [VERSION, task, 1, row["scope"], row["items"], row["prompt"]]
                )
                requests.append(row)
    seen = [bucket for row in requests for bucket in row["items"]]
    if len(seen) != len(set(seen)):
        raise ValueError("first-pass review patches overlap")
    return requests


def _load(output: Path) -> tuple[dict, object]:
    manifest = json.loads((output / "snapshot.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("two-pass review version changed")
    for entry in manifest["inputs"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"pinned two-pass input changed: {entry['path']}")
    for name in ("small_family_scopes.tsv", "family_sizes.tsv"):
        if sha256_file(output / name) != manifest[name.replace(".tsv", "_sha256")]:
            raise ValueError(f"frozen review census changed: {name}")
    connection = storage.review_database(output / "requests.sqlite3")
    initial = connection.execute(
        "SELECT request_id,prompt FROM requests WHERE pass_number IN (0,1) ORDER BY rowid"
    ).fetchall()
    if (
        _digest([(row["request_id"], row["prompt"]) for row in initial])
        != manifest["schedule_sha256"]
    ):
        raise ValueError("prepared review request bytes changed")
    for stage in (1, 2):
        key = f"pass{stage}_schedule_sha256"
        if key not in manifest:
            continue
        rows = connection.execute(
            "SELECT request_id,prompt FROM requests WHERE pass_number=? ORDER BY rowid",
            (stage,),
        ).fetchall()
        if (
            _digest([(row["request_id"], row["prompt"]) for row in rows])
            != manifest[key]
        ):
            raise ValueError(f"pass {stage} request bytes changed")
    return (manifest, connection)


def _provisional(output: Path, db) -> list[dict]:
    scopes = pd.read_csv(output / "small_family_scopes.tsv", sep="\t")
    by_bucket = scopes.set_index("semantic_bucket_id").to_dict("index")
    grouped = []
    used = set()
    for row in db.execute(
        "SELECT * FROM requests WHERE pass_number=1 ORDER BY request_id"
    ):
        if row["status"] != "complete":
            raise ValueError("first merge pass is incomplete")
        ids = json.loads(row["items_json"])
        response = validate_review_merge(row["response_json"], len(ids))
        for proposal in response["merge_sets"]:
            members = sorted((ids[index] for index in proposal["member_ids"]))
            if used.intersection(members):
                raise ValueError("first-pass family appears in two merge groups")
            used.update(members)
            grouped.append(
                (members, proposal["label"], proposal["rationale"], row["request_id"])
            )
    for bucket in sorted(set(by_bucket) - used):
        grouped.append(
            ([bucket], by_bucket[bucket]["label"], "Unmerged first-pass family", None)
        )
    result = []
    for members, label, rationale, request_id in grouped:
        scope = by_bucket[members[0]]
        if any(
            (
                by_bucket[member]["level"] != scope["level"]
                or by_bucket[member]["source_id"] != scope["source_id"]
                or by_bucket[member]["endpoint"] != scope["endpoint"]
                for member in members
            )
        ):
            raise ValueError("first-pass merge crossed its disjoint endpoint patch")
        result.append(
            {
                "group_id": "sbtp_" + _digest([VERSION, members])[:24],
                "member_bucket_ids": members,
                "label": label,
                "rationale": rationale,
                "pass1_request_id": request_id,
                "level": scope["level"],
                "source_id": scope["source_id"],
                "endpoint": scope["endpoint"],
            }
        )
    if sum((len(row["member_bucket_ids"]) for row in result)) != len(by_bucket):
        raise ValueError("first-pass groups do not partition small families")
    return result


def _group_card(group: dict, payloads: dict, groups: dict, task: str) -> str:
    members = group["member_bucket_ids"]
    source = group["source_id"]
    dimensions = defaultdict(set)
    samples = []
    for bucket in members:
        component = payloads[bucket][1]["identity"]["source_components"][0]
        for field, values in component["canonical_dimensions"].items():
            dimensions[field].update(values)
    records = [payloads[bucket][1]["sample_records"] for bucket in members]
    for index in range(max(map(len, records))):
        for record_list in records:
            if index < len(record_list) and len(samples) < 10:
                samples.append(record_list[index])
    combined = {
        "identity": {
            "source_components": [
                {
                    "source_id": source,
                    "source_label": TASKS[task]["source_labels"].get(source, source),
                    "canonical_dimensions": {
                        field: sorted(values)[:8]
                        for field, values in sorted(dimensions.items())
                    },
                }
            ]
        },
        "sample_records": samples,
    }
    physical = sum((groups[bucket]["record_count"] for bucket in members))
    atoms = sum(
        (
            int(payloads[bucket][0].split("intrinsic atoms: ", 1)[1].split(".", 1)[0])
            for bucket in members
        )
    )
    summaries = [
        f"    - {bucket}: {groups[bucket]['record_count']} records; "
        + "; ".join(groups[bucket]["values"][:2])[:160]
        for bucket in members
    ]
    return (
        f"Physical records: {physical}; intrinsic atoms: {atoms}; underlying families: {len(members)}.\nFirst-pass label: {group['label']}.\nFirst-pass rationale: {group['rationale']}\nUnderlying family summaries:\n"
        + "\n".join(summaries)
        + "\n"
        + bucket_cards._card(
            "Scientific profile", combined, source_labels=TASKS[task]["source_labels"]
        )
    )


def _render2(task: str, batch: list[dict], payloads: dict, groups: dict) -> str:
    first = batch[0]
    shown = "\n\n".join(
        (
            f"ID {i}\n{_group_card(group, payloads, groups, task)}"
            for i, group in enumerate(batch)
        )
    )
    return (
        Environment(undefined=StrictUndefined)
        .from_string(PROMPT.read_text())
        .render(
            task=task,
            level=first["level"],
            source=first["source_id"],
            endpoint=first["endpoint"],
            cards=shown,
        )
        .strip()
    )


def _fit2(task: str, batch: list[dict], payloads: dict, groups: dict) -> list[dict]:
    if len(batch) < 2:
        return []
    prompt = _render2(task, batch, payloads, groups)
    if len(batch) > MAX_BATCH or len(prompt) > MAX_PROMPT_CHARS:
        if len(batch) == 2:
            raise ValueError(f"{task}: second-pass pair exceeds prompt budget")
        middle = len(batch) // 2
        return _fit2(task, batch[:middle], payloads, groups) + _fit2(
            task, batch[middle:], payloads, groups
        )
    first = batch[0]
    scope = {
        "stage": "merge_pass2",
        "level": first["level"],
        "source_id": first["source_id"],
        "endpoint": first["endpoint"],
        "provider_family": "luna_openai_standard_medium",
    }
    ids = [group["group_id"] for group in batch]
    return [
        {
            "request_id": _digest([VERSION, task, 2, scope, ids, prompt]),
            "task": task,
            "pass_number": 2,
            "scope": scope,
            "items": ids,
            "prompt": prompt,
        }
    ]


def prepare_pass2(output: Path) -> dict:
    manifest, db = _load(output)
    if manifest["state"] != "complete_pass1":
        raise ValueError("first merge pass must complete")
    provisional = _provisional(output, db)
    sources = {name: Path(entry["path"]) for name, entry in manifest["inputs"].items()}
    atoms, families = _families(sources)
    payloads, groups = _cards(manifest["task"], atoms, families, _samples(sources))
    by_scope = defaultdict(list)
    for row in provisional:
        by_scope[row["level"], row["source_id"], row["endpoint"]].append(row)
    requests = []
    for (level, source, endpoint), members in sorted(by_scope.items()):
        batch_inputs = [
            {
                "parent_id": json.dumps([level, source, endpoint]),
                "group_id": row["group_id"],
                "label": row["label"],
                "values": [
                    groups[bucket]["values"][0] for bucket in row["member_bucket_ids"]
                ],
            }
            for row in members
        ]
        lookup = {row["group_id"]: row for row in members}
        for batch in batch_groups(batch_inputs, batch_size=MAX_BATCH):
            requests.extend(
                _fit2(
                    manifest["task"],
                    [lookup[item["group_id"]] for item in batch],
                    payloads,
                    groups,
                )
            )
    seen = [group_id for row in requests for group_id in row["items"]]
    if len(seen) != len(set(seen)):
        raise ValueError("second-pass review patches overlap")
    storage.insert_reviews(db, requests)
    db.close()
    manifest.update(
        state="prepared_pass2",
        provisional_groups=len(provisional),
        pass2_requests=len(requests),
        pass2_schedule_sha256=_digest(
            [(row["request_id"], row["prompt"]) for row in requests]
        ),
    )
    write_json_atomic(output / "snapshot.json", manifest)
    return manifest


def collect(output: Path) -> dict:
    manifest, db = _load(output)
    if manifest["state"] != "complete_pass2":
        raise ValueError("second merge pass is incomplete")
    provisional = _provisional(output, db)
    lookup = {row["group_id"]: row for row in provisional}
    final = []
    seen = set()
    for row in db.execute(
        "SELECT * FROM requests WHERE pass_number=2 ORDER BY request_id"
    ):
        if row["status"] != "complete":
            raise ValueError("second merge pass has incomplete requests")
        ids = json.loads(row["items_json"])
        response = validate_review_merge(row["response_json"], len(ids))
        for proposal in response["merge_sets"]:
            group_ids = [ids[index] for index in proposal["member_ids"]]
            if seen.intersection(group_ids):
                raise ValueError("second-pass group was merged twice")
            seen.update(group_ids)
            members = sorted(
                (
                    bucket
                    for group_id in group_ids
                    for bucket in lookup[group_id]["member_bucket_ids"]
                )
            )
            final.append(
                {
                    "request_id": row["request_id"],
                    "member_bucket_ids_json": json.dumps(members),
                    "label": proposal["label"],
                    "rationale": proposal["rationale"],
                }
            )
    db.close()
    for group_id, group in lookup.items():
        if group_id not in seen and len(group["member_bucket_ids"]) > 1:
            final.append(
                {
                    "request_id": group["pass1_request_id"],
                    "member_bucket_ids_json": json.dumps(group["member_bucket_ids"]),
                    "label": group["label"],
                    "rationale": group["rationale"],
                }
            )
    if len(
        {
            bucket
            for row in final
            for bucket in json.loads(row["member_bucket_ids_json"])
        }
    ) != sum((len(json.loads(row["member_bucket_ids_json"])) for row in final)):
        raise ValueError("final merge groups overlap")
    final.sort(key=lambda row: row["member_bucket_ids_json"])
    pd.DataFrame(
        final, columns=["request_id", "member_bucket_ids_json", "label", "rationale"]
    ).to_csv(output / "merge_proposals.tsv", sep="\t", index=False)
    reviews = _large_decisions(output, {})
    audit = pd.DataFrame(
        [
            {**review, "semantic_bucket_id": bucket}
            for bucket, review in reviews.items()
        ],
        columns=["semantic_bucket_id", "risk", "rationale", "suspected_activities"],
    )
    audit["suspected_activities_json"] = audit.suspected_activities.map(json.dumps)
    audit.drop(columns="suspected_activities").to_csv(
        output / "large_family_reviews.tsv", sep="\t", index=False
    )
    manifest.update(
        state="complete",
        final_merge_groups=len(final),
        large_audit_source_sha256=sha256_file(output / "large_family_reviews.tsv"),
    )
    write_json_atomic(output / "snapshot.json", manifest)
    review_path = output / "review_manifest.json"
    write_json_atomic(
        review_path,
        {
            "version": VERSION,
            "status": "candidate_review_required",
            "task": manifest["task"],
            "snapshot_sha256": sha256_file(output / "snapshot.json"),
            "merge_proposals": len(final),
            "large_family_reviews": len(audit),
            "high_risk_large_families": int(audit.risk.eq("high").sum()),
            "files": {
                name: {"path": name, "sha256": sha256_file(output / name)}
                for name in ("merge_proposals.tsv", "large_family_reviews.tsv")
            },
        },
    )
    decisions = {
        "version": VERSION,
        "task": manifest["task"],
        "review_manifest_sha256": sha256_file(review_path),
        "reviewer": "GPT-6 Luna partitioned two-pass merge review",
        "review_kind": "automated_unselected_candidate",
        "merges": [
            {
                "member_bucket_ids": json.loads(row["member_bucket_ids_json"]),
                "rationale": row["rationale"],
            }
            for row in final
        ],
        "splits": [],
    }
    write_json_atomic(output / "two_pass_decisions.json", decisions)
    return {
        "task": manifest["task"],
        "state": manifest["state"],
        "final_merge_groups": len(final),
        "large_high_risk": int(audit.risk.eq("high").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "run-pass1", "prepare-pass2", "run-pass2", "collect"),
    )
    parser.add_argument("--task", choices=("dili", "skin_reaction"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-inflight", type=int, default=32)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        if args.task is None:
            parser.error("prepare requires --task")
        result = prepare(args.task, output)
    elif args.action.startswith("run-pass"):
        result = run(output, int(args.action[-1]), args.max_inflight)
    elif args.action == "prepare-pass2":
        result = prepare_pass2(output)
    else:
        result = collect(output)
    print(json.dumps({key: result[key] for key in ("task", "state")}, indent=2))


def prepare(task: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    sources = _sources(task)
    sources["pass2_prompt"] = PROMPT
    for name in ("bucket_cards", "execution", "storage", "inputs"):
        sources[f"execution_{name}"] = Path(__file__).with_name(f"{name}.py")
    atoms, families = _families(sources)
    payloads, groups = _cards(task, atoms, families, _samples(sources))
    scopes = _scope_rows(groups)
    schedule = _requests(task, scopes, payloads, groups)
    audits = sorted(
        (bucket for bucket, group in groups.items() if group["record_count"] >= 100)
    )
    for start in range(0, len(audits), MAX_BATCH):
        for row in _fit_batches(
            task,
            "audit",
            audits[start : start + MAX_BATCH],
            payloads,
            groups,
            "__large_family__",
        ):
            row["pass_number"] = 0
            row["scope"]["stage"] = "large_audit"
            schedule.append(row)
    output.mkdir(parents=True)
    scopes.to_csv(output / "small_family_scopes.tsv", sep="\t", index=False)
    families.to_csv(output / "family_sizes.tsv", sep="\t", index=False)
    connection = storage.review_database(output / "requests.sqlite3")
    storage.insert_reviews(connection, schedule)
    connection.close()
    manifest = {
        "version": VERSION,
        "task": task,
        "state": "prepared_pass1",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "policy": {
            "merge_record_count_lt": 100,
            "max_batch": MAX_BATCH,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "model": MODEL,
            "service_tier": "standard",
            "reasoning_effort": "medium",
        },
        "pass1_requests": sum((row["pass_number"] == 1 for row in schedule)),
        "audit_requests": sum((row["pass_number"] == 0 for row in schedule)),
        "schedule_sha256": _digest(
            [(row["request_id"], row["prompt"]) for row in schedule]
        ),
        "small_family_scopes_sha256": sha256_file(output / "small_family_scopes.tsv"),
        "family_sizes_sha256": sha256_file(output / "family_sizes.tsv"),
    }
    write_json_atomic(output / "snapshot.json", manifest)
    return manifest


def run(output: Path, stage: int, max_inflight: int = 32) -> dict:
    from predict.api_client.pool import (
        build_provider_pool,
        load_provider_pool_config,
        ProviderPoolCompletionAdapter,
    )

    if stage not in {1, 2} or not 1 <= max_inflight <= 64:
        raise ValueError("stage must be 1 or 2 and max_inflight must be 1 through 64")
    with (output / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, db = _load(output)
        if manifest["state"] not in {
            f"prepared_pass{stage}",
            f"running_pass{stage}",
            f"incomplete_pass{stage}",
            f"complete_pass{stage}",
        }:
            db.close()
            raise ValueError("merge pass is not prepared")
        phases = (0, 1) if stage == 1 else (2,)
        placeholders = ",".join(("?" for _ in phases))
        rows = [
            dict(row)
            for row in db.execute(
                f"SELECT * FROM requests WHERE pass_number IN ({placeholders}) AND status!='complete'",
                phases,
            )
        ]
        if not rows:
            db.close()
            manifest["state"] = f"complete_pass{stage}"
            write_json_atomic(output / "snapshot.json", manifest)
            return manifest
        config = load_provider_pool_config(PROFILE)
        provider = build_provider_pool(
            config,
            timeout_s=3600,
            max_tokens=128000,
            temperature=None,
            tool_service_url="http://127.0.0.1:1",
            enable_group_tools=False,
            max_tool_rounds=0,
            reasoning_effort="medium",
            enable_thinking=True,
            transport_max_retries=0,
        )
        client = ProviderPoolCompletionAdapter(provider)
        phases = (0, 1) if stage == 1 else (2,)
        placeholders = ",".join(("?" for _ in phases))
        rows = [
            dict(row)
            for row in db.execute(
                f"SELECT * FROM requests WHERE pass_number IN ({placeholders}) AND status!='complete'",
                phases,
            )
        ]
        manifest["provider_config"] = config.public_dict()
        manifest["max_inflight"] = max_inflight
        manifest["state"] = f"running_pass{stage}"
        write_json_atomic(output / "snapshot.json", manifest)
        with ThreadPoolExecutor(max_workers=max_inflight) as pool:
            futures = {
                pool.submit(
                    execution.execute_review,
                    client,
                    row,
                    model=MODEL,
                    validate=_validate_response,
                ): row["request_id"]
                for row in rows
            }
            for future in as_completed(futures):
                result = future.result()
                columns = ",".join((f"{key}=?" for key in result))
                db.execute(
                    f"UPDATE requests SET {columns} WHERE request_id=?",
                    (*result.values(), futures[future]),
                )
                db.commit()
        incomplete = db.execute(
            f"SELECT count(*) FROM requests WHERE pass_number IN ({placeholders}) AND status!='complete'",
            phases,
        ).fetchone()[0]
        manifest["state"] = (
            f"incomplete_pass{stage}" if incomplete else f"complete_pass{stage}"
        )
        write_json_atomic(output / "snapshot.json", manifest)
    db.close()
    return manifest


def _large_decisions(output: Path, groups: dict) -> dict:
    db = storage.review_database(output / "requests.sqlite3")
    reviews = {}
    for row in db.execute(
        "SELECT * FROM requests WHERE pass_number=0 ORDER BY request_id"
    ):
        if row["status"] != "complete":
            raise ValueError("large-family audit is incomplete")
        response = _validate_response(row["response_json"], dict(row))
        ids = json.loads(row["items_json"])
        for review in response["reviews"]:
            bucket = ids[review["id"]]
            if bucket in reviews:
                raise ValueError("large-family audit repeats a bucket")
            reviews[bucket] = review
    db.close()
    sizes = pd.read_csv(output / "family_sizes.tsv", sep="\t")
    expected = set(
        sizes.loc[sizes.record_count.ge(100), "semantic_bucket_id"].astype(str)
    )
    if set(reviews) != expected:
        raise ValueError("large-family audit coverage differs")
    return reviews


def validate_review_merge(raw: str, count: int) -> dict[str, Any]:
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"merge_sets"}:
        raise ValueError("response must contain only merge_sets")
    if not isinstance(result["merge_sets"], list):
        raise ValueError("merge_sets must be a list")
    seen: set[int] = set()
    for group in result["merge_sets"]:
        if not isinstance(group, dict) or set(group) != {
            "member_ids",
            "label",
            "rationale",
        }:
            raise ValueError("merge group fields differ from approved schema")
        ids = group["member_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) < 2
            or any((type(x) is not int for x in ids))
        ):
            raise ValueError("merge group needs two or more integer IDs")
        if (
            len(ids) != len(set(ids))
            or any((x < 0 or x >= count for x in ids))
            or seen.intersection(ids)
        ):
            raise ValueError("merge group has repeated or out-of-range IDs")
        if not all(
            (
                isinstance(group[key], str) and group[key].strip()
                for key in ("label", "rationale")
            )
        ):
            raise ValueError("merge group lacks scientific label or rationale")
        seen.update(ids)
    return result


_REQUEST_DATABASE_WRITE_LOCK = threading.RLock()


def _text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub("\\W+", " ", normalized).strip()


def _search_tokens(group: dict) -> set[str]:
    tokens: set[str] = set()
    for value in [str(group["label"]), *group["values"]]:
        words = _text_key(value).split()
        tokens.update((word for word in words if len(word) > 1))
        for length in (2, 3):
            tokens.update(
                (
                    "".join((word[0] for word in words[start : start + length]))
                    for start in range(len(words) - length + 1)
                    if all((len(word) > 2 for word in words[start : start + length]))
                )
            )
    return tokens


def _nearby_batches(groups: list[dict], batch_size: int) -> list[list[dict]]:
    if len(groups) <= batch_size:
        return [groups]
    search_text = [
        _text_key(str(group["label"]))
        or _text_key(group["values"][0])
        or str(group["group_id"])
        for group in groups
    ]
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32
    ).fit_transform(search_text)
    postings: dict[str, list[int]] = defaultdict(list)
    member_tokens = [_search_tokens(group) for group in groups]
    for index, tokens in enumerate(member_tokens):
        for token in tokens:
            postings[token].append(index)
    remaining = set(range(len(groups)))
    batches = []
    cursor = 0
    while remaining:
        while cursor not in remaining:
            cursor += 1
        seed = cursor
        useful = sorted(
            (
                token
                for token in member_tokens[seed]
                if 1 < len(postings[token]) <= MAX_POSTING
            ),
            key=lambda token: (len(postings[token]), token),
        )[:4]
        candidates = sorted(
            ({index for token in useful for index in postings[token]} & remaining)
            - {seed}
        )
        if candidates:
            scores = np.asarray(
                matrix[candidates].dot(matrix[seed].T).toarray()
            ).ravel()
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: (
                    -float(item[1])
                    - sum(
                        (
                            1 / len(postings[token])
                            for token in useful
                            if token in member_tokens[item[0]]
                        )
                    ),
                    item[0],
                ),
            )
            chosen = [seed, *(index for index, _ in ranked[: batch_size - 1])]
        else:
            chosen = [seed]
        filler = cursor + 1
        chosen_set = set(chosen)
        while len(chosen) < batch_size and filler < len(groups):
            if filler in remaining and filler not in chosen_set:
                chosen.append(filler)
            filler += 1
        remaining.difference_update(chosen)
        batches.append([groups[index] for index in chosen])
    return batches


def batch_groups(groups: list[dict], *, batch_size: int = 40) -> list[list[dict]]:
    """Batch every provisional group once, across Pass-1 boundaries but not parents."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    by_parent: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for group in groups:
        parent = str(group["parent_id"])
        group_id = str(group["group_id"])
        values = group["values"]
        if (
            not isinstance(values, list)
            or not values
            or (not all((isinstance(v, str) for v in values)))
        ):
            raise ValueError(f"group {group_id} must contain canonical values")
        identity = (parent, group_id)
        if identity in seen:
            raise ValueError(f"duplicate provisional group: {identity}")
        seen.add(identity)
        by_parent[parent].append(group)
    batches = []
    for parent in sorted(by_parent):
        ordered = sorted(
            by_parent[parent],
            key=lambda group: (
                _text_key(str(group["label"])),
                _text_key(min(group["values"])),
                str(group["group_id"]),
            ),
        )
        batches.extend(_nearby_batches(ordered, batch_size))
    return batches


MAX_POSTING = 1000
if __name__ == "__main__":
    main()
