"""Prepare and run the V10 oral semantic/readout bucket refinement workflow."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from types import SimpleNamespace
from typing import Any

import httpx
from jinja2 import Environment, StrictUndefined
import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.common.json_utils import write_json_atomic
from semantic_buckets.artifacts import generation_root


VERSION = "bioavailability_semantic_readout.v1"
BATCH_SEED_VERSION = VERSION
TASK_NAME = "Bioavailability_Ma"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_URL = "http://dgx027:50001/v1"
ENDPOINTS: tuple[dict[str, str], ...] = (
    {
        "name": "dgx027",
        "base_url": BASE_URL,
        "provider": "local",
        "credential_env": "",
    },
)
ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = (
    ROOT
    / "outputs/analysis/evidence_library/"
    "bioavailability_relevance_levels_v10_degree20_v3"
)
RECORD_MAP = V3_ROOT / "record_relevance_map.parquet"
RECORDS = ROOT / "data/evidence_libraries/bioavailability_ma/v10/03_pair_buckets/records.parquet"
PROMPT_ROOT = Path(__file__).with_name("prompts") / "bioavailability_semantic_readout_v1"
CROSS_SOURCE_PROMPT_ROOT = (
    Path(__file__).with_name("prompts") / "bioavailability_semantic_cross_source_v1"
)
DEFAULT_REVIEW = Path("/tmp/bioavailability_semantic_readout_v1_prompt_review")
DEFAULT_OUTPUT = generation_root(
    "bioavailability_ma", "bioavailability_semantic_readout_v10_v1"
)
LEVELS = ("L2", "L3", "L4", "L5", "L6")
EXPECTED_RECORD_COUNT = 414_841
EXPECTED_ATOM_COUNT: int | None = 102_316
CROSS_SOURCE_LEVELS = ("L2",)
PAIR_KEY_OPTIONAL_TRAILING_COLUMNS = {"oral_exposure": 1}
SELECTOR_PROFILE_LIMIT: int | None = None
SELECTOR_VALUE_LIMIT: int | None = 5
MAX_SOURCE_ROUNDS = 3
MERGE_BATCH_SIZE = 40
SAMPLE_LIMIT = 5
CROSS_SOURCE_BATCH_SIZE = 40
CROSS_SOURCE_EXPLORATION_PER_SOURCE = 2
CROSS_SOURCE_BATCH_SEED_VERSION = "semantic_cross_source_candidates.v1"
MAX_ATTEMPTS = 4
WEIGHT_REQUEST_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": True}
}
LOW_MAX_TOKENS = 8_192
HIGH_MAX_TOKENS = 262_144
REQUEST_TIMEOUT_S = 900
MODEL_CONTEXT_LIMIT_OVERRIDE: int | None = None
TOKENIZER_ENCODING_OVERRIDE: str | None = None
COMPLETION_TOKEN_PARAMETER = "max_tokens"
SERVED_MODEL_ALIASES: frozenset[str] = frozenset()
_REQUEST_DATABASE_WRITE_LOCK = threading.RLock()
DEFER_TECHNICAL_BRANCHES = False
SEMANTIC_SIZE_REVIEW_REQUIRED = False
INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS = True
INCLUDE_DOWNSTREAM_PROMPT_REVIEW = True
SOURCE_LOCAL_FINAL_STATUS = "awaiting_cross_source_mapping"
PROMPT_REVIEW_TITLE = "Bioavailability V10 semantic/readout prompt review"


PAIR_COLUMNS = {
    "hf_bioavailability": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_bioavailability_report_type",
        "canonical_bioavailability_evidence_scope",
        "canonical_direct_condition_group",
    ),
    "oral_exposure": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_species_context",
        "canonical_biological_matrix",
        "canonical_oral_dose_key",
        "canonical_direct_condition_group",
    ),
    "fa": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_assay_context",
        "canonical_species_context",
    ),
    "fg": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_measurement_target_id",
        "canonical_assay_context",
        "canonical_species_context",
    ),
    "fh": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_assay_context",
        "canonical_species_context",
    ),
}

# V1 considered every pair-key column. Later versions may narrow this while
# retaining the complete pair key as immutable provenance.
REFINEMENT_COLUMNS = PAIR_COLUMNS
PROMPT_DIMENSION_COLUMNS = PAIR_COLUMNS

INITIAL_COLUMNS = {
    "hf_bioavailability": (
        "canonical_endpoint_concept",
        "canonical_bioavailability_report_type",
        "canonical_bioavailability_evidence_scope",
    ),
    "oral_exposure": (
        "canonical_endpoint_concept",
        "canonical_biological_matrix",
    ),
    "fa": ("canonical_endpoint_concept",),
    "fg": ("canonical_endpoint_concept",),
    "fh": ("canonical_endpoint_concept",),
}

SAMPLE_CARD_COLUMNS = {
    "hf_bioavailability": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_bioavailability_report_type",
        "canonical_bioavailability_evidence_scope",
    ),
    "oral_exposure": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_species_context",
        "canonical_biological_matrix",
        "canonical_oral_dose_key",
    ),
    "fa": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_assay_context",
        "canonical_species_context",
    ),
    "fg": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_measurement_scale_id",
        "canonical_measurement_target_id",
        "canonical_assay_context",
        "canonical_species_context",
    ),
    "fh": (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_assay_context",
        "canonical_species_context",
    ),
}

COLUMN_DESCRIPTIONS = {
    "canonical_endpoint_concept": "normalized biological or pharmacokinetic endpoint",
    "canonical_unit_text": "normalized measurement unit",
    "canonical_measurement_scale_id": "measurement scale or response representation",
    "canonical_bioavailability_report_type": "absolute, relative, or otherwise specified bioavailability report",
    "canonical_bioavailability_evidence_scope": "direct outcome or related evidence scope",
    "canonical_direct_condition_group": "normalized condition identity attached to a direct record",
    "canonical_species_context": "species or population in which the readout was measured",
    "canonical_biological_matrix": "sample matrix such as plasma or blood",
    "canonical_oral_dose_key": "normalized oral dose amount, unit, quantity kind, and basis",
    "canonical_assay_context": "assay system or experimental context",
    "canonical_measurement_target_id": "transporter, enzyme, or other measured target",
}


def validate_column_contracts() -> None:
    """Keep semantic decisions inside each source's pair-bucket schema."""
    if set(REFINEMENT_COLUMNS) != set(PAIR_COLUMNS):
        raise ValueError("refinement-column sources do not match pair-bucket sources")
    for source, pair_columns in PAIR_COLUMNS.items():
        pair = set(pair_columns)
        refinement = set(REFINEMENT_COLUMNS[source])
        prompt = set(PROMPT_DIMENSION_COLUMNS[source])
        initial = set(INITIAL_COLUMNS[source])
        if not refinement <= pair:
            raise ValueError(f"{source} refinement columns are not pair-bucket columns")
        if not prompt <= refinement:
            raise ValueError(f"{source} prompt dimensions are not refinement columns")
        if not initial <= refinement:
            raise ValueError(f"{source} initial columns are not refinement columns")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\0".join(values).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _render(name: str, payload: Mapping[str, Any], *, purpose: str = "semantic") -> str:
    environment = Environment(undefined=StrictUndefined, autoescape=False)
    template = environment.from_string(
        (PROMPT_ROOT / f"{name}.jinja").read_text(encoding="utf-8")
    )
    return template.render(
        purpose=purpose,
        payload=payload,
        compact_payload_json=_canonical_json(payload),
    ).strip()


def parse_pair_bucket(source: str, pair_bucket_key: str) -> dict[str, str]:
    values = json.loads(pair_bucket_key)
    columns = PAIR_COLUMNS[source]
    if not isinstance(values, list) or values[:1] != [source]:
        raise ValueError(f"invalid {source} pair bucket: {pair_bucket_key}")
    minimum = len(columns) - PAIR_KEY_OPTIONAL_TRAILING_COLUMNS.get(source, 0)
    if len(values) - 1 not in {minimum, len(columns)}:
        raise ValueError(
            f"{source} pair bucket has {len(values) - 1} values; expected "
            f"{minimum} or {len(columns)}"
        )
    parsed = {
        column: str(value)
        for column, value in zip(columns, values[1:], strict=False)
    }
    for column in columns[len(values) - 1 :]:
        parsed[column] = "__not_present_in_pair_key__"
    return parsed


def bucket_id(atom_ids: Iterable[str]) -> str:
    members = sorted(set(atom_ids))
    if not members:
        raise ValueError("a bucket must contain at least one atom")
    return _stable_id("sb", *members)


def deterministic_batches(
    item_ids: Sequence[str], *, seed: str, size: int = MERGE_BATCH_SIZE
) -> list[list[str]]:
    if size < 2:
        raise ValueError("merge batch size must be at least two")
    ordered = sorted(
        set(item_ids), key=lambda item: (_stable_id("order", seed, item), item)
    )
    return [ordered[index : index + size] for index in range(0, len(ordered), size)]


def degree_edges(
    bucket_ids: Sequence[str], *, level: str, degree: int = 25
) -> list[tuple[str, str]]:
    """Build a deterministic simple graph with the requested maximum degree."""
    keys = sorted(set(bucket_ids))
    if len(keys) != len(bucket_ids) or len(keys) < 2 or degree < 1:
        raise ValueError("degree scheduling requires unique buckets and positive degree")
    if len(keys) <= degree + 1:
        edges = []
        for left_index, right_index in combinations(range(len(keys)), 2):
            distance = right_index - left_index
            if distance < len(keys) / 2:
                edges.append((keys[left_index], keys[right_index]))
            elif distance > len(keys) / 2:
                edges.append((keys[right_index], keys[left_index]))
            else:
                edge = (keys[left_index], keys[right_index])
                edges.append(edge if left_index % 2 == 0 else edge[::-1])
    else:
        edges = []
        for offset in range(1, degree // 2 + 1):
            edges.extend(
                (keys[index], keys[(index + offset) % len(keys)])
                for index in range(len(keys))
            )
        if degree % 2:
            offset = len(keys) // 2
            limit = len(keys) // 2
            edges.extend(
                (keys[index], keys[index + offset])
                if index % 2 == 0
                else (keys[index + offset], keys[index])
                for index in range(limit)
            )
    if len(edges) != len({tuple(sorted(edge)) for edge in edges}):
        raise ValueError(f"degree-{degree} schedule contains repeated edges")
    return edges


def validate_merge_sets(
    response: Mapping[str, Any], valid_ids: Iterable[str]
) -> list[list[str]]:
    valid = set(valid_ids)
    merge_sets = response.get("merge_sets")
    if not isinstance(merge_sets, list):
        raise ValueError("merge_sets must be a list")
    seen: set[str] = set()
    output: list[list[str]] = []
    for index, entry in enumerate(merge_sets):
        if not isinstance(entry, Mapping):
            raise ValueError(f"merge_sets[{index}] must be an object")
        members = entry.get("member_ids")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"merge_sets[{index}].member_ids must contain at least two IDs")
        normalized = [str(member) for member in members]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"merge_sets[{index}] repeats an ID")
        unknown = set(normalized) - valid
        if unknown:
            raise ValueError(f"merge_sets[{index}] has unknown IDs: {sorted(unknown)}")
        overlap = set(normalized) & seen
        if overlap:
            raise ValueError(f"IDs occur in multiple merge sets: {sorted(overlap)}")
        seen.update(normalized)
        output.append(sorted(normalized))
    return output


def apply_merge_sets(
    items: Mapping[str, Sequence[str]], response: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Merge disjoint item memberships; omitted item IDs remain singleton groups."""
    groups = validate_merge_sets(response, items)
    consumed = {item for group in groups for item in group}
    groups.extend([[item] for item in sorted(items) if item not in consumed])
    output: dict[str, list[str]] = {}
    for group in groups:
        atoms = sorted({atom for item in group for atom in items[item]})
        if sum(len(items[item]) for item in group) != len(atoms):
            raise ValueError("merge inputs overlap in atom membership")
        output[bucket_id(atoms)] = atoms
    if Counter(atom for atoms in output.values() for atom in atoms) != Counter(
        atom for atoms in items.values() for atom in atoms
    ):
        raise ValueError("merge changed atom coverage")
    return output


def apply_split_decisions(
    active: Mapping[str, Sequence[str]],
    decisions: Mapping[str, str],
    children: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return active SPLIT/DEFERRED branches and terminal KEEP branches."""
    if set(decisions) != set(active):
        raise ValueError("every active bucket needs exactly one split decision")
    next_active: dict[str, list[str]] = {}
    terminal: dict[str, list[str]] = {}
    for parent_id, atoms in active.items():
        decision = decisions[parent_id]
        if decision == "keep":
            terminal[parent_id] = list(atoms)
            if parent_id in children:
                raise ValueError("KEEP bucket unexpectedly has children")
            continue
        if decision == "defer":
            next_active[parent_id] = list(atoms)
            if parent_id in children:
                raise ValueError("DEFERRED bucket unexpectedly has children")
            continue
        if decision != "split":
            raise ValueError(f"invalid split decision: {decision}")
        partition = children.get(parent_id)
        if not partition or len(partition) < 2:
            raise ValueError("SPLIT bucket must produce at least two children")
        if Counter(atom for child in partition.values() for atom in child) != Counter(atoms):
            raise ValueError("split children do not partition their parent")
        overlap = sum(len(child) for child in partition.values()) != len(
            {atom for child in partition.values() for atom in child}
        )
        if overlap:
            raise ValueError("split children overlap")
        next_active.update({key: list(value) for key, value in partition.items()})
    return next_active, terminal


def reverse_collapsed_splits(
    decisions: dict[str, str],
    children: dict[str, dict[str, list[str]]],
    technical_reasons: dict[str, str],
) -> int:
    """Treat a high-reasoning one-child merge as a final KEEP adjudication."""

    collapsed = [bucket for bucket, groups in children.items() if len(groups) == 1]
    for bucket in collapsed:
        decisions[bucket] = "keep"
        technical_reasons[bucket] = "high_reasoning_merge_reversed_split"
        del children[bucket]
    return len(collapsed)


def apply_readout_decision(
    atom_ids: Sequence[str],
    *,
    decision: str,
    depth: int,
    children: Mapping[str, Sequence[str]] | None = None,
    maximum_depth: int = 2,
) -> dict[str, Any]:
    """Apply one intrinsic coherence decision without crossing its parent bucket."""
    if not 0 <= depth <= maximum_depth:
        raise ValueError("invalid readout depth")
    members = list(atom_ids)
    if decision == "coherent":
        return {
            "active": {},
            "terminal": {bucket_id(members): members},
            "terminal_depth": max(1, depth),
            "termination_reason": "coherent_readout",
        }
    if decision != "split":
        raise ValueError(f"invalid readout decision: {decision}")
    if depth == maximum_depth:
        return {
            "active": {},
            "terminal": {},
            "blocked": {bucket_id(members): members},
            "blocked_depth": depth,
            "termination_reason": "incoherent_at_depth_cap",
        }
    if not children or len(children) < 2:
        raise ValueError("readout SPLIT must produce at least two children")
    if Counter(atom for child in children.values() for atom in child) != Counter(members):
        raise ValueError("readout children do not partition their parent")
    return {
        "active": {key: list(value) for key, value in children.items()},
        "terminal": {},
        "active_depth": depth + 1,
        "termination_reason": None,
    }


def _load_atoms() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    validate_column_contracts()
    mapping = pq.read_table(RECORD_MAP).to_pandas()
    mapping = mapping[mapping["level"].isin(LEVELS)].copy()
    if len(mapping) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"unexpected level-scoped record count: {len(mapping)}")
    if mapping["source_row_uid"].duplicated().any():
        raise ValueError("source_row_uid is not unique in the level-scoped mapping")

    key_columns = ["level", "source_id", "pair_bucket_key", "node_key", "relevance_bucket"]
    counts = mapping.groupby(key_columns, sort=True, dropna=False).size().rename("record_count")
    representatives = (
        mapping.sort_values("canonical_record_id")
        .drop_duplicates(key_columns)
        [key_columns + ["source_row_uid", "canonical_record_id"]]
        .set_index(key_columns)
    )
    atoms = counts.to_frame().join(representatives).reset_index()
    if EXPECTED_ATOM_COUNT is not None and len(atoms) != EXPECTED_ATOM_COUNT:
        raise ValueError(f"unexpected level-scoped pair-bucket count: {len(atoms)}")

    record_columns = sorted(
        {
            "source_row_uid",
            "canonical_record_id",
            *(column for columns in SAMPLE_CARD_COLUMNS.values() for column in columns),
        }
    )
    records = pq.read_table(RECORDS, columns=record_columns).to_pandas()
    records = records.drop_duplicates("source_row_uid").set_index("source_row_uid")
    sample_cards: dict[str, dict[str, Any]] = {}
    atom_rows = []
    for row in atoms.itertuples(index=False):
        atom = _stable_id("atom", row.level, row.source_id, row.pair_bucket_key)
        values = parse_pair_bucket(row.source_id, row.pair_bucket_key)
        source_record = records.loc[row.source_row_uid].to_dict()
        card = {
            column: source_record[column]
            for column in SAMPLE_CARD_COLUMNS[row.source_id]
            if pd.notna(source_record.get(column)) and str(source_record[column]).strip()
        }
        sample_cards[atom] = {
            "canonical_record_id": row.canonical_record_id,
            **(
                {"pair_bucket_key": row.pair_bucket_key}
                if INCLUDE_PAIR_BUCKET_KEY_IN_SAMPLE_CARDS
                else {}
            ),
            **card,
        }
        atom_rows.append(
            {
                "atom_id": atom,
                "level": row.level,
                "source_id": row.source_id,
                "pair_bucket_key": row.pair_bucket_key,
                "initial_node_key": row.node_key,
                "initial_relevance_bucket": row.relevance_bucket,
                "record_count": int(row.record_count),
                "values_json": _canonical_json(values),
            }
        )
    return pd.DataFrame(atom_rows), sample_cards


def initial_buckets(atoms: pd.DataFrame) -> dict[tuple[str, str], dict[str, list[str]]]:
    output: dict[tuple[str, str], dict[str, list[str]]] = {}
    for (level, source, node), rows in atoms.groupby(
        ["level", "source_id", "initial_node_key"], sort=True
    ):
        members = sorted(rows["atom_id"].tolist())
        output.setdefault((level, source), {})[bucket_id(members)] = members
    return output


def _atom_lookup(atoms: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output = {}
    for row in atoms.to_dict("records"):
        row = dict(row)
        row["values"] = json.loads(row.pop("values_json"))
        output[row["atom_id"]] = row
    return output


def _bucket_payload(
    bucket: str,
    atom_ids: Sequence[str],
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    atoms = [lookup[atom] for atom in atom_ids]
    columns = PROMPT_DIMENSION_COLUMNS[str(atoms[0]["source_id"])]
    dimensions = {
        column: sorted({str(atom["values"][column]) for atom in atoms})
        for column in columns
    }
    samples = [
        {
            column: sample_cards[atom][column]
            for column in columns
            if column in sample_cards[atom]
        }
        for atom in sorted(atom_ids)[:SAMPLE_LIMIT]
    ]
    return {
        "bucket_id": bucket,
        "pair_bucket_count": len(atom_ids),
        "record_count": sum(int(atom["record_count"]) for atom in atoms),
        "canonical_dimensions": dimensions,
        "sample_records": samples,
    }


def _value_payload(
    bucket: str,
    atom_ids: Sequence[str],
    column: str,
    lookup: Mapping[str, Mapping[str, Any]],
    *,
    context_columns: Iterable[str] = (),
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    by_value: dict[str, list[str]] = {}
    for atom_id in atom_ids:
        value = str(lookup[atom_id]["values"][column])
        by_value.setdefault(value, []).append(atom_id)
    value_items: dict[str, list[str]] = {}
    values = []
    for index, (value, members) in enumerate(sorted(by_value.items())):
        value_id = f"v{index:06d}"
        value_items[value_id] = sorted(members)
        values.append(
            {
                "value_id": value_id,
                "value": value,
                "pair_bucket_count": len(members),
                "record_count": sum(int(lookup[item]["record_count"]) for item in members),
            }
        )
    identity = {
        context: sorted(
            {str(lookup[atom]["values"][context]) for atom in atom_ids}
        )
        for context in context_columns
        if context != column
    }
    return {
        "bucket_id": bucket,
        "current_bucket_identity": identity,
        "column": column,
        "column_description": COLUMN_DESCRIPTIONS[column],
        "values": values,
    }, value_items


def _column_selection_payload(
    level: str,
    source: str,
    buckets: Mapping[str, Sequence[str]],
    lookup: Mapping[str, Mapping[str, Any]],
    consumed: Iterable[str],
) -> dict[str, Any]:
    consumed_set = set(consumed)
    profiled = dict(sorted(buckets.items()))
    if SELECTOR_PROFILE_LIMIT is not None:
        profiled = dict(list(profiled.items())[:SELECTOR_PROFILE_LIMIT])
    active_buckets = []
    for bucket, atom_ids in profiled.items():
        active_buckets.append(
            {
                "bucket_id": bucket,
                "current_bucket_identity": {
                    column: sorted(
                        {str(lookup[atom]["values"][column]) for atom in atom_ids}
                    )
                    for column in sorted(consumed_set)
                },
                "pair_bucket_count": len(atom_ids),
                "record_count": sum(
                    int(lookup[atom]["record_count"]) for atom in atom_ids
                ),
            }
        )
    candidates = []
    for column in REFINEMENT_COLUMNS[source]:
        if column in consumed_set:
            continue
        distinct_counts = [
            len({str(lookup[atom]["values"][column]) for atom in atom_ids})
            for atom_ids in buckets.values()
        ]
        profiles = []
        for bucket, atom_ids in profiled.items():
            values = Counter(str(lookup[atom]["values"][column]) for atom in atom_ids)
            profiles.append(
                {
                    "bucket_id": bucket,
                    "distinct_values": len(values),
                    "example_values": (
                        sorted(values)
                        if SELECTOR_VALUE_LIMIT is None
                        else sorted(values)[:SELECTOR_VALUE_LIMIT]
                    ),
                }
            )
        candidates.append(
            {
                "column": column,
                "description": COLUMN_DESCRIPTIONS[column],
                "buckets_with_multiple_values": sum(
                    count > 1 for count in distinct_counts
                ),
                "maximum_distinct_values_in_one_bucket": max(distinct_counts),
                "bucket_profiles": profiles,
            }
        )
    return {
        "task": TASK_NAME,
        "level": level,
        "source_id": source,
        "active_bucket_count": len(buckets),
        "profiled_bucket_count": len(profiled),
        "active_buckets": active_buckets,
        "consumed_columns": sorted(consumed_set),
        "candidate_columns": candidates,
    }


def _merge_bucket_payload(
    level: str,
    source: str,
    buckets: Mapping[str, Sequence[str]],
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "task": TASK_NAME,
        "level": level,
        "source_id": source,
        "buckets": [
            _bucket_payload(bucket, members, lookup, sample_cards)
            for bucket, members in sorted(buckets.items())
        ],
    }


def _render_cross_source(batch: Mapping[str, Any]) -> str:
    template = Environment(
        loader=None,
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    ).from_string(
        (CROSS_SOURCE_PROMPT_ROOT / "merge_buckets.jinja").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "task": batch["task"],
        "level": batch["level"],
        "anchor_bucket_id": batch["anchor_bucket_id"],
        "buckets": batch["buckets"],
    }
    return template.render(compact_payload_json=_canonical_json(payload)).strip()


def _cross_source_bucket_payloads(
    source_map: pd.DataFrame,
    atoms: pd.DataFrame,
    sample_cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup = _atom_lookup(atoms)
    payloads = {}
    for (level, source, bucket), members in source_map.groupby(
        ["level", "source_id", "source_semantic_bucket_id"], sort=True
    ):
        atom_ids = sorted(members["atom_id"].tolist())
        payload = _bucket_payload(str(bucket), atom_ids, lookup, sample_cards)
        payload.update({"level": str(level), "source_id": str(source)})
        payloads[str(bucket)] = payload
    return payloads


def _cross_source_tokens(bucket: Mapping[str, Any]) -> set[str]:
    values = bucket["canonical_dimensions"].values()
    return {
        token
        for value_group in values
        for value in value_group
        for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if len(token) > 1 and token != "unknown"
    }


def cross_source_candidate_batches(
    source_map: pd.DataFrame,
    atoms: pd.DataFrame,
    sample_cards: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int = CROSS_SOURCE_BATCH_SIZE,
    exploration_per_source: int = CROSS_SOURCE_EXPLORATION_PER_SOURCE,
) -> list[dict[str, Any]]:
    """Route every non-dominant-source bucket to cross-source candidates.

    Candidate routing is deterministic and never merges buckets. DeepSeek remains
    the only component allowed to make a scientific merge decision.
    """

    if batch_size < 2 or exploration_per_source < 0:
        raise ValueError("invalid cross-source candidate settings")
    payloads = _cross_source_bucket_payloads(source_map, atoms, sample_cards)
    tokens = {
        bucket: _cross_source_tokens(payload) for bucket, payload in payloads.items()
    }
    batches = []
    for level in CROSS_SOURCE_LEVELS:
        level_buckets = {
            source: sorted(rows["source_semantic_bucket_id"].unique())
            for source, rows in source_map[source_map["level"].eq(level)].groupby(
                "source_id", sort=True
            )
        }
        if len(level_buckets) < 2:
            continue
        dominant_source = min(
            level_buckets,
            key=lambda source: (-len(level_buckets[source]), source),
        )
        other_sources = sorted(level_buckets)
        quota = max(1, (batch_size - 1) // (len(other_sources) - 1))
        anchors = [
            bucket
            for source in other_sources
            if source != dominant_source
            for bucket in level_buckets[source]
        ]
        for anchor in anchors:
            anchor_payload = payloads[anchor]
            anchor_tokens = tokens[anchor]
            selected = []
            for source in other_sources:
                if source == anchor_payload["source_id"]:
                    continue
                candidates = level_buckets[source]
                ranked = sorted(
                    candidates,
                    key=lambda candidate: (
                        -len(anchor_tokens & tokens[candidate]),
                        len(anchor_tokens | tokens[candidate]),
                        _stable_id(
                            "cross-source-similarity",
                            CROSS_SOURCE_BATCH_SEED_VERSION,
                            anchor,
                            candidate,
                        ),
                        candidate,
                    ),
                )
                semantic_count = max(0, quota - exploration_per_source)
                chosen = ranked[:semantic_count]
                remaining = [candidate for candidate in candidates if candidate not in chosen]
                exploratory = sorted(
                    remaining,
                    key=lambda candidate: (
                        _stable_id(
                            "cross-source-exploration",
                            CROSS_SOURCE_BATCH_SEED_VERSION,
                            anchor,
                            candidate,
                        ),
                        candidate,
                    ),
                )[:exploration_per_source]
                chosen.extend(exploratory)
                if len(chosen) < min(quota, len(candidates)):
                    chosen.extend(
                        candidate
                        for candidate in ranked
                        if candidate not in chosen
                    )
                selected.extend(chosen[:quota])
            selected = selected[: batch_size - 1]
            batches.append(
                {
                    "task": TASK_NAME,
                    "level": level,
                    "anchor_bucket_id": anchor,
                    "dominant_source_id": dominant_source,
                    "buckets": [anchor_payload]
                    + [payloads[candidate] for candidate in selected],
                }
            )
    return batches


def validate_cross_source_merge_response(
    response: Mapping[str, Any], batch: Mapping[str, Any]
) -> list[list[str]]:
    valid_ids = [bucket["bucket_id"] for bucket in batch["buckets"]]
    merge_sets = validate_merge_sets(response, valid_ids)
    if len(merge_sets) > 1:
        raise ValueError("cross-source response may contain at most one merge set")
    if not merge_sets:
        return []
    members = merge_sets[0]
    anchor = str(batch["anchor_bucket_id"])
    if anchor not in members:
        raise ValueError("cross-source merge set must include the anchor bucket")
    sources = {
        str(bucket["source_id"])
        for bucket in batch["buckets"]
        if bucket["bucket_id"] in members
    }
    if len(sources) < 2:
        raise ValueError("cross-source merge set must span at least two sources")
    return merge_sets


def _endpoint_contract() -> tuple[int, list[str]]:
    if MODEL_CONTEXT_LIMIT_OVERRIDE is not None:
        if MODEL_CONTEXT_LIMIT_OVERRIDE <= 0:
            raise ValueError("MODEL_CONTEXT_LIMIT_OVERRIDE must be positive")
        return MODEL_CONTEXT_LIMIT_OVERRIDE, [MODEL]
    response = httpx.get(f"{BASE_URL}/models", timeout=10)
    response.raise_for_status()
    rows = response.json().get("data", [])
    models = sorted(str(row.get("id")) for row in rows)
    matching = [row for row in rows if row.get("id") == MODEL]
    if len(matching) != 1:
        raise ValueError(f"endpoint does not uniquely serve {MODEL}: {models}")
    maximum = int(matching[0].get("max_model_len") or 0)
    if maximum <= 0:
        raise ValueError("endpoint did not report max_model_len")
    return maximum, models


class _IndependentCompletionPool:
    """Route calls across independent endpoint-specific concurrency pools."""

    def __init__(
        self,
        providers: Sequence[tuple[str, str, Any, int]],
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60,
    ) -> None:
        self._providers = list(providers)
        self._active = [0] * len(self._providers)
        self._completed = [0] * len(self._providers)
        self._failures = [0] * len(self._providers)
        self._consecutive_failures = [0] * len(self._providers)
        self._circuit_open_until = [0.0] * len(self._providers)
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._next = 0
        self._condition = threading.Condition()
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        with self._condition:
            while True:
                now = time.monotonic()
                available = [
                    index
                    for index, provider in enumerate(self._providers)
                    if self._active[index] < provider[3]
                    and self._circuit_open_until[index] <= now
                ]
                if available:
                    index = min(
                        available,
                        key=lambda candidate: (
                            self._active[candidate] / self._providers[candidate][3],
                            (candidate - self._next) % len(self._providers),
                        ),
                    )
                    self._active[index] += 1
                    self._next = (index + 1) % len(self._providers)
                    break
                reopen_times = [
                    value for value in self._circuit_open_until if value > now
                ]
                timeout = min(reopen_times) - now if reopen_times else None
                self._condition.wait(timeout=timeout)
        name, base_url, client, _ = self._providers[index]
        try:
            completion = _stream_completion(client, kwargs)
        except Exception:
            with self._condition:
                self._active[index] -= 1
                self._failures[index] += 1
                self._consecutive_failures[index] += 1
                if self._consecutive_failures[index] >= self._failure_threshold:
                    self._circuit_open_until[index] = (
                        time.monotonic() + self._cooldown_seconds
                    )
                self._condition.notify_all()
            raise
        with self._condition:
            self._active[index] -= 1
            self._completed[index] += 1
            self._consecutive_failures[index] = 0
            self._circuit_open_until[index] = 0
            self._condition.notify_all()
        return SimpleNamespace(
            id=getattr(completion, "id", None),
            model=completion.model,
            usage=completion.usage,
            choices=completion.choices,
            provider_name=name,
            provider_base_url=base_url,
        )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            now = time.monotonic()
            return [
                {
                    "name": provider[0],
                    "base_url": provider[1],
                    "max_inflight": provider[3],
                    "active": self._active[index],
                    "completed": self._completed[index],
                    "failures": self._failures[index],
                    "circuit_open": self._circuit_open_until[index] > now,
                }
                for index, provider in enumerate(self._providers)
            ]


def _stream_completion(client: Any, kwargs: Mapping[str, Any]) -> Any:
    """Collect one streaming response while keeping the legacy completion shape."""

    request = dict(kwargs)
    request.update(stream=True, stream_options={"include_usage": True})
    content: list[str] = []
    reasoning: list[str] = []
    generation_id = model = None
    usage = None
    response = client.chat.completions.create(**request)
    if hasattr(response, "choices"):
        return response
    for chunk in response:
        generation_id = getattr(chunk, "id", None) or generation_id
        model = getattr(chunk, "model", None) or model
        usage = getattr(chunk, "usage", None) or usage
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            content.append(delta.content)
        extras = getattr(delta, "model_extra", None) or {}
        thought = (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
            or extras.get("reasoning_content")
            or extras.get("reasoning")
        )
        if thought:
            reasoning.append(thought)
    message = SimpleNamespace(
        content="".join(content),
        reasoning_content="".join(reasoning) or None,
        reasoning=None,
        model_extra={},
    )
    return SimpleNamespace(
        id=generation_id,
        model=model,
        usage=usage,
        choices=[SimpleNamespace(message=message)],
    )


def _build_completion_pool(
    per_endpoint_parallelism: int,
) -> tuple[_IndependentCompletionPool, list[dict[str, Any]], int]:
    """Build and preflight each configured endpoint without exposing credentials."""

    from data.processing.llm_api import DEFAULT_ENV_FILE, openai_compatible_client

    providers = []
    receipts = []
    for spec in ENDPOINTS:
        provider = spec["provider"]
        max_inflight = int(spec.get("max_inflight", per_endpoint_parallelism))
        if max_inflight <= 0:
            raise ValueError(f"endpoint {spec['name']!r} max_inflight must be positive")
        env_file = None if provider == "local" else DEFAULT_ENV_FILE
        client, credential = openai_compatible_client(
            base_url=spec["base_url"],
            provider=provider,
            env_file=env_file,
            credential_env=spec.get("credential_env") or None,
            max_connections=max_inflight,
            timeout_s=REQUEST_TIMEOUT_S,
            max_retries=0,
        )
        models = sorted(model.id for model in client.models.list().data)
        if MODEL not in models:
            raise ValueError(
                f"endpoint {spec['name']!r} does not advertise {MODEL!r}: {models}"
            )
        if provider == "local" and credential:
            raise ValueError("local DeepSeek unexpectedly selected a credential")
        providers.append(
            (spec["name"], spec["base_url"], client, max_inflight)
        )
        receipts.append(
            {
                "name": spec["name"],
                "base_url": spec["base_url"],
                "provider": provider,
                "credential_env": credential,
                "requested_model": MODEL,
                "advertised_models": models,
                "max_inflight": max_inflight,
            }
        )
    return (
        _IndependentCompletionPool(providers),
        receipts,
        sum(provider[3] for provider in providers),
    )


def _token_count(text: str) -> int:
    if TOKENIZER_ENCODING_OVERRIDE is not None:
        import tiktoken

        return len(tiktoken.get_encoding(TOKENIZER_ENCODING_OVERRIDE).encode(text))
    tokenize_url = BASE_URL.rsplit("/v1", 1)[0] + "/tokenize"
    response = httpx.post(
        tokenize_url,
        json={"model": MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return int(response.json()["count"])


def _request_database(
    path: Path, *, journal_mode: str = "WAL"
) -> sqlite3.Connection:
    if journal_mode not in {"WAL", "DELETE"}:
        raise ValueError("journal_mode must be WAL or DELETE")
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute(f"PRAGMA journal_mode={journal_mode}")
    connection.execute(
        f"PRAGMA synchronous={'FULL' if journal_mode == 'DELETE' else 'NORMAL'}"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            phase TEXT NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            prompt TEXT NOT NULL,
            reasoning_effort TEXT NOT NULL,
            max_tokens INTEGER NOT NULL,
            validation_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            response_json TEXT,
            reasoning_content TEXT,
            served_model TEXT,
            provider_name TEXT,
            provider_base_url TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            error TEXT
        )
        """
    )
    request_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(requests)")
    }
    for column in ("provider_name", "provider_base_url"):
        if column not in request_columns:
            connection.execute(f"ALTER TABLE requests ADD COLUMN {column} TEXT")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS request_attempt_receipts (
            request_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            receipt_json TEXT NOT NULL,
            PRIMARY KEY (request_id, attempt)
        )
        """
    )
    connection.commit()
    return connection


def _seed_request_database(path: Path, seed: Path | None) -> dict[str, Any] | None:
    if seed is None or path.exists():
        return None
    if not seed.is_file():
        raise FileNotFoundError(f"seed request cache does not exist: {seed}")
    source = sqlite3.connect(f"file:{seed}?mode=ro", uri=True, timeout=60)
    try:
        statuses = dict(source.execute(
            "SELECT status,count(*) FROM requests GROUP BY status"
        ))
        if not statuses.get("complete"):
            raise ValueError(f"seed request cache has no complete requests: {statuses}")
        served_models = {
            str(row[0])
            for row in source.execute(
                "SELECT DISTINCT served_model FROM requests WHERE served_model IS NOT NULL"
            )
        }
        if served_models != {MODEL}:
            raise ValueError(
                f"seed request cache served models do not match {MODEL}: "
                f"{sorted(served_models)}"
            )
        destination = sqlite3.connect(path, timeout=60)
        try:
            source.backup(destination)
            destination.execute(
                "DELETE FROM request_attempt_receipts WHERE request_id IN "
                "(SELECT request_id FROM requests WHERE status!='complete')"
            )
            destination.execute("DELETE FROM requests WHERE status!='complete'")
            destination.commit()
        finally:
            destination.close()
    finally:
        source.close()
    migrated = _request_database(path)
    try:
        copied = int(migrated.execute("SELECT count(*) FROM requests").fetchone()[0])
    finally:
        migrated.close()
    return {
        "path": str(seed),
        "sha256": _file_sha256(seed),
        "complete_requests_copied": copied,
        "source_status_counts": dict(sorted(statuses.items())),
        "noncomplete_requests_excluded": sum(
            count for status, count in statuses.items() if status != "complete"
        ),
    }


def _queue_request(
    connection: sqlite3.Connection,
    *,
    request_id: str,
    kind: str,
    phase: str,
    prompt: str,
    reasoning_effort: str,
    max_tokens: int,
    validation: Mapping[str, Any],
    commit: bool = True,
) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    values = (
        request_id,
        kind,
        phase,
        prompt_hash,
        prompt,
        reasoning_effort,
        max_tokens,
        _canonical_json(validation),
    )
    with _REQUEST_DATABASE_WRITE_LOCK:
        connection.execute(
            """
            INSERT OR IGNORE INTO requests
            (request_id,kind,phase,prompt_sha256,prompt,reasoning_effort,max_tokens,validation_json)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            values,
        )
        stored = connection.execute(
            """
            SELECT request_id,kind,phase,prompt_sha256,prompt,reasoning_effort,max_tokens,
                   validation_json
            FROM requests WHERE request_id=?
            """,
            (request_id,),
        ).fetchone()
        if tuple(stored) != values:
            raise ValueError(f"cached request contract changed: {request_id}")
        if commit:
            connection.commit()
    return request_id


def _validate_decision_response(
    kind: str, result: dict[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    decisions = {
        "select_column": {"stop", "select"},
        "split_bucket": {"keep", "split"},
        "readout_coherence": {"coherent", "split"},
        "retrieval_eligibility": {"include", "exclude"},
    }
    decision = result.get("decision")
    if decision not in decisions[kind]:
        raise ValueError(f"invalid {kind} decision")
    expected = {"decision", "rationale"}
    if kind == "retrieval_eligibility":
        expected.add("reason_code")
        reason = result.get("reason_code")
        if reason not in validation["reason_codes"]:
            raise ValueError("retrieval eligibility has an unknown reason code")
        informative = reason == "potentially_informative"
        if informative != (decision == "include"):
            raise ValueError("retrieval eligibility decision and reason disagree")
    elif decision in {"select", "split"} and kind != "split_bucket":
        if result.get("column") not in validation["candidate_columns"]:
            raise ValueError("selected column was not supplied")
        expected.add("column")
    if set(result) != expected:
        raise ValueError(f"{kind} response has unexpected fields")
    return result


def _validate_merge_response(
    kind: str, result: dict[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    if set(result) != {"merge_sets", "rationale"}:
        raise ValueError("merge response has unexpected fields")
    merge_sets = validate_merge_sets(result, validation["valid_ids"])
    if validation.get("require_multiple_groups"):
        merged = {item for group in merge_sets for item in group}
        if len(merge_sets) + len(set(validation["valid_ids"]) - merged) < 2:
            raise ValueError("binding split must retain at least two groups")
    for index, entry in enumerate(result["merge_sets"]):
        if set(entry) != {"member_ids", "label", "rationale"}:
            raise ValueError(f"merge_sets[{index}] has unexpected fields")
        if not all(isinstance(entry[field], str) and entry[field].strip()
                   for field in ("label", "rationale")):
            raise ValueError(f"merge_sets[{index}] needs label and rationale")
    if kind == "cross_source_merge" and len(result["merge_sets"]) > 1:
        raise ValueError("cross-source response may contain at most one merge set")
    if kind == "cross_source_merge" and result["merge_sets"]:
        members = result["merge_sets"][0]["member_ids"]
        if validation["anchor_bucket_id"] not in members:
            raise ValueError("cross-source merge must include the anchor bucket")
        if len({validation["source_by_id"][member] for member in members}) < 2:
            raise ValueError("cross-source merge must span at least two sources")
    return result


def _validate_model_response(
    kind: str, response: Any, validation: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise ValueError("response must be a JSON object")
    result = dict(response)
    if kind not in {"ranking", "weight_assignment"} and (
        not isinstance(result.get("rationale"), str) or not result["rationale"].strip()
    ):
        raise ValueError("response requires a non-empty rationale")
    if kind in {"select_column", "split_bucket", "readout_coherence", "retrieval_eligibility"}:
        return _validate_decision_response(kind, result, validation)
    if kind in {"merge_values", "merge_buckets", "readout_merge_values",
                "readout_merge_profiles", "cross_source_merge"}:
        return _validate_merge_response(kind, result, validation)
    if kind == "ranking":
        if set(result) != {"winner_bucket_id"}:
            raise ValueError("ranking response has unexpected fields")
        if result["winner_bucket_id"] not in validation["candidate_bucket_ids"]:
            raise ValueError("ranking winner was not supplied")
        return result
    if kind == "weight_assignment":
        return _validate_weight_response(result, validation)
    raise ValueError(f"unknown request kind: {kind}")


def _validate_weight_response(
    result: dict[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    if set(result) != {"scores"} or not isinstance(result["scores"], list):
        raise ValueError("weight response must contain only a scores list")
    expected = list(validation["candidate_aliases"])
    found = []
    for index, score in enumerate(result["scores"]):
        if not isinstance(score, Mapping) or set(score) != {"candidate", "weight", "rationale"}:
            raise ValueError(f"scores[{index}] has unexpected fields")
        candidate, weight, rationale = score["candidate"], score["weight"], score["rationale"]
        if candidate not in expected or candidate in found:
            raise ValueError(f"scores[{index}] has an unknown or repeated candidate")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"scores[{index}] weight must be numeric")
        if not 0 <= float(weight) <= 1 or abs(float(weight) * 100 - round(float(weight) * 100)) > 1e-8:
            raise ValueError(f"scores[{index}] weight must use hundredth increments")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"scores[{index}] requires a rationale")
        found.append(candidate)
    if found != expected:
        raise ValueError("weight response must cover candidates in displayed order")
    return result


def _run_pending(
    connection: sqlite3.Connection,
    request_ids: Sequence[str],
    *,
    parallelism: int,
    client: Any,
    request_slots: threading.BoundedSemaphore,
) -> None:
    unique_request_ids = list(dict.fromkeys(request_ids))
    lookup = {}
    for start in range(0, len(unique_request_ids), 500):
        batch = unique_request_ids[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        lookup.update(
            (row["request_id"], row)
            for row in connection.execute(
                f"SELECT * FROM requests WHERE request_id IN ({placeholders})",
                batch,
            )
        )
    rows = [lookup.get(request_id) for request_id in unique_request_ids]
    if any(row is None for row in rows):
        raise ValueError("requested an unknown cached request")
    pending = [dict(row) for row in rows if row["status"] != "complete"]
    if not pending:
        return

    def call(row: Mapping[str, Any]) -> dict[str, Any]:
        attempts = int(row["attempts"])
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        receipts = []
        last_error = None
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            started = time.time()
            raw_response = None
            attempted_model = None
            attempted_reasoning = None
            attempted_provider_name = None
            attempted_provider_base_url = None
            try:
                with request_slots:
                    validation = json.loads(row["validation_json"])
                    requested_model = validation.get("requested_model", MODEL)
                    request = {
                        "model": requested_model,
                        "messages": [
                            {"role": "system", "content": "Return valid JSON."},
                            {"role": "user", "content": row["prompt"]},
                        ],
                        "reasoning_effort": row["reasoning_effort"],
                        "response_format": {"type": "json_object"},
                    }
                    request[COMPLETION_TOKEN_PARAMETER] = int(row["max_tokens"])
                    if row["kind"] == "weight_assignment":
                        request["extra_body"] = dict(WEIGHT_REQUEST_EXTRA_BODY)
                        if validation.get("provider_routing"):
                            routing = dict(validation["provider_routing"])
                            if routing.pop("omit_response_format", False):
                                request.pop("response_format")
                            request["extra_body"]["provider"] = routing
                    completion = client.chat.completions.create(
                        **request
                    )
                usage = completion.usage
                message = completion.choices[0].message
                raw_response = message.content
                attempted_model = str(completion.model)
                attempted_provider_name = getattr(completion, "provider_name", None)
                attempted_provider_base_url = getattr(
                    completion, "provider_base_url", None
                )
                extras = getattr(message, "model_extra", None) or {}
                reasoning = (
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning", None)
                    or extras.get("reasoning_content")
                    or extras.get("reasoning")
                )
                attempted_reasoning = reasoning
                input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                parsed = json.loads(message.content or "")
                parsed = _validate_model_response(
                    str(row["kind"]), parsed, validation
                )
                allowed_models = validation.get(
                    "allowed_served_models", [requested_model, *SERVED_MODEL_ALIASES]
                )
                if str(completion.model) not in set(allowed_models):
                    raise ValueError(
                        f"served model changed: {completion.model!s} not in {allowed_models}"
                    )
                receipt = {
                    "attempt": attempts,
                    "elapsed_seconds": time.time() - started,
                    "generation_id": getattr(completion, "id", None),
                    "requested_model": requested_model,
                    "selected_provider_route": validation.get("selected_provider_route"),
                    "provider_pool_snapshot_sha256": validation.get(
                        "provider_pool_snapshot_sha256"
                    ),
                    "model": str(completion.model),
                    "provider_name": getattr(completion, "provider_name", None),
                    "provider_base_url": getattr(
                        completion, "provider_base_url", None
                    ),
                    "usage": usage.model_dump() if usage is not None else None,
                    "response": parsed,
                    "reasoning_content": reasoning,
                }
                receipts.append(receipt)
                return {
                    "request_id": row["request_id"],
                    "status": "complete",
                    "attempts": attempts,
                    "response_json": _canonical_json(parsed),
                    "reasoning_content": reasoning,
                    "served_model": str(completion.model),
                    "provider_name": getattr(completion, "provider_name", None),
                    "provider_base_url": getattr(
                        completion, "provider_base_url", None
                    ),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "error": None,
                    "receipts": receipts,
                }
            except Exception as error:  # retry transport and schema failures uniformly
                last_error = f"{type(error).__name__}: {error}"
                receipts.append(
                    {
                        "attempt": attempts,
                        "elapsed_seconds": time.time() - started,
                        "error": last_error,
                        "raw_response": raw_response,
                        "model": attempted_model,
                        "provider_name": attempted_provider_name,
                        "provider_base_url": attempted_provider_base_url,
                        "reasoning_content": attempted_reasoning,
                    }
                )
                if attempts < MAX_ATTEMPTS:
                    time.sleep(min(8, 2 ** (attempts - 1)))
        return {
            "request_id": row["request_id"],
            "status": "failed",
            "attempts": attempts,
            "response_json": None,
            "reasoning_content": None,
            "served_model": None,
            "provider_name": None,
            "provider_base_url": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": last_error,
            "receipts": receipts,
        }

    results = []
    with ThreadPoolExecutor(max_workers=min(parallelism, len(pending))) as executor:
        futures = {executor.submit(call, row): row["request_id"] for row in pending}
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                with _REQUEST_DATABASE_WRITE_LOCK:
                    for receipt in result.pop("receipts"):
                        connection.execute(
                            "INSERT OR REPLACE INTO request_attempt_receipts VALUES (?,?,?)",
                            (
                                result["request_id"],
                                receipt["attempt"],
                                _canonical_json(receipt),
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE requests
                        SET status=?,attempts=?,response_json=?,reasoning_content=?,served_model=?,
                            provider_name=?,provider_base_url=?,
                            input_tokens=?,output_tokens=?,error=?
                        WHERE request_id=?
                        """,
                        (
                            result["status"],
                            result["attempts"],
                            result["response_json"],
                            result["reasoning_content"],
                            result["served_model"],
                            result["provider_name"],
                            result["provider_base_url"],
                            result["input_tokens"],
                            result["output_tokens"],
                            result["error"],
                            result["request_id"],
                        ),
                    )
                    # Long reasoning calls finish unevenly across source workers. Do
                    # not keep a write transaction open while another call runs.
                    connection.commit()
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    connection.commit()
    failed = [result for result in results if result["status"] != "complete"]
    if failed:
        raise RuntimeError(
            "DeepSeek requests failed closed: "
            + ", ".join(f"{row['request_id']} ({row['error']})" for row in failed[:5])
        )


def _response(connection: sqlite3.Connection, request_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT status,response_json FROM requests WHERE request_id=?", (request_id,)
    ).fetchone()
    if row is None or row["status"] != "complete":
        raise ValueError(f"request is not complete: {request_id}")
    return json.loads(row["response_json"])


def _request_id(kind: str, phase: str, prompt: str) -> str:
    return _stable_id("req", kind, phase, hashlib.sha256(prompt.encode()).hexdigest())


def _queue_checked_request(
    connection: sqlite3.Connection,
    *,
    kind: str,
    phase: str,
    prompt: str,
    reasoning_effort: str,
    max_tokens: int,
    validation: Mapping[str, Any],
    maximum_context: int,
    commit: bool = True,
) -> str | None:
    if _token_count(prompt) + max_tokens > maximum_context:
        return None
    request_id = _request_id(kind, phase, prompt)
    return _queue_request(
        connection,
        request_id=request_id,
        kind=kind,
        phase=phase,
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        validation=validation,
        commit=commit,
    )


def _write_gzip_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    write_json_atomic(path, dict(state))


def _initialize_semantic_run(
    output: Path,
    *,
    review_manifest_path: Path,
    approved_review_sha256: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    actual_review_hash = _file_sha256(review_manifest_path)
    if actual_review_hash != approved_review_sha256:
        raise ValueError(
            f"approved review hash mismatch: {actual_review_hash} != {approved_review_sha256}"
        )
    review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review.get("status") != "awaiting_user_prompt_approval":
        raise ValueError("review manifest is not an approval candidate")
    if review.get("completion_requests_made") != 0:
        raise ValueError("review artifact unexpectedly contains completion calls")
    current_templates = {
        path.name: _file_sha256(path) for path in sorted(PROMPT_ROOT.glob("*.jinja"))
    }
    if review.get("prompt_template_hashes") != current_templates:
        raise ValueError("reviewed prompt templates changed")
    for field, path in (("source_record_map", RECORD_MAP), ("v10_records", RECORDS)):
        if review[field]["sha256"] != _file_sha256(path):
            raise ValueError(f"reviewed {field} input changed")

    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "semantic_state.json"
    atoms_path = output / "input_atoms.parquet"
    samples_path = output / "sample_cards.json.gz"
    manifest_path = output / "manifest.json"
    if state_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("approved_prompt_review_sha256") != approved_review_sha256:
            raise ValueError("existing run uses a different prompt review")
        return (
            pd.read_parquet(atoms_path),
            _read_gzip_json(samples_path),
            json.loads(state_path.read_text(encoding="utf-8")),
        )
    if any(output.iterdir()):
        raise FileExistsError(f"new semantic output directory is not empty: {output}")

    atoms, sample_cards = _load_atoms()
    atoms.to_parquet(atoms_path, index=False)
    _write_gzip_json(samples_path, sample_cards)
    groups = {}
    for (level, source), buckets in initial_buckets(atoms).items():
        key = f"{level}|{source}"
        groups[key] = {
            "level": level,
            "source_id": source,
            "round": 0,
            "consumed_columns": list(INITIAL_COLUMNS[source]),
            "active": buckets,
            "terminal": {},
            "termination_reasons": {},
            "history": [],
            "finished": False,
            "final_buckets": None,
        }
    state = {"version": VERSION, "status": "running_semantic", "groups": groups}
    _save_state(state_path, state)
    manifest = {
        "version": VERSION,
        "status": "running_semantic",
        "approved_prompt_review": str(review_manifest_path),
        "approved_prompt_review_sha256": approved_review_sha256,
        "model": MODEL,
        "base_url": BASE_URL,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "input_atoms": {"path": atoms_path.name, "rows": len(atoms)},
        "sample_cards": {"path": samples_path.name, "rows": len(sample_cards)},
        "requests": "requests.sqlite3",
    }
    write_json_atomic(manifest_path, manifest)
    return atoms, sample_cards, state


def _merge_bucket_round(
    connection: sqlite3.Connection,
    *,
    level: str,
    source: str,
    buckets: Mapping[str, Sequence[str]],
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
    phase: str,
    maximum_context: int,
    parallelism: int,
    client: Any,
    request_slots: threading.BoundedSemaphore,
) -> dict[str, list[str]]:
    if len(buckets) < 2:
        return {key: list(value) for key, value in buckets.items()}
    pending_batches = deterministic_batches(
        list(buckets), seed=f"{BATCH_SEED_VERSION}|{phase}"
    )
    sendable: list[tuple[list[str], str]] = []
    while pending_batches:
        batch = pending_batches.pop(0)
        if len(batch) < 2:
            sendable.append((batch, ""))
            continue
        batch_buckets = {key: buckets[key] for key in batch}
        prompt = _render(
            "merge_buckets",
            _merge_bucket_payload(level, source, batch_buckets, lookup, sample_cards),
        )
        if _token_count(prompt) + HIGH_MAX_TOKENS <= maximum_context:
            sendable.append((batch, prompt))
            continue
        if len(batch) == 2:
            sendable.extend((([item], "") for item in batch))
            continue
        midpoint = len(batch) // 2
        pending_batches[0:0] = [batch[:midpoint], batch[midpoint:]]

    request_ids = []
    with _REQUEST_DATABASE_WRITE_LOCK:
        try:
            for batch_index, (batch, prompt) in enumerate(sendable):
                if len(batch) < 2:
                    continue
                request_id = _queue_checked_request(
                    connection,
                    kind="merge_buckets",
                    phase=f"{phase}|batch{batch_index:04d}",
                    prompt=prompt,
                    reasoning_effort="high",
                    max_tokens=HIGH_MAX_TOKENS,
                    validation={"valid_ids": batch},
                    maximum_context=maximum_context,
                    commit=False,
                )
                if request_id is None:
                    raise AssertionError("prechecked merge prompt became oversized")
                request_ids.append(request_id)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    _run_pending(
        connection,
        request_ids,
        parallelism=parallelism,
        client=client,
        request_slots=request_slots,
    )

    output: dict[str, list[str]] = {}
    request_index = 0
    for batch, _ in sendable:
        batch_buckets = {key: buckets[key] for key in batch}
        if len(batch) < 2:
            merged = batch_buckets
        else:
            merged = apply_merge_sets(
                batch_buckets, _response(connection, request_ids[request_index])
            )
            request_index += 1
        overlap = set(output) & set(merged)
        if overlap:
            raise ValueError(f"merge round produced duplicate bucket IDs: {sorted(overlap)}")
        output.update(merged)
    return output


def _run_source_semantic(
    connection: sqlite3.Connection,
    group: dict[str, Any],
    *,
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
    maximum_context: int,
    parallelism: int,
    client: Any,
    request_slots: threading.BoundedSemaphore,
    checkpoint: Callable[[], None],
) -> None:
    level = str(group["level"])
    source = str(group["source_id"])
    while not group["finished"] and group["round"] < MAX_SOURCE_ROUNDS:
        active = {key: list(value) for key, value in group["active"].items()}
        if not active:
            group["finished"] = True
            break
        remaining = [
            column
            for column in REFINEMENT_COLUMNS[source]
            if column not in set(group["consumed_columns"])
        ]
        if not remaining:
            for bucket, members in active.items():
                group["terminal"][bucket] = members
                group["termination_reasons"][bucket] = "columns_exhausted"
            group["active"] = {}
            group["finished"] = True
            break

        round_number = int(group["round"]) + 1
        selector_payload = _column_selection_payload(
            level,
            source,
            active,
            lookup,
            group["consumed_columns"],
        )
        selector_prompt = _render("select_column", selector_payload)
        selector_id = _queue_checked_request(
            connection,
            kind="select_column",
            phase=f"semantic|{level}|{source}|round{round_number}|select",
            prompt=selector_prompt,
            reasoning_effort="high",
            max_tokens=LOW_MAX_TOKENS,
            validation={"candidate_columns": remaining},
            maximum_context=maximum_context,
        )
        if selector_id is None:
            raise ValueError(f"source column selector exceeds model context: {level} {source}")
        _run_pending(
            connection,
            [selector_id],
            parallelism=1,
            client=client,
            request_slots=request_slots,
        )
        selector = _response(connection, selector_id)
        if selector["decision"] == "stop":
            for bucket, members in active.items():
                group["terminal"][bucket] = members
                group["termination_reasons"][bucket] = "source_selector_stop"
            group["active"] = {}
            group["finished"] = True
            group["history"].append(
                {"round": round_number, "selector_request_id": selector_id, "decision": "stop"}
            )
            checkpoint()
            break

        column = str(selector["column"])
        context_columns = [*group["consumed_columns"], column]
        group["consumed_columns"].append(column)
        decisions: dict[str, str] = {}
        value_items: dict[str, dict[str, list[str]]] = {}
        split_request_ids: dict[str, str] = {}
        merge_prompts: dict[str, str] = {}
        technical_reasons: dict[str, str] = {}
        for bucket, members in sorted(active.items()):
            payload, items = _value_payload(
                bucket,
                members,
                column,
                lookup,
                context_columns=context_columns,
            )
            payload.update({"task": TASK_NAME, "level": level, "source_id": source})
            if len(items) < 2:
                decisions[bucket] = (
                    "defer" if DEFER_TECHNICAL_BRANCHES else "keep"
                )
                technical_reasons[bucket] = "selected_column_has_one_value"
                continue
            split_prompt = _render("split_bucket", payload)
            merge_prompt = _render("merge_values", payload)
            if (
                _token_count(split_prompt) + LOW_MAX_TOKENS > maximum_context
                or _token_count(merge_prompt) + HIGH_MAX_TOKENS > maximum_context
            ):
                decisions[bucket] = (
                    "defer" if DEFER_TECHNICAL_BRANCHES else "keep"
                )
                technical_reasons[bucket] = "full_value_prompt_exceeds_context"
                continue
            request_id = _queue_checked_request(
                connection,
                kind="split_bucket",
                phase=f"semantic|{level}|{source}|round{round_number}|split|{bucket}",
                prompt=split_prompt,
                reasoning_effort="high",
                max_tokens=LOW_MAX_TOKENS,
                validation={},
                maximum_context=maximum_context,
            )
            if request_id is None:
                raise AssertionError("prechecked split prompt became oversized")
            split_request_ids[bucket] = request_id
            value_items[bucket] = items
            merge_prompts[bucket] = merge_prompt

        _run_pending(
            connection,
            list(split_request_ids.values()),
            parallelism=parallelism,
            client=client,
            request_slots=request_slots,
        )
        merge_request_ids: dict[str, str] = {}
        for bucket, request_id in split_request_ids.items():
            decision = str(_response(connection, request_id)["decision"])
            decisions[bucket] = decision
            if decision != "split":
                continue
            merge_id = _queue_checked_request(
                connection,
                kind="merge_values",
                phase=f"semantic|{level}|{source}|round{round_number}|values|{bucket}",
                prompt=merge_prompts[bucket],
                reasoning_effort="high",
                max_tokens=HIGH_MAX_TOKENS,
                validation={
                    "valid_ids": sorted(value_items[bucket]),
                    "require_multiple_groups": True,
                },
                maximum_context=maximum_context,
            )
            if merge_id is None:
                raise AssertionError("prechecked value-merge prompt became oversized")
            merge_request_ids[bucket] = merge_id
        _run_pending(
            connection,
            list(merge_request_ids.values()),
            parallelism=parallelism,
            client=client,
            request_slots=request_slots,
        )

        children = {
            bucket: apply_merge_sets(
                value_items[bucket], _response(connection, request_id)
            )
            for bucket, request_id in merge_request_ids.items()
        }
        reverse_collapsed_splits(decisions, children, technical_reasons)
        next_active, newly_terminal = apply_split_decisions(active, decisions, children)
        deferred = {
            bucket: list(active[bucket])
            for bucket, decision in decisions.items()
            if decision == "defer"
        }
        new_children = {
            bucket: members
            for bucket, members in next_active.items()
            if bucket not in deferred
        }
        group["terminal"].update(newly_terminal)
        for bucket in newly_terminal:
            group["termination_reasons"][bucket] = technical_reasons.get(
                bucket, "deepseek_keep"
            )

        split_count = sum(decision == "split" for decision in decisions.values())
        if new_children:
            for merge_round in (1, 2):
                new_children = _merge_bucket_round(
                    connection,
                    level=level,
                    source=source,
                    buckets=new_children,
                    lookup=lookup,
                    sample_cards=sample_cards,
                    phase=(
                        f"semantic|{level}|{source}|round{round_number}|"
                        f"new_children_merge{merge_round}"
                    ),
                    maximum_context=maximum_context,
                    parallelism=parallelism,
                    client=client,
                    request_slots=request_slots,
                )
        next_active = {**new_children, **deferred}
        group["round"] = round_number
        group["active"] = next_active
        group["history"].append(
            {
                "round": round_number,
                "column": column,
                "selector_request_id": selector_id,
                "input_bucket_count": len(active),
                "split_count": split_count,
                "keep_count": sum(
                    decision == "keep" for decision in decisions.values()
                ),
                "deferred_count": len(deferred),
                "output_active_bucket_count": len(next_active),
                "technical_outcome_reasons": dict(Counter(technical_reasons.values())),
            }
        )
        if split_count == 0 and not deferred:
            group["finished"] = True
        checkpoint()

    if group["active"]:
        reason = "three_source_round_cap" if group["round"] >= MAX_SOURCE_ROUNDS else "no_active_split"
        if reason == "three_source_round_cap" and DEFER_TECHNICAL_BRANCHES:
            group["blocked_reason"] = "three_source_round_cap_with_active_branches"
            checkpoint()
            raise RuntimeError(
                f"semantic {level}|{source} reached the three-round cap with "
                f"{len(group['active'])} active branches"
            )
        for bucket, members in group["active"].items():
            group["terminal"][bucket] = members
            group["termination_reasons"][bucket] = reason
        group["active"] = {}
    leaves = {key: list(value) for key, value in group["terminal"].items()}
    for merge_round in (1, 2):
        leaves = _merge_bucket_round(
            connection,
            level=level,
            source=source,
            buckets=leaves,
            lookup=lookup,
            sample_cards=sample_cards,
            phase=f"semantic|{level}|{source}|final_all_leaves_merge{merge_round}",
            maximum_context=maximum_context,
            parallelism=parallelism,
            client=client,
            request_slots=request_slots,
        )
    group["final_buckets"] = leaves
    group["finished"] = True
    checkpoint()


def run_semantic(
    output: Path,
    *,
    review_manifest_path: Path,
    approved_review_sha256: str,
    parallelism: int,
    seed_request_cache: Path | None = None,
) -> dict[str, Any]:
    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    maximum_context, endpoint_models = _endpoint_contract()
    atoms, sample_cards, state = _initialize_semantic_run(
        output,
        review_manifest_path=review_manifest_path,
        approved_review_sha256=approved_review_sha256,
    )
    lookup = _atom_lookup(atoms)
    state_path = output / "semantic_state.json"
    database_path = output / "requests.sqlite3"
    seed_receipt = _seed_request_database(database_path, seed_request_cache)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if seed_receipt is not None:
        manifest["seed_request_cache"] = seed_receipt
        write_json_atomic(manifest_path, manifest)
    else:
        seed_receipt = manifest.get("seed_request_cache")
    connection = _request_database(database_path)
    connection.close()
    client, endpoint_receipts, total_parallelism = _build_completion_pool(parallelism)
    request_slots = threading.BoundedSemaphore(total_parallelism)
    state_lock = threading.Lock()

    def run_group(group_key: str) -> None:
        group = deepcopy(state["groups"][group_key])
        connection = _request_database(database_path)

        def checkpoint() -> None:
            with state_lock:
                state["groups"][group_key] = deepcopy(group)
                _save_state(state_path, state)

        try:
            print(
                f"semantic {group_key}: round={group['round']} active={len(group['active'])}",
                flush=True,
            )
            _run_source_semantic(
                connection,
                group,
                lookup=lookup,
                sample_cards=sample_cards,
                maximum_context=maximum_context,
                parallelism=total_parallelism,
                client=client,
                request_slots=request_slots,
                checkpoint=checkpoint,
            )
            print(
                f"semantic {group_key}: complete buckets={len(group['final_buckets'])}",
                flush=True,
            )
        finally:
            connection.close()

    pending_groups = [
        key
        for key in sorted(state["groups"])
        if state["groups"][key].get("final_buckets") is None
    ]
    with ThreadPoolExecutor(max_workers=len(pending_groups) or 1) as executor:
        futures = [executor.submit(run_group, key) for key in pending_groups]
        for future in as_completed(futures):
            future.result()

    rows = []
    for group in state["groups"].values():
        for source_bucket, atom_ids in group["final_buckets"].items():
            for atom_id in atom_ids:
                rows.append(
                    {
                        "level": group["level"],
                        "source_id": group["source_id"],
                        "source_semantic_bucket_id": source_bucket,
                        "atom_id": atom_id,
                    }
                )
    mapping = pd.DataFrame(rows).sort_values(["level", "source_id", "source_semantic_bucket_id", "atom_id"])
    if len(mapping) != len(atoms) or mapping["atom_id"].duplicated().any():
        raise ValueError("final source-local mapping does not cover every atom exactly once")
    mapping.to_parquet(output / "source_semantic_bucket_map.parquet", index=False)
    state["status"] = SOURCE_LOCAL_FINAL_STATUS
    with state_lock:
        _save_state(state_path, state)

    connection = _request_database(output / "requests.sqlite3")
    request_summary = connection.execute(
        """
        SELECT count(*) AS requests,
               sum(status='complete') AS complete,
               sum(status='failed') AS failed,
               coalesce(sum(input_tokens),0) AS input_tokens,
               coalesce(sum(output_tokens),0) AS output_tokens
        FROM requests
        """
    ).fetchone()
    connection.close()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": SOURCE_LOCAL_FINAL_STATUS,
            "endpoint_models_at_start": endpoint_models,
            "endpoint_pool_at_start": endpoint_receipts,
            "endpoint_pool_final_snapshot": client.snapshot(),
            "per_endpoint_parallelism": parallelism,
            "total_parallelism": total_parallelism,
            "max_model_len": maximum_context,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "seed_request_cache": seed_receipt,
            "source_semantic_bucket_map": {
                "path": "source_semantic_bucket_map.parquet",
                "rows": len(mapping),
                "source_bucket_count": int(mapping["source_semantic_bucket_id"].nunique()),
                "sha256": _file_sha256(output / "source_semantic_bucket_map.parquet"),
            },
            "request_summary": dict(request_summary),
        }
    )
    write_json_atomic(manifest_path, manifest)
    return manifest


def _cross_source_inputs(
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {
        "awaiting_cross_source_mapping",
        "awaiting_agentic_cross_source_review",
    }:
        raise ValueError("source-local semantic run is not complete")
    source_path = output / "source_semantic_bucket_map.parquet"
    expected_hash = manifest["source_semantic_bucket_map"]["sha256"]
    if _file_sha256(source_path) != expected_hash:
        raise ValueError("source-local semantic map changed after publication")
    source = pd.read_parquet(source_path)
    atoms = pd.read_parquet(output / "input_atoms.parquet")
    if source["atom_id"].duplicated().any() or set(source["atom_id"]) != set(atoms["atom_id"]):
        raise ValueError("source-local semantic map does not cover each atom exactly once")
    sample_cards = _read_gzip_json(output / "sample_cards.json.gz")
    batches = cross_source_candidate_batches(source, atoms, sample_cards)
    return source, atoms, sample_cards, batches


def _cross_source_schedule_sha256(batches: Sequence[Mapping[str, Any]]) -> str:
    schedule = [
        {
            "level": batch["level"],
            "anchor_bucket_id": batch["anchor_bucket_id"],
            "candidate_bucket_ids": [
                bucket["bucket_id"] for bucket in batch["buckets"][1:]
            ],
        }
        for batch in batches
    ]
    return hashlib.sha256(_canonical_json(schedule).encode()).hexdigest()


def prepare_cross_source_prompt_review(
    output: Path, *, review_output: Path
) -> dict[str, Any]:
    """Render worst-size real prompts without making completion requests."""

    if review_output.exists() and any(review_output.iterdir()):
        raise FileExistsError(f"prompt review directory is not empty: {review_output}")
    review_output.mkdir(parents=True, exist_ok=True)
    source, _atoms, _sample_cards, batches = _cross_source_inputs(output)
    if not batches:
        raise ValueError("no cross-source candidate batches were generated")
    maximum_context, _endpoint_models = _endpoint_contract()
    prompt_rows = []
    for level in CROSS_SOURCE_LEVELS:
        candidates = [batch for batch in batches if batch["level"] == level]
        if not candidates:
            continue
        batch = max(candidates, key=lambda item: len(_canonical_json(item)))
        prompt = _render_cross_source(batch)
        path = review_output / f"cross_source_{level}.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        input_tokens = _token_count(prompt)
        prompt_rows.append(
            {
                "level": level,
                "anchor_bucket_id": batch["anchor_bucket_id"],
                "path": path.name,
                "sha256": _file_sha256(path),
                "input_tokens": input_tokens,
                "completion_token_reserve": HIGH_MAX_TOKENS,
                "fits_model_context": input_tokens + HIGH_MAX_TOKENS <= maximum_context,
            }
        )
    source_counts = (
        source[source["level"].isin(CROSS_SOURCE_LEVELS)]
        .groupby(["level", "source_id"])["source_semantic_bucket_id"]
        .nunique()
    )
    manifest = {
        "version": f"{VERSION}.cross_source_prompt_review.v1",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "model": MODEL,
        "base_url": BASE_URL,
        "endpoint_models": _endpoint_models,
        "max_model_len": maximum_context,
        "source_semantic_bucket_map": {
            "path": str(output / "source_semantic_bucket_map.parquet"),
            "sha256": _file_sha256(output / "source_semantic_bucket_map.parquet"),
        },
        "template": {
            "path": str(CROSS_SOURCE_PROMPT_ROOT / "merge_buckets.jinja"),
            "sha256": _file_sha256(CROSS_SOURCE_PROMPT_ROOT / "merge_buckets.jinja"),
        },
        "settings": {
            "reasoning_effort": "high",
            "batch_size": CROSS_SOURCE_BATCH_SIZE,
            "exploration_per_source": CROSS_SOURCE_EXPLORATION_PER_SOURCE,
            "batch_seed_version": CROSS_SOURCE_BATCH_SEED_VERSION,
            "sample_limit": SAMPLE_LIMIT,
            "candidate_routing_is_not_a_merge_decision": True,
        },
        "candidate_schedule": {
            "batch_count": len(batches),
            "sha256": _cross_source_schedule_sha256(batches),
            "source_bucket_counts": {
                f"{level}|{source_id}": int(count)
                for (level, source_id), count in source_counts.items()
            },
        },
        "representative_prompts": prompt_rows,
        "non_sendable_prompts": [
            row["path"] for row in prompt_rows if not row["fits_model_context"]
        ],
    }
    write_json_atomic(review_output / "manifest.json", manifest)
    return manifest


def _cross_source_components(
    source: pd.DataFrame, decisions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    parent = {
        bucket: bucket for bucket in source["source_semantic_bucket_id"].unique()
    }

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for decision in decisions:
        merge_sets = decision["response"]["merge_sets"]
        if not merge_sets:
            continue
        members = merge_sets[0]["member_ids"]
        for member in members[1:]:
            union(members[0], member)
    groups: dict[str, list[str]] = {}
    for bucket in parent:
        groups.setdefault(find(bucket), []).append(bucket)
    source_by_bucket = source.groupby("source_semantic_bucket_id")["source_id"].first().to_dict()
    level_by_bucket = source.groupby("source_semantic_bucket_id")["level"].first().to_dict()
    components = []
    for members in sorted(groups.values(), key=lambda group: (len(group), group)):
        if len(members) < 2:
            continue
        member_set = set(members)
        evidence = [
            decision
            for decision in decisions
            if decision["response"]["merge_sets"]
            and member_set.intersection(
                decision["response"]["merge_sets"][0]["member_ids"]
            )
        ]
        components.append(
            {
                "level": str(level_by_bucket[members[0]]),
                "member_ids": sorted(members),
                "source_ids": sorted({str(source_by_bucket[member]) for member in members}),
                "decision_request_ids": sorted(
                    decision["request_id"] for decision in evidence
                ),
                "proposed_labels": sorted(
                    {
                        decision["response"]["merge_sets"][0]["label"]
                        for decision in evidence
                    }
                ),
                "proposed_rationales": sorted(
                    {
                        decision["response"]["merge_sets"][0]["rationale"]
                        for decision in evidence
                    }
                ),
            }
        )
    return components


def run_cross_source_merges(
    output: Path,
    *,
    review_manifest_path: Path,
    approved_review_sha256: str,
    parallelism: int,
) -> dict[str, Any]:
    """Run DeepSeek cross-source proposals and stop for agentic review."""

    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    actual_review_hash = _file_sha256(review_manifest_path)
    if actual_review_hash != approved_review_sha256:
        raise ValueError("approved cross-source prompt review hash mismatch")
    review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review.get("status") != "awaiting_user_prompt_approval":
        raise ValueError("cross-source review manifest is not an approval candidate")
    source, _atoms, _sample_cards, batches = _cross_source_inputs(output)
    if review["source_semantic_bucket_map"]["sha256"] != _file_sha256(
        output / "source_semantic_bucket_map.parquet"
    ):
        raise ValueError("approved cross-source source map changed")
    template_path = CROSS_SOURCE_PROMPT_ROOT / "merge_buckets.jinja"
    if review["template"]["sha256"] != _file_sha256(template_path):
        raise ValueError("approved cross-source prompt template changed")
    if review["candidate_schedule"]["sha256"] != _cross_source_schedule_sha256(batches):
        raise ValueError("approved cross-source candidate schedule changed")
    maximum_context, endpoint_models = _endpoint_contract()
    client, endpoint_receipts, total_parallelism = _build_completion_pool(parallelism)
    request_slots = threading.BoundedSemaphore(total_parallelism)
    database_path = output / "cross_source_requests.sqlite3"
    connection = _request_database(database_path)
    request_batches = {}
    request_ids = []
    for batch in batches:
        prompt = _render_cross_source(batch)
        source_by_id = {
            bucket["bucket_id"]: bucket["source_id"] for bucket in batch["buckets"]
        }
        request_id = _queue_checked_request(
            connection,
            kind="cross_source_merge",
            phase=f"cross_source|{batch['level']}|{batch['anchor_bucket_id']}",
            prompt=prompt,
            reasoning_effort="high",
            max_tokens=HIGH_MAX_TOKENS,
            validation={
                "valid_ids": list(source_by_id),
                "anchor_bucket_id": batch["anchor_bucket_id"],
                "source_by_id": source_by_id,
            },
            maximum_context=maximum_context,
            commit=False,
        )
        if request_id is None:
            raise ValueError(
                f"cross-source prompt exceeds model context: {batch['anchor_bucket_id']}"
            )
        request_ids.append(request_id)
        request_batches[request_id] = batch
    connection.commit()
    _run_pending(
        connection,
        request_ids,
        parallelism=total_parallelism,
        client=client,
        request_slots=request_slots,
    )
    decisions = []
    for request_id in request_ids:
        batch = request_batches[request_id]
        response = _response(connection, request_id)
        validate_cross_source_merge_response(response, batch)
        decisions.append(
            {
                "request_id": request_id,
                "level": batch["level"],
                "anchor_bucket_id": batch["anchor_bucket_id"],
                "candidate_bucket_ids": [
                    bucket["bucket_id"] for bucket in batch["buckets"][1:]
                ],
                "response": response,
            }
        )
    request_summary = dict(
        connection.execute(
            """
            SELECT count(*) AS requests,
                   sum(status='complete') AS complete,
                   sum(status='failed') AS failed,
                   coalesce(sum(input_tokens),0) AS input_tokens,
                   coalesce(sum(output_tokens),0) AS output_tokens
            FROM requests
            """
        ).fetchone()
    )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    proposal = {
        "version": f"{VERSION}.cross_source_model_proposals.v1",
        "status": "awaiting_agentic_review",
        "model": MODEL,
        "approved_prompt_review_sha256": approved_review_sha256,
        "source_semantic_bucket_map_sha256": _file_sha256(
            output / "source_semantic_bucket_map.parquet"
        ),
        "candidate_schedule_sha256": _cross_source_schedule_sha256(batches),
        "decisions": decisions,
        "components": _cross_source_components(source, decisions),
    }
    proposal_path = output / "cross_source_model_proposals.json"
    write_json_atomic(proposal_path, proposal)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "awaiting_agentic_cross_source_review",
            "cross_source_prompt_review": str(review_manifest_path),
            "cross_source_prompt_review_sha256": approved_review_sha256,
            "cross_source_endpoint_models_at_start": endpoint_models,
            "cross_source_endpoint_pool_at_start": endpoint_receipts,
            "cross_source_endpoint_pool_final_snapshot": client.snapshot(),
            "cross_source_per_endpoint_parallelism": parallelism,
            "cross_source_total_parallelism": total_parallelism,
            "cross_source_request_summary": request_summary,
            "cross_source_requests": {
                "path": database_path.name,
                "sha256": _file_sha256(database_path),
            },
            "cross_source_model_proposals": {
                "path": proposal_path.name,
                "sha256": _file_sha256(proposal_path),
                "component_count": len(proposal["components"]),
            },
        }
    )
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def apply_cross_source_review(
    source: pd.DataFrame, atoms: pd.DataFrame, merge_sets: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    if source["atom_id"].duplicated().any() or set(source["atom_id"]) != set(atoms["atom_id"]):
        raise ValueError("source-local semantic map does not cover every atom exactly once")
    reviewable = source[source["level"].isin(CROSS_SOURCE_LEVELS)]
    items = {
        bucket: sorted(rows["atom_id"].tolist())
        for bucket, rows in reviewable.groupby("source_semantic_bucket_id", sort=True)
    }
    sources = reviewable.groupby("source_semantic_bucket_id")["source_id"].first().to_dict()
    levels = reviewable.groupby("source_semantic_bucket_id")["level"].first().to_dict()
    for index, entry in enumerate(merge_sets):
        if set(entry) != {"member_ids", "label", "rationale"}:
            raise ValueError(f"cross-source merge_sets[{index}] has unexpected fields")
        if not all(isinstance(entry[field], str) and entry[field].strip() for field in ("label", "rationale")):
            raise ValueError(f"cross-source merge_sets[{index}] requires label and rationale")
        member_sources = {sources.get(member) for member in entry["member_ids"]}
        if None in member_sources or len(member_sources) < 2:
            raise ValueError(f"cross-source merge_sets[{index}] does not cross sources")
        member_levels = {levels[member] for member in entry["member_ids"]}
        if len(member_levels) != 1:
            raise ValueError(f"cross-source merge_sets[{index}] crosses levels")
    merged = apply_merge_sets(items, {"merge_sets": merge_sets})
    atom_to_final = {
        atom: final_bucket
        for final_bucket, members in merged.items()
        for atom in members
    }
    for row in source[~source["level"].isin(CROSS_SOURCE_LEVELS)].itertuples(index=False):
        atom_to_final[row.atom_id] = row.source_semantic_bucket_id
    if set(atom_to_final) != set(atoms["atom_id"]):
        raise ValueError("final semantic mapping changed atom coverage")
    return atom_to_final


def publish_final_map(output: Path, *, review_path: Path) -> dict[str, Any]:
    """Apply the agent-reviewed cross-source merges and publish full coverage."""
    source_path = output / "source_semantic_bucket_map.parquet"
    source = pd.read_parquet(source_path)
    atoms = pd.read_parquet(output / "input_atoms.parquet")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("version") != f"{VERSION}.cross_source_review.v1":
        raise ValueError("unexpected cross-source review version")
    merge_sets = review.get("merge_sets")
    if not isinstance(merge_sets, list):
        raise ValueError("cross-source review merge_sets must be a list")
    atom_to_final = apply_cross_source_review(source, atoms, merge_sets)

    final = source.copy()
    final["semantic_bucket_id"] = final["atom_id"].map(atom_to_final)
    final = final[
        [
            "level",
            "source_id",
            "source_semantic_bucket_id",
            "semantic_bucket_id",
            "atom_id",
        ]
    ].sort_values(["level", "semantic_bucket_id", "source_id", "atom_id"])
    final_path = output / "semantic_bucket_map.parquet"
    final.to_parquet(final_path, index=False)

    records = pd.read_parquet(RECORD_MAP, columns=["level", "source_id", "pair_bucket_key"])
    record_atoms = {
        _stable_id("atom", level, source_id, pair_bucket)
        for level, source_id, pair_bucket in records.itertuples(index=False, name=None)
        if level in LEVELS
    }
    if (
        len(records[records["level"].isin(LEVELS)]) != EXPECTED_RECORD_COUNT
        or record_atoms != set(final["atom_id"])
    ):
        raise ValueError("final semantic map does not cover every level-scoped record")
    manifest = {
        "version": f"{VERSION}.final_semantic_map.v1",
        "status": (
            "awaiting_size_review"
            if SEMANTIC_SIZE_REVIEW_REQUIRED
            else "complete_reviewed"
        ),
        "reviewer": review.get("reviewer"),
        "review_sha256": _file_sha256(review_path),
        "source_semantic_bucket_map_sha256": _file_sha256(source_path),
        "semantic_bucket_map_sha256": _file_sha256(final_path),
        "atom_count": len(final),
        "record_count": EXPECTED_RECORD_COUNT,
        "source_bucket_count": int(final["source_semantic_bucket_id"].nunique()),
        "semantic_bucket_count": int(final["semantic_bucket_id"].nunique()),
        "cross_source_merge_count": len(merge_sets),
        "levels": {
            level: {
                "atoms": len(rows),
                "semantic_buckets": int(rows["semantic_bucket_id"].nunique()),
            }
            for level, rows in final.groupby("level", sort=True)
        },
    }
    write_json_atomic(output / "semantic_bucket_map_manifest.json", manifest)
    return manifest


def audit_semantic_bucket_sizes(
    output: Path, *, fraction_threshold: float = 0.10, top_per_level: int = 10
) -> dict[str, Any]:
    """Prepare a deterministic pre-ranking review of unusually large buckets."""
    if not 0 < fraction_threshold < 1 or top_per_level < 1:
        raise ValueError("invalid semantic size-audit settings")
    final_path = output / "semantic_bucket_map.parquet"
    atoms_path = output / "input_atoms.parquet"
    final = pd.read_parquet(final_path)
    atoms = pd.read_parquet(atoms_path)
    joined = final.merge(
        atoms[["atom_id", "record_count", "values_json"]],
        on="atom_id",
        how="left",
        validate="one_to_one",
    )
    if joined["record_count"].isna().any():
        raise ValueError("semantic size audit could not resolve every atom")

    rows = []
    for (level, semantic_bucket), members in joined.groupby(
        ["level", "semantic_bucket_id"], sort=True
    ):
        rows.append(
            {
                "level": str(level),
                "semantic_bucket_id": str(semantic_bucket),
                "record_count": int(members["record_count"].sum()),
                "pair_bucket_count": len(members),
                "source_ids_json": _canonical_json(sorted(set(members["source_id"]))),
            }
        )
    audit = pd.DataFrame(rows)
    level_records = audit.groupby("level")["record_count"].transform("sum")
    level_atoms = audit.groupby("level")["pair_bucket_count"].transform("sum")
    audit["level_record_fraction"] = audit["record_count"] / level_records
    audit["level_pair_bucket_fraction"] = audit["pair_bucket_count"] / level_atoms
    audit["record_size_rank"] = audit.groupby("level")["record_count"].rank(
        method="first", ascending=False
    ).astype(int)
    audit["pair_bucket_size_rank"] = audit.groupby("level")["pair_bucket_count"].rank(
        method="first", ascending=False
    ).astype(int)
    audit["over_fraction_threshold"] = (
        (audit["level_record_fraction"] > fraction_threshold)
        | (audit["level_pair_bucket_fraction"] > fraction_threshold)
    )
    audit["top_size_bucket"] = (
        (audit["record_size_rank"] <= top_per_level)
        | (audit["pair_bucket_size_rank"] <= top_per_level)
    )
    audit["review_required"] = (
        audit["over_fraction_threshold"] | audit["top_size_bucket"]
    )

    flagged_ids = set(audit.loc[audit["review_required"], "semantic_bucket_id"])
    sample_cards = _read_gzip_json(output / "sample_cards.json.gz")
    state = json.loads((output / "semantic_state.json").read_text(encoding="utf-8"))
    atom_reason = {}
    for group in state["groups"].values():
        for terminal_bucket, atom_ids in group["terminal"].items():
            reason = group["termination_reasons"][terminal_bucket]
            for atom_id in atom_ids:
                if atom_id in atom_reason:
                    raise ValueError("semantic terminal lineage repeats an atom")
                atom_reason[atom_id] = reason
    if set(atom_reason) != set(atoms["atom_id"]):
        raise ValueError("semantic terminal lineage does not cover every atom")

    details = {}
    for semantic_bucket, members in joined[
        joined["semantic_bucket_id"].isin(flagged_ids)
    ].groupby("semantic_bucket_id", sort=True):
        varying = {}
        for source, source_members in members.groupby("source_id", sort=True):
            parsed = [json.loads(value) for value in source_members["values_json"]]
            columns = {}
            for column in REFINEMENT_COLUMNS[str(source)]:
                values = sorted({str(row[column]) for row in parsed})
                if len(values) > 1:
                    columns[column] = {
                        "distinct_values": len(values),
                        "examples": values[:5],
                    }
            varying[str(source)] = columns
        reasons = Counter(atom_reason.get(atom, "unknown") for atom in members["atom_id"])
        sample_ids = sorted(members["atom_id"])[:SAMPLE_LIMIT]
        details[str(semantic_bucket)] = {
            "varying_refinement_columns": varying,
            "termination_reasons": dict(sorted(reasons.items())),
            "sample_records": [sample_cards[atom] for atom in sample_ids],
        }
    audit["review_details_json"] = audit["semantic_bucket_id"].map(
        lambda bucket: _canonical_json(details.get(bucket, {}))
    )
    audit = audit.sort_values(
        ["level", "review_required", "record_count", "pair_bucket_count"],
        ascending=[True, False, False, False],
    )
    table_path = output / "semantic_bucket_size_audit.parquet"
    audit.to_parquet(table_path, index=False)

    flagged = audit[audit["review_required"]]
    report = [
        f"# {TASK_NAME} semantic bucket size review",
        "",
        "Degree-25 ranking and readout construction are blocked until every flagged bucket is reviewed as coherent or split further.",
        "",
        "| Level | Bucket | Records | Level records | Pair buckets | Level pair buckets | Threshold |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for row in flagged.itertuples(index=False):
        report.append(
            f"| {row.level} | `{row.semantic_bucket_id}` | {row.record_count:,} | "
            f"{row.level_record_fraction:.1%} | {row.pair_bucket_count:,} | "
            f"{row.level_pair_bucket_fraction:.1%} | "
            f"{'yes' if row.over_fraction_threshold else 'top-10'} |"
        )
    report_path = output / "semantic_bucket_size_review.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "version": f"{VERSION}.semantic_bucket_size_audit.v1",
        "status": "awaiting_scientific_review",
        "semantic_bucket_map_sha256": _file_sha256(final_path),
        "fraction_threshold": fraction_threshold,
        "top_per_level": top_per_level,
        "semantic_bucket_count": len(audit),
        "flagged_bucket_count": len(flagged),
        "over_fraction_threshold_count": int(audit["over_fraction_threshold"].sum()),
        "artifacts": {
            "table": {"path": table_path.name, "sha256": _file_sha256(table_path)},
            "report": {"path": report_path.name, "sha256": _file_sha256(report_path)},
        },
    }
    write_json_atomic(output / "semantic_bucket_size_audit_manifest.json", manifest)
    return manifest


def approve_semantic_bucket_sizes(
    output: Path, *, review_path: Path
) -> dict[str, Any]:
    """Record the scientific size review and unlock only a coherent final map."""
    final_path = output / "semantic_bucket_map.parquet"
    final_manifest_path = output / "semantic_bucket_map_manifest.json"
    audit_path = output / "semantic_bucket_size_audit.parquet"
    audit_manifest_path = output / "semantic_bucket_size_audit_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    map_sha256 = _file_sha256(final_path)
    if final_manifest.get("semantic_bucket_map_sha256") != map_sha256:
        raise ValueError("final semantic map hash does not match its manifest")
    if audit_manifest.get("semantic_bucket_map_sha256") != map_sha256:
        raise ValueError("size audit does not match the final semantic map")
    if audit_manifest["artifacts"]["table"]["sha256"] != _file_sha256(audit_path):
        raise ValueError("semantic size-audit table changed after publication")

    review = json.loads(review_path.read_text(encoding="utf-8"))
    expected_fields = {
        "version",
        "reviewer",
        "semantic_bucket_map_sha256",
        "bucket_reviews",
    }
    if set(review) != expected_fields:
        raise ValueError("semantic size review has unexpected fields")
    if review["version"] != f"{VERSION}.semantic_bucket_size_review.v1":
        raise ValueError("unexpected semantic size-review version")
    if review["semantic_bucket_map_sha256"] != map_sha256:
        raise ValueError("semantic size review targets a different final map")
    if not isinstance(review["reviewer"], str) or not review["reviewer"].strip():
        raise ValueError("semantic size review requires a reviewer")
    if not isinstance(review["bucket_reviews"], list):
        raise ValueError("semantic size review bucket_reviews must be a list")

    audit = pd.read_parquet(audit_path)
    expected = {
        (str(row.level), str(row.semantic_bucket_id))
        for row in audit[audit["review_required"]].itertuples(index=False)
    }
    decisions: dict[tuple[str, str], str] = {}
    for index, entry in enumerate(review["bucket_reviews"]):
        if set(entry) != {"level", "semantic_bucket_id", "decision", "rationale"}:
            raise ValueError(f"bucket_reviews[{index}] has unexpected fields")
        key = (str(entry["level"]), str(entry["semantic_bucket_id"]))
        if key in decisions:
            raise ValueError(f"bucket_reviews repeats {key}")
        if entry["decision"] not in {"coherent", "needs_refinement"}:
            raise ValueError(f"bucket_reviews[{index}] has an invalid decision")
        if not isinstance(entry["rationale"], str) or not entry["rationale"].strip():
            raise ValueError(f"bucket_reviews[{index}] requires a rationale")
        decisions[key] = entry["decision"]
    if set(decisions) != expected:
        missing = sorted(expected - set(decisions))
        extra = sorted(set(decisions) - expected)
        raise ValueError(
            f"semantic size review coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    canonical_review_path = output / "semantic_bucket_size_review_decisions.json"
    write_json_atomic(canonical_review_path, review)
    status = (
        "complete_reviewed"
        if all(decision == "coherent" for decision in decisions.values())
        else "refinement_required"
    )
    audit_manifest.update(
        {
            "status": status,
            "reviewer": review["reviewer"],
            "review_input_sha256": _file_sha256(review_path),
            "review_artifact": {
                "path": canonical_review_path.name,
                "sha256": _file_sha256(canonical_review_path),
            },
            "coherent_bucket_count": sum(
                decision == "coherent" for decision in decisions.values()
            ),
            "needs_refinement_bucket_count": sum(
                decision == "needs_refinement" for decision in decisions.values()
            ),
        }
    )
    write_json_atomic(audit_manifest_path, audit_manifest)

    if status == "complete_reviewed":
        final_manifest.update(
            {
                "status": "complete_reviewed",
                "size_review_manifest_sha256": _file_sha256(audit_manifest_path),
                "size_review_artifact_sha256": _file_sha256(canonical_review_path),
            }
        )
        write_json_atomic(final_manifest_path, final_manifest)
    return audit_manifest


def _downstream_review_examples(
    source_examples: Mapping[str, tuple[str, dict[str, list[str]]]],
    lookup: Mapping[str, Mapping[str, Any]],
    sample_cards: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    readout_source = "fg"
    readout_level, readout_buckets = source_examples[readout_source]
    readout_bucket, readout_members = max(
        readout_buckets.items(), key=lambda item: len(item[1])
    )
    payload = _bucket_payload(readout_bucket, readout_members, lookup, sample_cards)
    payload.update(
        {
            "task": TASK_NAME,
            "level": readout_level,
            "source_id": readout_source,
            "depth": 0,
            "candidate_columns": [
                {"column": column, "description": COLUMN_DESCRIPTIONS[column]}
                for column in REFINEMENT_COLUMNS[readout_source]
                if column not in INITIAL_COLUMNS[readout_source]
            ],
        }
    )
    values, _ = _value_payload(
        readout_bucket,
        readout_members,
        "canonical_assay_context",
        lookup,
        context_columns=INITIAL_COLUMNS[readout_source],
    )
    values.update({"source_id": readout_source, "depth": 0})
    connection = sqlite3.connect(V3_ROOT / "requests.sqlite3")
    row = connection.execute(
        "SELECT prompt FROM requests ORDER BY batch_id LIMIT 1"
    ).fetchone()
    connection.close()
    if row is None:
        raise ValueError("V3 ranking cache has no prepared prompt")
    return {
        "readout_coherence": _render("readout_coherence", payload),
        "readout_merge_values": _render("readout_merge_values", values),
        "semantic_ranking": str(row[0]),
    }


def prepare_prompt_review(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prompt review directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atoms, sample_cards = _load_atoms()
    lookup = _atom_lookup(atoms)
    buckets_by_group = initial_buckets(atoms)

    rendered: dict[str, str] = {}
    source_examples: dict[str, tuple[str, dict[str, list[str]]]] = {}
    for source in sorted(PAIR_COLUMNS):
        candidates = [
            (level, buckets)
            for (level, candidate_source), buckets in buckets_by_group.items()
            if candidate_source == source
        ]
        level, buckets = max(candidates, key=lambda item: (len(item[1]), item[0]))
        source_examples[source] = (level, buckets)
        consumed = INITIAL_COLUMNS[source]
        selector = _column_selection_payload(level, source, buckets, lookup, consumed)
        rendered[f"select_column_{source}"] = _render("select_column", selector)

        best: tuple[int, str, str] | None = None
        for bucket, members in buckets.items():
            for column in REFINEMENT_COLUMNS[source]:
                if column in consumed:
                    continue
                distinct = len({lookup[atom]["values"][column] for atom in members})
                if distinct > 1 and (best is None or distinct > best[0]):
                    best = (distinct, bucket, column)
        if best is None:
            bucket = next(iter(buckets))
            column = next(
                column for column in REFINEMENT_COLUMNS[source] if column not in consumed
            )
        else:
            _, bucket, column = best
        values, _ = _value_payload(
            bucket,
            buckets[bucket],
            column,
            lookup,
            context_columns=consumed,
        )
        values.update({"task": TASK_NAME, "level": level, "source_id": source})
        rendered[f"split_bucket_{source}"] = _render("split_bucket", values)
        rendered[f"merge_values_{source}"] = _render("merge_values", values)

    merge_source = max(source_examples, key=lambda source: len(source_examples[source][1]))
    merge_level, merge_candidates = source_examples[merge_source]
    selected = dict(sorted(merge_candidates.items())[:MERGE_BATCH_SIZE])
    rendered["merge_buckets_40"] = _render(
        "merge_buckets",
        _merge_bucket_payload(
            merge_level, merge_source, selected, lookup, sample_cards
        ),
    )

    if INCLUDE_DOWNSTREAM_PROMPT_REVIEW:
        rendered.update(_downstream_review_examples(source_examples, lookup, sample_cards))

    maximum_tokens, endpoint_models = _endpoint_contract()
    prompt_rows = []
    for name, prompt in sorted(rendered.items()):
        path = output / f"{name}.txt"
        path.write_text(prompt + "\n", encoding="utf-8")
        input_tokens = _token_count(prompt)
        completion_reserve = HIGH_MAX_TOKENS if name.startswith("merge_") else LOW_MAX_TOKENS
        prompt_rows.append(
            {
                "name": name,
                "path": path.name,
                "sha256": _file_sha256(path),
                "input_tokens": input_tokens,
                "completion_token_reserve": completion_reserve,
                "fits_model_context": input_tokens + completion_reserve <= maximum_tokens,
            }
        )

    review_lines = [
        f"# {PROMPT_REVIEW_TITLE}",
        "",
        "No completion requests were made. Review and approve the prompt hashes in `manifest.json` before running DeepSeek.",
        "",
        "| Prompt | Input tokens | Reserved completion | Sendable |",
        "|---|---:|---:|:---:|",
    ]
    for row in prompt_rows:
        review_lines.append(
            f"| [{row['name']}]({row['path']}) | {row['input_tokens']:,} | "
            f"{row['completion_token_reserve']:,} | "
            f"{'yes' if row['fits_model_context'] else 'no'} |"
        )
    review_lines.extend(
        (
            "",
            "A non-sendable full-value prompt is retained to demonstrate the fail-closed size gate. That bucket/column combination must be treated as ineligible; it must not be truncated or sampled.",
            "",
        )
    )
    (output / "README.md").write_text("\n".join(review_lines), encoding="utf-8")

    atoms.to_parquet(output / "input_atoms.parquet", index=False)
    manifest = {
        "version": f"{VERSION}.prompt_review.v1",
        "status": "awaiting_user_prompt_approval",
        "completion_requests_made": 0,
        "model": MODEL,
        "base_url": BASE_URL,
        "endpoint_models": endpoint_models,
        "max_model_len": maximum_tokens,
        "source_record_map": {
            "path": str(RECORD_MAP),
            "sha256": _file_sha256(RECORD_MAP),
        },
        "v10_records": {"path": str(RECORDS), "sha256": _file_sha256(RECORDS)},
        "counts": {
            "records_level_scoped": EXPECTED_RECORD_COUNT,
            "level_source_pair_bucket_atoms": len(atoms),
            "initial_level_scoped_buckets": sum(
                len(buckets) for buckets in buckets_by_group.values()
            ),
        },
        "settings": {
            "max_source_rounds": MAX_SOURCE_ROUNDS,
            "merge_batch_size": MERGE_BATCH_SIZE,
            "merge_rounds_per_operation": 2,
            "sample_limit": SAMPLE_LIMIT,
            "column_selection_reasoning": "high",
            "split_reasoning": "high",
            "value_and_bucket_merge_reasoning": "high",
            "ranking_reasoning": "high",
            "keep_is_terminal": True,
            "technical_noop_policy": (
                "defer" if DEFER_TECHNICAL_BRANCHES else "keep"
            ),
            "batch_seed_version": BATCH_SEED_VERSION,
            "refinement_columns": {
                source: list(columns)
                for source, columns in sorted(REFINEMENT_COLUMNS.items())
            },
            "ignored_pair_bucket_columns": {
                source: sorted(set(PAIR_COLUMNS[source]) - set(columns))
                for source, columns in sorted(REFINEMENT_COLUMNS.items())
            },
            "same_parent_recombination_allowed": True,
            "readout_maximum_depth": 2,
            "readout_coherence_is_intrinsic": True,
            "readout_keep_is_terminal": True,
            "readout_cross_bucket_merge": False,
        },
        "prompts": prompt_rows,
        "non_sendable_prompts": [
            row["name"] for row in prompt_rows if not row["fits_model_context"]
        ],
        "oversized_prompt_policy": "mark_bucket_column_ineligible_without_a_completion_request",
        "review_index_sha256": _file_sha256(output / "README.md"),
        "prompt_template_hashes": {
            path.name: _file_sha256(path) for path in sorted(PROMPT_ROOT.glob("*.jinja"))
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-prompts", help="render and validate prompts without model completions"
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_REVIEW)
    run = subparsers.add_parser(
        "run-semantic", help="run the approved source-local semantic refinement"
    )
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--review-manifest", type=Path, required=True)
    run.add_argument("--approved-review-sha256", required=True)
    run.add_argument("--parallelism", type=int, default=128)
    run.add_argument("--seed-request-cache", type=Path)
    publish = subparsers.add_parser(
        "publish-final-map", help="apply the agent-reviewed cross-source merge map"
    )
    publish.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    publish.add_argument("--review", type=Path, required=True)
    cross_review = subparsers.add_parser(
        "prepare-cross-source-prompts",
        help="render cross-source prompts without model completions",
    )
    cross_review.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cross_review.add_argument("--review-output", type=Path, required=True)
    cross_run = subparsers.add_parser(
        "run-cross-source", help="run approved DeepSeek cross-source merge proposals"
    )
    cross_run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cross_run.add_argument("--review-manifest", type=Path, required=True)
    cross_run.add_argument("--approved-review-sha256", required=True)
    cross_run.add_argument("--parallelism", type=int, default=64)
    audit = subparsers.add_parser(
        "audit-semantic-size", help="prepare the pre-ranking semantic size review"
    )
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    approve = subparsers.add_parser(
        "approve-semantic-size", help="record the scientific semantic size review"
    )
    approve.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    approve.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-prompts":
        manifest = prepare_prompt_review(args.output)
        print(json.dumps(manifest, indent=2))
    elif args.command == "run-semantic":
        manifest = run_semantic(
            args.output,
            review_manifest_path=args.review_manifest,
            approved_review_sha256=args.approved_review_sha256,
            parallelism=args.parallelism,
            seed_request_cache=args.seed_request_cache,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "publish-final-map":
        manifest = publish_final_map(args.output, review_path=args.review)
        print(json.dumps(manifest, indent=2))
    elif args.command == "prepare-cross-source-prompts":
        manifest = prepare_cross_source_prompt_review(
            args.output, review_output=args.review_output
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "run-cross-source":
        manifest = run_cross_source_merges(
            args.output,
            review_manifest_path=args.review_manifest,
            approved_review_sha256=args.approved_review_sha256,
            parallelism=args.parallelism,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "audit-semantic-size":
        manifest = audit_semantic_bucket_sizes(args.output)
        print(json.dumps(manifest, indent=2))
    elif args.command == "approve-semantic-size":
        manifest = approve_semantic_bucket_sizes(
            args.output, review_path=args.review
        )
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
