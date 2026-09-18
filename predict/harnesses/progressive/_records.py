"""Internal record-to-card assembly used by the progressive prompt boundary.

The immutable cache has already selected records. This module turns those rows
into cumulative molecule cards while preserving record conditions and score
scope. ``prompt.py`` is the public rendering API and ``inference.py`` owns model
calls, validation, and checkpoints.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq
import yaml

from predict.harnesses.progressive.state import (
    ProgressiveTaskContract,
    build_progressive_messages,
    card_alias_maps,
    molecule_card_contract,
    render_prior_state,
)
from predict.harnesses.progressive.references import (
    build_reference_index,
    validate_prompt_index,
    validate_unique_visible_ids,
)
from predict.retrieval.assay_reranking.v9 import (
    RANKING_SCHEMA_VERSION,
    model_profile,
    verify_vendored_assets,
)
from predict.harnesses.progressive.prompt import ACTIVE_PROMPT_VERSION, PROMPT_DIR, behavior_version, prompt_asset_path, prompt_directory, prompt_assets, render_progressive_messages
from predict.retrieval.policies import normalize_molecule_identity, seeded_rank_tie_key
from predict.utils.json import read_jsonl, sha256_file
from data.processing.evidence_library.versions.v9.tasks.bbb_martins.semantic_display import (
    semantic_display_fields,
    semantic_prompt_payload,
)


PROTOCOL_VERSION = "conditioned_gold_context_records_progressive.v1"
INDIRECT_PROTOCOL_VERSION = (
    "conditioned_gold_context_records_plus_stage3_later_levels.v2"
)
TIANANG_ALIGNED_PROTOCOL_VERSION = "conditioned_context_records_tianang_aligned.v1"
PROMPT_PROFILES = {
    ACTIVE_PROMPT_VERSION: {
        "name": "progressive.reranked.v8",
        "card": prompt_directory(ACTIVE_PROMPT_VERSION) / "card.yaml",
        "template": prompt_directory(ACTIVE_PROMPT_VERSION) / "system.jinja",
        "ranking": "morgan",
        "tianang_aligned": True,
        "stage_ranked": True,
    },
}
V7_ROOT = Path("data/evidence_libraries")
BIOAVAILABILITY_CLAIMS = Path(
    "data/artifacts/starling/bioavailability_ma/canonical_sources/"
    "canonical_direct_v2/direct_claims.parquet"
)
BIOAVAILABILITY_CLAIMS_MANIFEST = BIOAVAILABILITY_CLAIMS.with_name(
    "merge_manifest.json"
)
CONTEXT_LIMIT = 10
RECORD_LIMIT = 10
INDIRECT_RECORD_LIMIT = 50
LEVEL_RECORD_BUNDLE_PATH = Path(__file__).with_name("level_record_bundle.yaml")
GOLD_SOURCE_DOMAINS = {
    "bbb_martins": frozenset({"direct_bbb"}),
    "bioavailability_ma": frozenset({"hf_bioavailability", "oral_exposure"}),
    "skin_reaction": frozenset({"direct_skin_reaction", "sensitization_aop"}),
}
LEVELS = (
    {
        "level": 1,
        "endpoint_group": "current_gold_voting_records",
        "description": "Raw source records that contribute to the current conditioned-gold context.",
    },
    {
        "level": 2,
        "endpoint_group": "associated_nonvoting_records",
        "description": "Other rows from the gold source domain with the same parent and exact condition; these are context, not extra votes.",
    },
)

_SOURCE_RECORD_FIELDS = (
    "endpoint_name",
    "measurement_text",
    "unit_text",
    "support_text",
    "confidence",
    "extra_details",
    "assay_model",
    "assay_system",
    "assay_type",
    "assay_or_test",
    "assay_method",
    "biological_system",
    "biological_context",
    "evidence_system",
    "species",
    "species_or_population",
    "qualifying_conditions",
    "experimental_conditions",
    "study_context",
    "perturbation",
    "transporter_identifier",
    "transport_mechanism",
    "evidence_type",
    "evidence_basis",
    "interaction_conclusion",
    "metric_uncertainty",
    "statistic_type",
    "oral_dose",
    "dose",
    "comparator",
    "comparator_exposure",
    "bioavailability_report_type",
    "oral_exposure_mode",
    "condition_medium",
    "formulation_or_solid_form",
    "formulation_vehicle",
    "intestinal_site",
    "molecular_form",
    "substrate_status",
    "transporter_or_enzyme",
    "enzyme_or_pathway",
    "dose_or_concentration",
    "positive_count",
    "total_tested",
    "aop_event",
    "light_conditions",
    "exposure_time",
    "skin_source",
    "study_design",
)

_CANONICAL_RECORD_FIELDS = (
    "canonical_endpoint_name",
    "canonical_endpoint_concept",
    "canonical_measurement_text",
    "canonical_unit_text",
    "canonical_assay_context",
    "canonical_species_context",
    "canonical_reference_scope",
    "canonical_reference_basis",
    "canonical_assay_type",
    "canonical_evidence_type",
    "canonical_transport_mechanism",
    "canonical_transporter_identifier",
    "canonical_kinetic_symbol",
    "canonical_bioavailability_report_type",
    "canonical_bioavailability_evidence_scope",
    "canonical_oral_dose_value",
    "canonical_oral_dose_unit",
    "canonical_oral_dose_basis",
    "canonical_biological_matrix",
    "canonical_assay_or_test",
    "canonical_species_or_population",
    "canonical_severity_grade",
    "canonical_aop_event",
    "canonical_assay_method",
    "canonical_evidence_system",
    "canonical_study_design",
)

_V7_COLUMNS = tuple(
    dict.fromkeys(
        (
            "canonical_record_id",
            "canonical_smiles",
            "source_id",
            "measurement_kind",
            "canonical_measurement_scale_id",
            "canonical_category_id",
            "finite_scalar_value",
            "source_index",
            "source_row_number",
            "source_record_id",
            "pmid",
            "condition_group",
            "canonical_endpoint_name",
            "canonical_measurement_text",
            "resolved_measurement_text",
            "canonical_unit_text",
            "resolved_unit_text",
            "canonical_assay_context",
            "canonical_species_context",
            "canonical_species_or_population",
            *_SOURCE_RECORD_FIELDS,
            *_CANONICAL_RECORD_FIELDS,
        )
    )
)


def record_level_names(task: str, *, first_level: int = 3) -> tuple[str, ...]:
    task_levels = prompt_assets(ACTIVE_PROMPT_VERSION)["levels"].get(task)
    if not isinstance(task_levels, Mapping):
        raise ValueError(f"cache-backed record levels are unsupported for {task}")
    names = tuple(
        sorted(
            (name for name in task_levels if int(name[1:]) >= first_level),
            key=lambda name: int(name[1:]),
        )
    )
    expected = tuple(f"L{level}" for level in range(first_level, first_level + len(names)))
    if names != expected:
        raise ValueError(
            f"{task} cache-backed levels must be contiguous from L{first_level}"
        )
    return names


def indirect_level_names(task: str) -> tuple[str, ...]:
    return record_level_names(task)


def levels(
    max_level: int = 0,
    *,
    task: str | None = None,
    include_indirect: bool = False,
    prompt_version: str = 'v1',
) -> list[dict[str, Any]]:
    # L1/L2 context definitions are task-independent, including custom test contracts.
    context_levels = prompt_assets(prompt_version)['context_levels']
    rows = [dict(row) for row in next(iter(context_levels.values()))]
    if include_indirect:
        if task is None:
            raise ValueError("task is required for cache-backed later levels")
        task_levels = level_record_bundle_contract(prompt_version)["task_levels"][task]
        for level_name in indirect_level_names(task):
            definition = task_levels[level_name]
            rows.append(
                {
                    "level": int(level_name[1:]),
                    "endpoint_group": str(definition["evidence_family"]),
                    "description": str(definition["description"]),
                }
            )
    return [row for row in rows if not max_level or int(row["level"]) <= max_level]


def resolve_prompt_version(
    task: str, prompt_version: str, *, ranking: str = "assay_transfer"
) -> str:
    del task, ranking
    resolved = ACTIVE_PROMPT_VERSION if prompt_version == "task_best" else prompt_version
    prompt_profile(resolved)
    return resolved


def prompt_profile(prompt_version: str) -> Mapping[str, Any]:
    behavior = behavior_version(prompt_version)
    if behavior != ACTIVE_PROMPT_VERSION:
        raise ValueError(
            f"inactive prompt version {prompt_version!r}; use or clone "
            f"{ACTIVE_PROMPT_VERSION!r}"
        )
    directory = prompt_directory(prompt_version)
    return {
        **PROMPT_PROFILES[ACTIVE_PROMPT_VERSION],
        'name': prompt_version,
        'card': prompt_asset_path(prompt_version, 'card.yaml'),
        'template': prompt_asset_path(prompt_version, 'system.jinja'),
        'directory': directory,
    }


def is_tianang_aligned(prompt_version: str) -> bool:
    return bool(prompt_profile(prompt_version).get("tianang_aligned"))


@lru_cache(maxsize=1)
def tianang_aligned_card_contract(prompt_version: str = ACTIVE_PROMPT_VERSION) -> dict[str, Any]:
    contract = molecule_card_contract(prompt_asset_path(prompt_version, 'card.yaml'))
    contract['task_levels'] = prompt_assets(prompt_version)['levels']
    task_levels = contract.get("task_levels")
    if not isinstance(task_levels, Mapping):
        raise ValueError(f"{prompt_profile(prompt_version)['card']} lacks task_levels")
    for task, expected_last_level in (("bbb_martins", 5), ("bioavailability_ma", 6)):
        names = list((task_levels.get(task) or {}).keys())
        if names != [f"L{level}" for level in range(1, expected_last_level + 1)]:
            raise ValueError(
                f"{prompt_profile(prompt_version)['card']} has invalid {task} levels"
            )
    return contract


def tianang_aligned_levels(task: str, max_level: int = 0, *, prompt_version: str = ACTIVE_PROMPT_VERSION) -> list[dict[str, Any]]:
    try:
        definitions = tianang_aligned_card_contract(prompt_version)["task_levels"][task]
    except KeyError as exc:
        raise ValueError(f"Tianang-aligned levels are unsupported for {task}") from exc
    rows = [
        {
            "level": int(name[1:]),
            "endpoint_group": str(definition["evidence_family"]),
            "description": str(definition["description"]),
        }
        for name, definition in definitions.items()
    ]
    return rows[:max_level] if max_level else rows


def selected_physical_record_ids(
    contexts: Iterable[Mapping[str, Any]],
) -> set[str]:
    return {
        str(card["_canonical_record_id"])
        for context in contexts
        for level_cards in (context.get("l1_cards") or [], context.get("l2_cards") or [])
        for card in level_cards
    }


def _score_field(prompt_version: str) -> str:
    return (
        "morgan_similarity"
        if prompt_profile(prompt_version).get("ranking") == "morgan"
        else "transfer_likelihood"
    )


def _replace_score_field(contract: dict[str, Any], prompt_version: str) -> dict[str, Any]:
    if _score_field(prompt_version) == "transfer_likelihood":
        return contract
    transformed = deepcopy(contract)
    for section in ("context", "record"):
        for field in (transformed.get(section) or {}).get("fields") or []:
            if field.get("name") == "transfer_likelihood":
                field["name"] = "morgan_similarity"
                field["source"] = "morgan_similarity"
    return transformed


@lru_cache(maxsize=None)
def card_contract(prompt_version: str = "v1") -> dict[str, Any]:
    profile = prompt_profile(prompt_version)
    path = profile["card"]
    contract = _replace_score_field(
        deepcopy(prompt_assets(prompt_version)['card']), prompt_version
    )
    card_schema_version = str(profile.get("card_schema_version") or prompt_version)
    expected_schema = f"progressive_context_record_card.{card_schema_version}"
    if not isinstance(contract, dict) or contract.get("schema_version") != expected_schema:
        raise ValueError(f"{path} has unsupported schema_version; expected {expected_schema}")
    required_record_fields = {"card_id", "level"}
    if card_schema_version == "v1":
        required_record_fields.add("result")
    required = {
        "context": {"context_card_id", "canonical_smiles", "condition_group", _score_field(prompt_version), "available_record_counts"},
        "record": required_record_fields,
    }
    for section, required_names in required.items():
        fields = (contract.get(section) or {}).get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"{path} {section}.fields must be non-empty")
        names = [str(field.get("name") or "") for field in fields]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"{path} {section}.fields has invalid names")
        if not required_names <= set(names):
            raise ValueError(f"{path} {section}.fields lacks required fields")
        for field in fields:
            if not str(field.get("source") or "").strip():
                raise ValueError(f"{path} field {field.get('name')!r} needs a source")
    if not str(contract["context"].get("records_field") or "").strip():
        raise ValueError(f"{path} context.records_field is required")
    return contract


@lru_cache(maxsize=None)
def level_record_bundle_contract(prompt_version: str = "v1") -> dict[str, Any]:
    contract = _replace_score_field(
        deepcopy(prompt_assets(prompt_version)['bundle']),
        prompt_version,
    )
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != "progressive_level_record_bundle.v1"
    ):
        raise ValueError(f"{LEVEL_RECORD_BUNDLE_PATH} has unsupported schema_version")
    for section, required_names in {
        "bundle": {
            "level_card_id",
            "level",
            "evidence_family",
            "description",
            "selection",
            "available_record_count",
            "selected_record_count",
        },
        "record": {
            "card_id",
            "reference_molecule_id",
            _score_field(prompt_version),
            "source_schema_id",
            "source_values",
        },
        "molecule": {"molecule_id", "smiles"},
        "source_schema": {"schema_id", "source_id", "source_columns"},
    }.items():
        fields = (contract.get(section) or {}).get("fields")
        names = [str(field.get("name") or "") for field in fields or []]
        if not fields or len(names) != len(set(names)) or not required_names <= set(names):
            raise ValueError(f"{LEVEL_RECORD_BUNDLE_PATH} has invalid {section}.fields")
        if any(not str(field.get("source") or "").strip() for field in fields):
            raise ValueError(f"{LEVEL_RECORD_BUNDLE_PATH} {section} field lacks source")
    required_names = (
        ("bundle", "molecules_field"),
        ("bundle", "source_schemas_field"),
        ("bundle", "records_field"),
        ("cumulative_context_records", "columns_field"),
        ("cumulative_context_records", "rows_field"),
    )
    for section, name in required_names:
        if not str((contract.get(section) or {}).get(name) or "").strip():
            raise ValueError(f"{LEVEL_RECORD_BUNDLE_PATH} {section}.{name} is required")
    return contract


def _project(values: Mapping[str, Any], fields: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in fields:
        name = str(field["name"])
        value = values.get(str(field["source"]))
        if value in (None, "", []):
            if field.get("required") is True:
                raise ValueError(f"required context-record card field is blank: {name}")
            continue
        output[name] = value
    return output


def _first(row: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _task_instructions(
    contract: ProgressiveTaskContract, *, structure_score_visible: bool = False
) -> list[str]:
    """Drop instructions that refer to similarity fields hidden by this profile."""
    if structure_score_visible:
        return list(contract.task_instructions)
    hidden_terms = ("tanimoto", "distant_analog", "very_distant_analog")
    return [
        instruction
        for instruction in contract.task_instructions
        if not any(term in instruction.lower() for term in hidden_terms)
    ]


def _record_surface(row: Mapping[str, Any], *, level: int, context_id: str) -> dict[str, Any]:
    canonical_id = str(row.get("canonical_record_id") or "")
    if not canonical_id:
        raise ValueError("normalized V7 source row lacks canonical_record_id")
    card_id = "record_" + hashlib.sha256(
        f"{PROTOCOL_VERSION}\0{context_id}\0{canonical_id}".encode("utf-8")
    ).hexdigest()[:16]
    display = semantic_display_fields(row)
    result = str(display["measurement_text"] or "")
    if not result:
        result = "not explicitly reported"
    surface = {
        "card_id": card_id,
        "first_seen_level": level,
        "level": f"L{level}",
        "endpoint": _first(row, "canonical_endpoint_name", "endpoint_name"),
        "result": result,
        "unit": str(display["unit_text"] or ""),
        "assay_context": _first(
            row,
            "canonical_assay_context",
            "assay_model",
            "assay_or_test",
            "assay_method",
        ),
        "species": _first(
            row,
            "canonical_species_context",
            "canonical_species_or_population",
            "species",
            "species_or_population",
        ),
        "conditions": _first(
            row,
            "qualifying_conditions",
            "experimental_conditions",
            "study_context",
        ),
        "support_text": _first(row, "support_text"),
        "source_id": _first(row, "source_id"),
        "measurement_kind": _first(row, "measurement_kind"),
        "_canonical_record_id": canonical_id,
    }
    source = {field: row.get(field) for field in _SOURCE_RECORD_FIELDS}
    if row.get("source_id") in {"direct_bbb", "efflux_transport", "influx_transport", "passive_permeability"}:
        source = semantic_prompt_payload(row, source)
    surface.update({f"source__{field}": value for field, value in source.items()})
    surface.update({field: row.get(field) for field in _CANONICAL_RECORD_FIELDS})
    surface.update(
        canonical_measurement_text=display["measurement_text"],
        canonical_unit_text=display["unit_text"],
        canonical_transporter_identifier=display["transporter_identifier"],
    )
    return surface


def _stable_sample(
    rows: list[dict[str, Any]],
    *,
    task: str,
    context_id: str,
    level: int,
    record_limit: int,
) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            f"{PROTOCOL_VERSION}\0{task}\0{context_id}\0L{level}\0{row['canonical_record_id']}".encode("utf-8")
        ).hexdigest()

    return sorted(rows, key=key)[:record_limit]


def v21_molecule_contexts(
    query_id: str, ranked_molecules: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Project ranked raw records into append-only molecule cards."""
    contexts = []
    for molecule in ranked_molecules:
        score_field = "morgan_similarity" if "morgan_similarity" in molecule else "transfer_likelihood"
        molecule_id = str(molecule["reference_molecule_id"])
        context_id = "context_" + hashlib.sha256(
            f"{INDIRECT_PROTOCOL_VERSION}\0{query_id}\0{molecule_id}".encode("utf-8")
        ).hexdigest()[:16]

        def cards(level: int) -> list[dict[str, Any]]:
            output = []
            for ranked_record in molecule[f"l{level}_records"]:
                payload = {
                    **dict(ranked_record["payload"]),
                    "canonical_record_id": ranked_record["record_id"],
                }
                card = _record_surface(payload, level=level, context_id=context_id)
                card["_record_ranking_score"] = float(ranked_record[score_field])
                if score_field == "transfer_likelihood":
                    card["_v21_transfer_likelihood"] = float(ranked_record[score_field])
                if "node_key" in ranked_record:
                    card["_relevance_node_key"] = ranked_record["node_key"]
                    card["_relevance_level_rank"] = ranked_record["level_rank"]
                output.append(card)
            return output

        contexts.append(
            {
                "context_card_id": context_id,
                "canonical_smiles": str(molecule["canonical_smiles"]),
                "condition_group": "no_reported_external_condition" if "level_mapping" in molecule else "source_native_assay_contexts",
                score_field: round(float(molecule[score_field]), 4),
                "available_l1": int(molecule["available_l1"]),
                "available_l2": int(molecule["available_l2"]),
                "l1_cards": cards(1),
                "l2_cards": cards(2),
                "_selection_rank": int(molecule["selection_rank"]),
                "_gold_record_id": None,
                "_context_origin": "mapped_record_ranked_molecule" if "level_mapping" in molecule else "v21_record_ranked_molecule",
                "_reference_molecule_id": molecule_id,
                **({"_level_mapping": molecule["level_mapping"]} if "level_mapping" in molecule else {}),
            }
        )
    return contexts


def _bbb_gold_source_indices(gold: Mapping[str, Any]) -> set[int]:
    indices = set()
    for value in gold.get("source_record_ids") or []:
        tail = str(value).rsplit(":", 1)[-1]
        if not tail.isdigit():
            raise ValueError(f"BBB gold source ID has no source index: {value!r}")
        indices.add(int(tail))
    return indices


def _bioavailability_claim_by_source(
    selected_gold: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    selected_claims = {
        str(claim_id)
        for gold in selected_gold.values()
        for claim_id in gold.get("source_record_ids") or []
    }
    manifest = json.loads(
        BIOAVAILABILITY_CLAIMS_MANIFEST.read_text(encoding="utf-8")
    )
    expected_hash = (manifest.get("paths") or {}).get("direct_claims_sha256")
    actual_hash = sha256_file(BIOAVAILABILITY_CLAIMS)
    if expected_hash != actual_hash:
        raise ValueError("Bioavailability direct-claim hash disagrees with its manifest")
    source_to_claim: dict[str, str] = {}
    table = pq.read_table(
        BIOAVAILABILITY_CLAIMS,
        columns=["canonical_claim_id", "source_record_ids"],
    )
    for row in table.to_pylist():
        claim_id = str(row["canonical_claim_id"])
        if claim_id not in selected_claims:
            continue
        for source_id in row.get("source_record_ids") or []:
            source_id = str(source_id)
            previous = source_to_claim.setdefault(source_id, claim_id)
            if previous != claim_id:
                raise ValueError(
                    f"Bioavailability source row maps to multiple selected claims: {source_id}"
                )
    missing = selected_claims - set(source_to_claim.values())
    if missing:
        raise ValueError(
            f"Bioavailability selected gold claims lack source rows: {len(missing)}"
        )
    return source_to_claim, {
        "path": str(BIOAVAILABILITY_CLAIMS),
        "sha256": actual_hash,
        "manifest": str(BIOAVAILABILITY_CLAIMS_MANIFEST),
        "manifest_sha256": sha256_file(BIOAVAILABILITY_CLAIMS_MANIFEST),
    }


def _bioavailability_source_id(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_id") or "")
    if source == "hf_bioavailability":
        return f"hf:{row.get('source_record_id')}"
    if source == "oral_exposure" and row.get("source_row_number") is not None:
        return f"local:{int(row['source_row_number']) - 1}:{row.get('source_record_id')}"
    return ""


def load_candidates(
    *,
    task: str,
    valid_records: list[dict[str, Any]],
    ranking_root: Path,
    benchmark_root: Path,
    v7_root: Path = V7_ROOT,
    record_limit: int = RECORD_LIMIT,
    l2_record_limit: int | None = None,
    context_limit: int = CONTEXT_LIMIT,
    ranking: str = "assay_transfer",
    tie_seed: int = 0,
    distinct_molecules: bool = False,
    reference_path: Path | None = None,
    cache_dir: Path | None = None,
    gold_candidates_only: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load V9 contexts per query and attach sampled V7 L1/L2 rows."""
    if l2_record_limit is None:
        l2_record_limit = record_limit
    if record_limit < 1:
        raise ValueError("record_limit must be positive")
    if l2_record_limit < 1:
        raise ValueError("l2_record_limit must be positive")
    if not 1 <= context_limit <= 100:
        raise ValueError("context_limit must be between 1 and 100")
    if ranking not in {"assay_transfer", "morgan"}:
        raise ValueError(f"unsupported context ranking: {ranking}")
    task_dir = {
        "bbb_martins": "BBB_Martins",
        "bioavailability_ma": "Bioavailability_Ma",
        "skin_reaction": "Skin_Reaction",
    }[task]
    if (reference_path is None) != (cache_dir is None):
        raise ValueError('Custom references require an explicit matched cache directory')
    if reference_path is None:
        current_path = benchmark_root / task_dir / "CURRENT"
        release = current_path.read_text(encoding="utf-8").strip()
        if not release or "/" in release or "\\" in release:
            raise ValueError(f"invalid active gold pointer: {current_path}")
        gold_path = benchmark_root / task_dir / release / "scaffold/train_molecule_condition_labels.jsonl"
    else:
        release, gold_path = 'train_only_internal_reference', reference_path
    gold_rows = read_jsonl(gold_path)
    gold_by_id = {str(row["benchmark_row_id"]): row for row in gold_rows}

    cache_dir = cache_dir or ranking_root / task / "scaffold" / "valid"
    version_path = cache_dir / "VERSION.json"
    rankings_path = cache_dir / "rankings.parquet"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    if version.get("schema_version") != RANKING_SCHEMA_VERSION or version.get("status") != "complete":
        raise ValueError(f"{task} V9 ranking cache is incomplete or incompatible")
    if version.get("rankings_sha256") != sha256_file(rankings_path):
        raise ValueError(f"{task} V9 rankings hash disagrees with VERSION.json")
    if version.get("model") != model_profile(task):
        raise ValueError(f"{task} V9 cache uses the wrong direct model")
    if version.get("prompt_assets") != verify_vendored_assets():
        raise ValueError(f"{task} V9 prompt assets differ from the cache build")
    if set(version.get("training_record_ids") or []) != set(gold_by_id):
        raise ValueError(f"{task} V9 cache training IDs differ from active gold")

    valid_by_id = {str(row["benchmark_row_id"]): row for row in valid_records}
    ranking_rows = pq.read_table(rankings_path).to_pylist()
    if {str(row["query_record_id"]) for row in ranking_rows} != set(valid_by_id):
        raise ValueError(f"{task} V9 cache queries differ from active valid gold")
    rows_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranking_rows:
        query_id = str(row["query_record_id"])
        rows_by_query[query_id].append(dict(row))
    selected_by_query: dict[str, list[dict[str, Any]]] = {}
    for query_id, rows in rows_by_query.items():
        if ranking == "assay_transfer":
            ranked = sorted(rows, key=lambda row: int(row["model_rank"]))
        else:
            ranked = sorted(
                rows,
                key=lambda row: (
                    -float(row["morgan_tanimoto_similarity"]),
                    seeded_rank_tie_key(
                        tie_seed,
                        task,
                        query_id,
                        row["retrieval_molecule_identity_key"],
                        row["retrieval_record_id"],
                    ),
                ),
            )
        if distinct_molecules:
            selected = []
            seen = set()
            for row in ranked:
                identity = str(row["retrieval_molecule_identity_key"])
                if identity in seen:
                    continue
                seen.add(identity)
                selected.append(row)
                if len(selected) == context_limit:
                    break
        else:
            selected = ranked[:context_limit]
        for selection_rank, row in enumerate(selected):
            row["_selection_rank"] = selection_rank
        selected_by_query[query_id] = selected
    for query_id, rows in selected_by_query.items():
        if len(rows) != context_limit:
            raise ValueError(f"{task} query {query_id} lacks {context_limit} contexts")
        if ranking == "assay_transfer" and not distinct_molecules:
            ranks = sorted(int(row["model_rank"]) for row in rows)
            if ranks != list(range(context_limit)):
                raise ValueError(
                    f"{task} query {query_id} lacks exact V9 ranks 0-{context_limit - 1}"
                )
        for row in rows:
            retrieval_id = str(row["retrieval_record_id"])
            gold = gold_by_id.get(retrieval_id)
            if gold is None:
                raise ValueError(
                    f"{task} V9 cache contains stale gold context {retrieval_id}"
                )
            if (
                str(row["retrieval_molecule_identity_key"])
                != str(gold["molecule_identity_key"])
                or str(row["retrieval_condition_group"])
                != str(gold["condition_group"])
            ):
                raise ValueError(
                    f"{task} V9 context identity disagrees with active gold: {retrieval_id}"
                )

    if gold_candidates_only:
        return selected_by_query, {
            'gold_train_labels': str(gold_path), 'gold_train_labels_sha256': sha256_file(gold_path),
            'ranking_version': str(version_path), 'ranking_version_sha256': sha256_file(version_path),
            'rankings': str(rankings_path), 'rankings_sha256': sha256_file(rankings_path),
            'candidate_policy': version.get('candidate_policy'),
        }

    selected_gold_ids = {
        str(row["retrieval_record_id"])
        for rows in selected_by_query.values()
        for row in rows
    }
    selected_gold = {record_id: gold_by_id[record_id] for record_id in selected_gold_ids}
    selected_keys = {
        (str(row["molecule_identity_key"]), str(row["condition_group"])): record_id
        for record_id, row in selected_gold.items()
    }
    if len(selected_keys) != len(selected_gold):
        raise ValueError(f"{task} active gold has duplicate parent-condition context IDs")

    bbb_context_by_source_index: dict[int, str] = {}
    if task == "bbb_martins":
        for record_id, gold in selected_gold.items():
            for source_index in _bbb_gold_source_indices(gold):
                previous = bbb_context_by_source_index.setdefault(source_index, record_id)
                if previous != record_id:
                    raise ValueError(
                        f"BBB source index belongs to multiple selected gold contexts: {source_index}"
                    )
    bio_source_to_claim: dict[str, str] = {}
    bio_context_by_claim: dict[str, str] = {}
    bio_claim_audit: dict[str, Any] | None = None
    if task == "bioavailability_ma":
        bio_source_to_claim, bio_claim_audit = _bioavailability_claim_by_source(
            selected_gold
        )
        for record_id, gold in selected_gold.items():
            for claim_id in gold.get("source_record_ids") or []:
                claim_id = str(claim_id)
                previous = bio_context_by_claim.setdefault(claim_id, record_id)
                if previous != record_id:
                    raise ValueError(
                        f"Bioavailability claim belongs to multiple selected contexts: {claim_id}"
                    )
    skin_context_by_signature: dict[tuple[str, str, str], str] = {}
    skin_contexts_by_source_pmid: dict[tuple[str, str], set[str]] = defaultdict(set)
    skin_context_by_source_index: dict[tuple[str, int], str] = {}
    skin_gold_source_counts: dict[str, int] = {}
    if task == "skin_reaction":
        for record_id, gold in selected_gold.items():
            skin_gold_source_counts[record_id] = int(gold["source_record_count"])
            parent = str(gold["molecule_identity_key"])
            for source_id in gold.get("source_record_ids") or []:
                source_id = str(source_id)
                source_name, separator, source_index_text = source_id.rpartition(":")
                if separator and source_name in GOLD_SOURCE_DOMAINS[task]:
                    source_index = int(source_index_text)
                    previous = skin_context_by_source_index.setdefault(
                        (source_name, source_index), record_id
                    )
                    if previous != record_id:
                        raise ValueError(
                            f"Skin source index belongs to multiple contexts: {source_index}"
                        )
                for pmid in gold.get("source_pmids") or []:
                    signature = (parent, source_id, str(pmid))
                    skin_contexts_by_source_pmid[(source_id, str(pmid))].add(record_id)
                    previous = skin_context_by_signature.setdefault(signature, record_id)
                    if previous != record_id:
                        raise ValueError(
                            f"Skin provenance signature belongs to multiple contexts: {signature}"
                        )

    stage3_dir = v7_root / task / "v7" / "03_pair_buckets"
    manifest_path = stage3_dir / "manifest.json"
    records_path = stage3_dir / "records.parquet"
    stage3_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stage3_manifest.get("task_id") != task or stage3_manifest.get("version") != "starling_core_stage3.v1":
        raise ValueError(f"{task} V7 Stage 3 manifest has the wrong lineage")
    if (stage3_manifest.get("outputs") or {}).get("records.parquet") != sha256_file(records_path):
        raise ValueError(f"{task} V7 Stage 3 records hash disagrees with its manifest")

    parquet = pq.ParquetFile(records_path)
    available_columns = set(parquet.schema_arrow.names)
    columns = [field for field in _V7_COLUMNS if field in available_columns]
    rows_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_l1_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        for row in batch.to_pylist():
            in_gold_source_domain = str(row.get("source_id") or "") in GOLD_SOURCE_DOMAINS[task]
            is_skin_provenance_candidate = task == "skin_reaction" and (
                str(row.get("source_record_id") or ""),
                str(row.get("pmid") or ""),
            ) in skin_contexts_by_source_pmid
            if not in_gold_source_domain and not is_skin_provenance_candidate:
                continue
            exact_context_id = None
            if task == "bbb_martins" and row.get("source_index") is not None:
                exact_context_id = bbb_context_by_source_index.get(
                    int(row["source_index"])
                )
            elif task == "bioavailability_ma":
                claim_id = bio_source_to_claim.get(_bioavailability_source_id(row))
                exact_context_id = bio_context_by_claim.get(claim_id or "")
            elif task == "skin_reaction" and row.get("source_row_number") is not None:
                exact_context_id = skin_context_by_source_index.get(
                    (
                        str(row.get("source_id") or ""),
                        int(row["source_row_number"]) - 1,
                    )
                )
            identity = normalize_molecule_identity(str(row.get("canonical_smiles") or ""))
            parent = identity.parent_inchi_key or identity.parent_smiles
            condition = str(row.get("condition_group") or "")
            record_id = selected_keys.get((parent, condition))
            if (
                task == "skin_reaction"
                and exact_context_id is None
                and is_skin_provenance_candidate
            ):
                source_pmid = (
                    str(row.get("source_record_id") or ""),
                    str(row.get("pmid") or ""),
                )
                provenance_contexts = skin_contexts_by_source_pmid[source_pmid]
                smallest_count = min(
                    skin_gold_source_counts[value] for value in provenance_contexts
                )
                most_specific = {
                    value
                    for value in provenance_contexts
                    if skin_gold_source_counts[value] == smallest_count
                }
                exact_context_id = (
                    next(iter(most_specific))
                    if len(most_specific) == 1
                    else skin_context_by_signature.get((parent, *source_pmid))
                )
            if exact_context_id is not None:
                exact_l1_by_context[exact_context_id].append(row)
            if record_id is not None and in_gold_source_domain:
                rows_by_context[record_id].append(row)

    context_records: dict[str, dict[str, Any]] = {}
    classification_counts: Counter[str] = Counter()
    for record_id, gold in selected_gold.items():
        rows = rows_by_context.get(record_id) or []
        if not rows and not exact_l1_by_context.get(record_id):
            raise ValueError(f"{task} selected gold context has no matching V7 source rows: {record_id}")
        l1_by_id = {
            str(row["canonical_record_id"]): row
            for row in exact_l1_by_context.get(record_id, [])
        }
        l1 = list(l1_by_id.values())
        l1_ids = {str(row["canonical_record_id"]) for row in l1}
        l2 = [row for row in rows if str(row["canonical_record_id"]) not in l1_ids]
        if not l1:
            raise ValueError(f"{task} selected gold context has no current-gold constituent rows: {record_id}")
        context_id = "context_" + hashlib.sha256(
            f"{task}\0{record_id}".encode("utf-8")
        ).hexdigest()[:16]
        sampled_l1 = _stable_sample(
            l1,
            task=task,
            context_id=context_id,
            level=1,
            record_limit=record_limit,
        )
        sampled_l2 = _stable_sample(
            l2,
            task=task,
            context_id=context_id,
            level=2,
            record_limit=l2_record_limit,
        )
        context_records[record_id] = {
            "context_card_id": context_id,
            "canonical_smiles": str(gold["drug"]),
            "condition_group": str(gold["condition_group"]),
            "available_l1": len(l1),
            "available_l2": len(l2),
            "l1_cards": [_record_surface(row, level=1, context_id=context_id) for row in sampled_l1],
            "l2_cards": [_record_surface(row, level=2, context_id=context_id) for row in sampled_l2],
        }
        classification_counts.update(
            {
                "available_l1": len(l1),
                "available_l2": len(l2),
                "sampled_l1": len(sampled_l1),
                "sampled_l2": len(sampled_l2),
            }
        )

    candidates: dict[str, list[dict[str, Any]]] = {}
    for query_id, selected_rows in selected_by_query.items():
        contexts = []
        for row in sorted(selected_rows, key=lambda value: int(value["_selection_rank"])):
            record_id = str(row["retrieval_record_id"])
            score = (
                {"morgan_similarity": round(float(row["morgan_tanimoto_similarity"]), 4)}
                if ranking == "morgan"
                else {"transfer_likelihood": round(float(row["prob_transfer"]), 4)}
            )
            contexts.append(
                {
                    **context_records[record_id],
                    **score,
                    "_selection_rank": int(row["_selection_rank"]),
                    "_gold_record_id": record_id,
                    "_molecule_identity_key": str(
                        row["retrieval_molecule_identity_key"]
                    ),
                }
            )
        candidates[query_id] = contexts

    audit = {
        "protocol": PROTOCOL_VERSION,
        "task": task,
        "gold_release": release,
        "n_valid_queries": len(candidates),
        "n_unique_contexts": len(context_records),
        "contexts_per_query": context_limit,
        "selection_unit": "distinct_molecule" if distinct_molecules else "parent_condition_context",
        "ranking": ranking,
        "ranking_tie_seed": tie_seed if ranking == "morgan" else None,
        "assay_transfer_scores_used": ranking == "assay_transfer",
        "candidate_universe": "exact_v9_morgan_top100",
        "morgan_fingerprint": (
            {"radius": 2, "bits": 2048, "similarity": "Tanimoto"}
            if ranking == "morgan"
            else None
        ),
        "record_sample_limit_per_context_level": record_limit,
        "record_counts": dict(classification_counts),
        "l1_membership": (
            "active-gold source provenance only; V7 retrieval_source_id is ignored"
        ),
        "l2_membership": (
            "other V7 rows in the task gold source domain with the same normalized parent and exact condition"
        ),
        "gold_train_labels": str(gold_path),
        "gold_train_labels_sha256": sha256_file(gold_path),
        "ranking_version": str(version_path),
        "ranking_version_sha256": sha256_file(version_path),
        "rankings": str(rankings_path),
        "rankings_sha256": sha256_file(rankings_path),
        "v7_manifest": str(manifest_path),
        "v7_manifest_sha256": sha256_file(manifest_path),
        "v7_records": str(records_path),
        "v7_records_sha256": sha256_file(records_path),
        "bioavailability_direct_claims": bio_claim_audit,
    }
    if l2_record_limit != record_limit:
        audit["l2_record_sample_limit_per_context"] = l2_record_limit
    return candidates, audit


def _visible_source_values(values: Mapping[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 2) if math.isfinite(value) else None
        if isinstance(value, list):
            return [item for item in (clean(item) for item in value) if item is not None]
        if isinstance(value, Mapping):
            return {
                str(key): item
                for key, raw in value.items()
                if (item := clean(raw)) not in (None, "", [], {})
            }
        return value

    return clean(values)


def build_level_record_bundles(
    task: str,
    ranked_records: Mapping[str, Mapping[str, Any]],
    *,
    record_limit: int | Mapping[str, int] = INDIRECT_RECORD_LIMIT,
    prompt_version: str = "v1",
    level_names: Sequence[str] | None = None,
    transfer_model: str = "V19.1",
) -> dict[int, dict[str, Any]]:
    """Convert frozen cache rows into one model-visible bundle per level."""
    contract = level_record_bundle_contract(prompt_version)
    score_field = _score_field(prompt_version)
    try:
        task_levels = contract["task_levels"][task]
    except KeyError as exc:
        raise ValueError(f"cache-backed later levels are unsupported for {task}") from exc
    bundles: dict[int, dict[str, Any]] = {}
    selected_levels = tuple(level_names or indirect_level_names(task))
    if not set(selected_levels) <= set(task_levels):
        raise ValueError(f"{task} lacks cache-backed levels {selected_levels}")
    for level_name in selected_levels:
        level = int(level_name[1:])
        level_record_limit = int(
            record_limit.get(level_name, 0)
            if isinstance(record_limit, Mapping)
            else record_limit
        )
        if level_record_limit < 1:
            raise ValueError(f"record limit for {level_name} must be positive")
        selection = ranked_records.get(level_name)
        if not isinstance(selection, Mapping):
            raise ValueError(f"{task} query lacks cache-backed {level_name} records")
        rows = list(selection.get("records") or [])
        if len(rows) != level_record_limit and not (selection.get("allow_shortfall") and len(rows) < level_record_limit):
            raise ValueError(
                f"{task} {level_name} requires exactly {level_record_limit} records"
            )
        definition = task_levels[level_name]
        cards: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(rows, start=1):
            payload = row.get("payload") or {}
            source_values = payload.get("source_fields") or {}
            if task == "bbb_martins":
                source_values = semantic_prompt_payload(payload, source_values)
            record_id = str(row.get("record_id") or "")
            if not record_id:
                raise ValueError(f"{task} {level_name} cache row lacks record_id")
            card_id = "record_" + hashlib.sha256(
                f"{INDIRECT_PROTOCOL_VERSION}\0{task}\0{level_name}\0{record_id}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            cards[card_id] = {
                "card_id": card_id,
                "first_seen_level": level,
                "level": level_name,
                "reference_smiles": str(payload.get("canonical_smiles") or ""),
                score_field: round(float(row[score_field]), 2),
                "source_id": str(payload.get("source_id") or ""),
                "measurement_kind": str(payload.get("measurement_kind") or ""),
                "source_values": _visible_source_values(
                    source_values
                ),
                "_canonical_record_id": record_id,
                "_reference_molecule_id": str(
                    row.get("reference_molecule_id") or ""
                ),
                "_selection_rank": rank,
            }
        bundle_id = f"{task}_{level_name.lower()}_record_bundle"
        bundles[level] = {
            "_card_kind": "level_record_bundle",
            "level_card_id": bundle_id,
            "level": level_name,
            "evidence_family": str(definition["evidence_family"]),
            "description": str(definition["description"]),
            "selection": (
                f"top {level_record_limit} Stage 3 records by "
                + (
                    "Morgan fingerprint similarity"
                    if score_field == "morgan_similarity"
                    else f"frozen {transfer_model} record-transfer likelihood"
                )
            ),
            "available_record_count": int(selection["available_record_count"]),
            "selected_record_count": len(cards),
            "first_seen_level": level,
            "_selection_rank": 100 + level,
            "cards": cards,
        }
    return bundles


def _aligned_analog_id(
    smiles: str, identity_key: str | None = None
) -> tuple[str, str]:
    if identity_key:
        return (
            "analog_" + hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:12],
            smiles,
        )
    identity = normalize_molecule_identity(smiles)
    parent = identity.parent_smiles or smiles
    key = identity.parent_inchi_key or parent
    if not key:
        raise ValueError("aligned molecule card lacks a normalized identity")
    return "analog_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], parent


def _aligned_context_card(
    source: Mapping[str, Any], card: Mapping[str, Any], *, rank: int
) -> dict[str, Any]:
    level = int(card["first_seen_level"])
    source_values = {
        key.removeprefix("source__"): value
        for key, value in card.items()
        if key.startswith("source__") and value not in (None, "", [], {})
    }
    condition = str(source.get("condition_group") or "")
    qualifying = _first(card, "qualifying_conditions", "source__qualifying_conditions")
    if not qualifying and condition != "no_reported_external_condition":
        qualifying = condition
    score_field = (
        "morgan_similarity"
        if source.get("morgan_similarity") is not None
        else "transfer_likelihood"
    )
    return {
        "card_id": str(card["card_id"]),
        "evidence_family": (f"mapped_L{level}_records" if source.get("_level_mapping") else
            "current_gold_voting_records"
            if level == 1
            else "associated_nonvoting_records"
        ),
        "assay_context": _first(card, "canonical_assay_context", "assay_context"),
        "endpoint": _first(card, "endpoint", "canonical_endpoint_name"),
        "reported_value": _first(card, "result", "canonical_measurement_text"),
        "reported_unit": _first(card, "unit", "canonical_unit_text"),
        "species": _first(card, "canonical_species_context", "source__species"),
        "qualifying_conditions": qualifying,
        "support_text": _first(card, "support_text", "source__support_text"),
        "source_id": _first(card, "source_id"),
        "measurement_kind": _first(card, "measurement_kind"),
        "source_values": _visible_source_values(source_values),
        score_field: round(float(source[score_field]), 4),
        "first_seen_level": level,
        "_selection_rank": rank,
        "_canonical_record_id": str(card["_canonical_record_id"]),
    }


def _aligned_later_card(
    task: str,
    level_name: str,
    definition: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    rank: int,
    score_field: str,
) -> dict[str, Any]:
    payload = row.get("payload") or {}
    source_values = payload.get("source_fields") or {}
    if task == "bbb_martins" and payload.get('source_projection') != 'library_source_contract.v2':
        source_values = semantic_prompt_payload(payload, source_values)
    record_id = str(row.get("record_id") or "")
    if not record_id:
        raise ValueError(f"{task} {level_name} cache row lacks record_id")
    level = int(level_name[1:])
    return {
        "card_id": "record_" + hashlib.sha256(
            f"{TIANANG_ALIGNED_PROTOCOL_VERSION}\0{task}\0{record_id}".encode("utf-8")
        ).hexdigest()[:16],
        "evidence_family": str(definition["evidence_family"]),
        "assay_context": _first(
            source_values,
            "assay_context",
            "assay_description",
            "assay_model",
            "assay_system",
        ),
        "endpoint": _first(source_values, "endpoint_name", "standard_type", "measurement_type"),
        "reported_value": _first(
            source_values,
            "measurement_text",
            "reported_value",
            "standard_value",
            "value",
        ),
        "reported_unit": _first(source_values, "unit_text", "standard_units", "unit"),
        "species": _first(source_values, "species", "organism", "species_or_population"),
        "qualifying_conditions": _first(
            source_values,
            "qualifying_conditions",
            "experimental_conditions",
            "study_context",
        ),
        "support_text": _first(source_values, "support_text", "description"),
        "source_id": str(payload.get("source_id") or ""),
        "measurement_kind": str(payload.get("measurement_kind") or ""),
        "source_values": _visible_source_values(source_values),
        score_field: round(float(row[score_field]), 4),
        "first_seen_level": level,
        "_selection_rank": level * 1000 + rank,
        "_canonical_record_id": record_id,
        "_semantic_bucket_id": row.get("_semantic_bucket_id"),
        "_semantic_rank": row.get("_semantic_rank"),
        "_reference_molecule_id": str(row.get("reference_molecule_id") or ""),
    }


def stage_ranked_snapshots(
    molecules, *, task, records_by_level, prompt_version,
    molecule_descriptions: Mapping[str, str] | None = None,
):
    """Append independent level selections to parent cards; retain record conditions."""
    definitions = tianang_aligned_card_contract(prompt_version)['task_levels'][task]
    active, snapshots, seen = {}, {}, set()
    levels = ({'L1': [r for m in molecules for r in m['l1_records']]}
              if molecules else {})
    levels.update({k: v['records'] for k, v in records_by_level.items()})
    introductions = {
        _aligned_analog_id(
            m['canonical_smiles'], m.get('context_card_id') or m.get('reference_molecule_id')
        )[0]: m
        for m in molecules
    }
    for level_name, rows in levels.items():
        level = int(level_name[1:])
        for rank, row in enumerate(rows):
            record_id = row['record_id']
            if record_id in seen:
                raise ValueError(f'Duplicate physical record: {record_id}')
            seen.add(record_id)
            semantic_bucket = row.get('semantic_bucket_id') if level > 1 else None
            reference_smiles = str(row['payload']['canonical_smiles'])
            if semantic_bucket:
                analog_id = 'semantic_' + hashlib.sha256(
                    str(semantic_bucket).encode('utf-8')
                ).hexdigest()[:12]
                smiles = 'grouped_evidence_records'
            else:
                analog_id, smiles = _aligned_analog_id(
                    reference_smiles,
                    row.get('context_card_id') or row.get('reference_molecule_id'),
                )
            if analog_id not in active:
                source = introductions.get(analog_id, row) if level == 1 else row
                molecule = dict(analog_id=analog_id, canonical_smiles=smiles,
                    morgan_similarity=round(float(row['morgan_similarity']), 4), first_seen_level=level,
                    group_kind=('semantic_bucket' if semantic_bucket else
                                'l1_context' if level == 1 else 'parent_molecule'),
                    _selection_rank=len(active), cards={})
                for key in (
                    'transfer_likelihood',
                    'selected_condition',
                    'morgan_top5_rank', 'assay_transfer_top5_rank',
                    'morgan_panel_rank', 'assay_transfer_panel_rank',
                    '_diagnostic_label', '_diagnostic_context_id',
                    '_diagnostic_parent_id',
                ):
                    if key in source:
                        molecule[key] = (round(float(source[key]), 4)
                            if key == 'transfer_likelihood' else source[key])
                if 'transfer_likelihood' in molecule:
                    molecule['transfer_score_level'] = level
                if molecule_descriptions is not None and not semantic_bucket:
                    description = molecule_descriptions.get(reference_smiles)
                    if description is not None:
                        molecule['molecule_description'] = description
                active[analog_id] = molecule
            field = 'transfer_likelihood' if 'transfer_likelihood' in row else 'morgan_similarity'
            card = _aligned_later_card(task, level_name, definitions[level_name], row,
                                       rank=rank, score_field=field)
            # The cache reader already applies each source's approved-column contract.
            # Keep experimental columns across schemas, not extraction metadata or opaque extras.
            metadata_fields = {
                'source_id', 'measurement_kind', 'source_index', 'source_record_id',
                'source_row_uid', 'extraction_id', 'global_identifier', 'paragraph_idx',
                'pmid', 'smiles', 'source_smiles', 'molecule_name', 'extra_details',
                'confidence', 'needs_more_context',
            }
            card['experimental_details'] = {
                key: value for key, value in card['source_values'].items()
                if key not in metadata_fields and value not in (None, '', [], {})
                and value not in [card[k] for k in ('assay_context', 'endpoint', 'reported_value',
                    'reported_unit', 'species', 'qualifying_conditions', 'support_text')]
            }
            if not card['experimental_details']:
                card.pop('experimental_details')
            if level > 1:
                card['reference_smiles'] = reference_smiles
                if molecule_descriptions is not None:
                    description = molecule_descriptions.get(reference_smiles)
                    if description is not None:
                        card['reference_molecule_description'] = description
            else:
                card.pop('morgan_similarity', None)
            card['retrieved_by'] = row['ranking_method']
            # These cards have no molecule-wide condition header: each record owns its conditions.
            active[analog_id]['cards'][card['card_id']] = card
        snapshots[level] = deepcopy(active)
    return snapshots


def tianang_aligned_snapshots(
    contexts: list[Mapping[str, Any]],
    *,
    task: str,
    indirect_records: Mapping[str, Mapping[str, Any]],
    prompt_version: str,
    indirect_record_limit: int | Mapping[str, int] = INDIRECT_RECORD_LIMIT,
    record_levels: Sequence[str] | None = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Present record-first selections as append-only molecule cards."""
    if not is_tianang_aligned(prompt_version):
        raise ValueError("Tianang-aligned snapshots require an aligned prompt profile")
    score_field = _score_field(prompt_version)
    output: dict[int, dict[str, dict[str, Any]]] = {}
    active: dict[str, dict[str, Any]] = {}
    seen_records: set[str] = set()
    for level in (1, 2):
        active = deepcopy(active)
        for source in contexts:
            analog_id, parent_smiles = _aligned_analog_id(str(source["canonical_smiles"]))
            analog = active.setdefault(
                analog_id,
                {
                    "analog_id": analog_id,
                    "canonical_smiles": parent_smiles,
                    score_field: round(float(source[score_field]), 4),
                    "selected_condition": (
                        source["condition_group"]
                        if source.get("condition_group") != "no_reported_external_condition"
                        else ""
                    ),
                    "first_seen_level": 1,
                    "_selection_rank": int(source["_selection_rank"]),
                    "cards": {},
                },
            )
            cards = source["l1_cards"] if level == 1 else source["l2_cards"]
            for index, card in enumerate(cards):
                record_id = str(card["_canonical_record_id"])
                if record_id in seen_records:
                    continue
                seen_records.add(record_id)
                projected = _aligned_context_card(
                    source,
                    card,
                    rank=level * 10000 + int(source["_selection_rank"]) * 100 + index,
                )
                analog["cards"][projected["card_id"]] = projected
        output[level] = deepcopy(active)

    contract = tianang_aligned_card_contract(prompt_version)
    selected_levels = (
        tuple(indirect_level_names(task))
        if record_levels is None
        else tuple(record_levels)
    )
    for level_name in selected_levels:
        level = int(level_name[1:])
        limit = int(
            indirect_record_limit.get(level_name, 0)
            if isinstance(indirect_record_limit, Mapping)
            else indirect_record_limit
        )
        selection = indirect_records.get(level_name)
        if not isinstance(selection, Mapping):
            raise ValueError(f"{task} query lacks cache-backed {level_name} records")
        rows = list(selection.get("records") or [])
        if len(rows) != limit and not (selection.get("allow_shortfall") and len(rows) < limit):
            raise ValueError(f"{task} {level_name} requires exactly {limit} records")
        active = deepcopy(active)
        for rank, row in enumerate(rows, start=1):
            payload = row.get("payload") or {}
            analog_id, parent_smiles = _aligned_analog_id(
                str(payload.get("canonical_smiles") or "")
            )
            analog = active.setdefault(
                analog_id,
                {
                    "analog_id": analog_id,
                    "canonical_smiles": parent_smiles,
                    score_field: round(float(row[score_field]), 4),
                    "first_seen_level": level,
                    "_selection_rank": level * 1000 + rank,
                    "cards": {},
                },
            )
            analog[score_field] = max(
                float(analog.get(score_field) or 0.0), float(row[score_field])
            )
            card = _aligned_later_card(
                task,
                level_name,
                contract["task_levels"][task][level_name],
                row,
                rank=rank,
                score_field=score_field,
            )
            record_id = str(card["_canonical_record_id"])
            if record_id in seen_records:
                raise ValueError(f"duplicate physical record in aligned evidence: {record_id}")
            seen_records.add(record_id)
            if card["card_id"] in analog["cards"]:
                raise ValueError(f"duplicate aligned card ID: {card['card_id']}")
            analog["cards"][card["card_id"]] = card
        output[level] = deepcopy(active)
    return output


def stage_score_view(
    active, policy, *, sampling_only=False, precision=2, show_both=False
):
    """Project stage-scoped scores without changing retrieval or stored tool receipts."""
    from predict.harnesses.progressive._visibility import similarity_view
    view = {}
    l1_method = policy.get('L1')
    for analog_id, molecule in active.items():
        cards = list(molecule.get('cards', {}).values())
        morgan_levels = sorted({int(c['first_seen_level']) for c in cards
                                if show_both or policy[f"L{int(c['first_seen_level'])}"] in {
                                    'morgan', 'morgan_contrastive', 'joint',
                                    'semantic_lap', 'semantic_weighted',
                                    'morgan_parent_control', 'morgan_parent_semantic',
                                    'morgan_parent_llm_semantic'
                                }})
        projected = similarity_view(
            {analog_id: molecule}, hidden=not morgan_levels, precision=precision
        )[analog_id]
        if morgan_levels:
            projected['morgan_score_levels'] = morgan_levels
        else:
            projected.pop('morgan_score_levels', None)
        # Only L1 scores describe a gold context; later scores belong to records.
        if (not show_both and (sampling_only or int(molecule['first_seen_level']) != 1
                or l1_method in {'morgan', 'morgan_contrastive'})):
            projected.pop('transfer_likelihood', None)
            projected.pop('transfer_score_level', None)
        if sampling_only or int(molecule['first_seen_level']) != 1 or l1_method != 'joint':
            projected.pop('morgan_top5_rank', None)
            projected.pop('assay_transfer_top5_rank', None)
            projected.pop('morgan_panel_rank', None)
            projected.pop('assay_transfer_panel_rank', None)
        for card in projected.get('cards', {}).values():
            level = int(card['first_seen_level'])
            allowed = ({'joint', 'morgan', 'assay_transfer'}
                       if level == 1 and l1_method == 'joint'
                       else {policy[f'L{level}']})
            if card.get('retrieved_by') not in allowed:
                raise ValueError('Evidence record disagrees with stage retrieval policy')
            if (sampling_only or level == 1
                    or policy[f'L{level}'] in {
                        'morgan', 'morgan_contrastive', 'semantic_lap',
                        'semantic_weighted', 'morgan_parent_control',
                        'morgan_parent_semantic', 'morgan_parent_llm_semantic'
                    }):
                card.pop('transfer_likelihood', None)
        view[analog_id] = projected
    return view


def build_tianang_aligned_messages(
    *,
    contract: ProgressiveTaskContract,
    levels: list[Mapping[str, Any]],
    current_level: int,
    query_smiles: str,
    query_molecule_description: str | None = None,
    condition_sentence: str,
    query_prior: Mapping[str, Any] | None,
    query_tool_summary: Mapping[str, Any] | None,
    active: Mapping[str, Mapping[str, Any]],
    prior_state: Mapping[str, Any] | None,
    prompt_version: str,
    record_limit: int,
    l2_record_limit: int,
    indirect_record_limit: int | Mapping[str, int],
    retrieval_policy: Mapping[str, str] | None = None,
    molecule_limit: int = CONTEXT_LIMIT,
    return_reference_index: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    assets = prompt_assets(prompt_version)
    prompt_text = dict(assets['user_shared'])
    mode = None
    if assets['modes']:
        l1_method = (retrieval_policy or {}).get('L1', 'morgan')
        mode = {
            'morgan': 'morgan',
            'morgan_contrastive': 'morgan',
            'assay_transfer': 'assay-transfer',
            'assay_transfer_within_morgan': 'paired-order',
            'assay_transfer_contrastive': 'assay-transfer-contrastive',
            'joint': 'joint',
        }.get(l1_method)
        if ('L1' in (retrieval_policy or {})
                and 'paired-order' in assets['modes'] and l1_method == 'morgan'):
            mode = 'paired-order'
        if mode not in assets['modes']:
            raise ValueError('Mode-specific prompt requires a valid L1 retrieval method')
        for level, method in retrieval_policy.items():
            expected = ('joint' if mode == 'joint' and level == 'L1' else
                        'morgan' if mode == 'morgan' or level == 'L5' else 'assay_transfer')
            if mode == 'morgan' and level == 'L2' and method == 'semantic_lap':
                expected = 'semantic_lap'
            if mode == 'morgan' and level == 'L2' and method == 'semantic_weighted':
                expected = 'semantic_weighted'
            if mode == 'morgan' and level in {'L2', 'L3', 'L4'} and method in {
                'morgan_parent_control', 'morgan_parent_semantic',
                'morgan_parent_llm_semantic'
            }:
                expected = method
            if mode == 'paired-order' and level == 'L1':
                expected = method if method in {
                    'morgan', 'assay_transfer_within_morgan'
                } else None
            if mode == 'paired-order' and level == 'L2' and method in {
                'morgan_parent_control', 'morgan_parent_semantic',
                'morgan_parent_llm_semantic'
            }:
                expected = method
            if mode == 'morgan' and level == 'L1' and method == 'morgan_contrastive':
                expected = 'morgan_contrastive'
            if mode == 'assay-transfer-contrastive' and level == 'L1':
                expected = 'assay_transfer_contrastive'
            if method != expected:
                raise ValueError(f'Prompt mode {mode} disagrees with retrieval at {level}')
        prompt_text.update(assets['modes'][mode])
    if not is_tianang_aligned(prompt_version):
        raise ValueError(prompt_text['tianang_aligned_messages'])
    visible_max_level = max(int(row["level"]) for row in levels)
    stage_ranked = prompt_profile(prompt_version).get('stage_ranked')
    if stage_ranked:
        expected_levels = [f"L{row['level']}" for row in levels]
        if not retrieval_policy or list(retrieval_policy) != expected_levels:
            raise ValueError('Stage-ranked rendering requires the complete ordered retrieval policy')
        for level, method in retrieval_policy.items():
            if method not in {
                'morgan', 'morgan_contrastive', 'assay_transfer',
                'assay_transfer_within_morgan',
                'assay_transfer_contrastive', 'joint', 'semantic_lap',
                'semantic_weighted', 'morgan_parent_control',
                'morgan_parent_semantic', 'morgan_parent_llm_semantic'
            } or (method in {
                'joint', 'morgan_contrastive', 'assay_transfer_within_morgan',
                'assay_transfer_contrastive'
            } and level != 'L1') or (
                method in {
                    'semantic_lap', 'semantic_weighted',
                    'morgan_parent_control', 'morgan_parent_semantic',
                    'morgan_parent_llm_semantic'
                } and level != 'L2'
                and not ('L1' not in retrieval_policy and level in {'L3', 'L4'})
            ):
                raise ValueError(f'Invalid stage ranking: {level}={method}')
    score_precision = 4
    if mode:
        score_visibility = assets['settings'].get('retrieval_scores')
        active = stage_score_view(
            active,
            retrieval_policy,
            precision=score_precision,
            show_both=(
                score_visibility == 'both'
                or score_visibility == 'mode_specific'
                and retrieval_policy.get('L1', 'morgan') not in {'morgan', 'morgan_contrastive'}
            ),
        )
    display = prompt_profile(prompt_version).get('similarity_display')
    if display:
        from predict.harnesses.progressive._visibility import similarity_view
        if not stage_ranked and (current_level != 1 or visible_max_level != 1):
            raise ValueError('visibility/joint variants are L1-only')
        active = similarity_view(
            active, hidden=display == 'hidden', precision=score_precision
        )
    messages = build_progressive_messages(
        contract=contract,
        levels=levels,
        current_level=current_level,
        query_smiles=query_smiles,
        query_molecule_description=query_molecule_description,
        condition_sentence=condition_sentence,
        query_prior=query_prior,
        query_tool_summary=query_tool_summary,
        active=active,
        prior_state=prior_state,
        protocol_version=TIANANG_ALIGNED_PROTOCOL_VERSION,
        card_contract=tianang_aligned_card_contract(prompt_version),
        prompt_template=f"prompts/{prompt_version}/"
        + prompt_profile(prompt_version)["template"].name,
        prompt_version=prompt_version,
        protocol_details={
            'retrieval_by_level': dict(retrieval_policy), 'prompt_mode': mode,
        },
    )
    if (display or mode) and not assets['settings'].get('user_template'):
        from predict.harnesses.progressive._visibility import validate_visible_messages
        validate_visible_messages(
            messages, hidden=display == 'hidden', stage_scoped=bool(mode)
        )
    reference_index = None
    reference_contract = assets['settings'].get('reasoning_reference_contract')
    if reference_contract:
        card_id_to_alias, _ = card_alias_maps(active)
        reference_index = build_reference_index(
            active,
            card_id_to_alias=card_id_to_alias,
            current_level=current_level,
            layout=str(reference_contract['layout']),
        )
        validate_unique_visible_ids(reference_index)
        validate_prompt_index(messages[1]['content'], reference_index)
    return (messages, reference_index) if return_reference_index else messages


def snapshots(
    contexts: list[Mapping[str, Any]],
    *,
    task: str | None = None,
    indirect_records: Mapping[str, Mapping[str, Any]] | None = None,
    indirect_record_limit: int | Mapping[str, int] = INDIRECT_RECORD_LIMIT,
    prompt_version: str = "v1",
    record_levels: Sequence[str] | None = None,
    transfer_model: str = "V19.1",
) -> dict[int, dict[str, dict[str, Any]]]:
    """Return append-only context cards, optionally followed by later-level bundles."""
    output: dict[int, dict[str, dict[str, Any]]] = {}
    score_field = _score_field(prompt_version)
    record_only = (
        bool(record_levels)
        and min(int(name[1:]) for name in record_levels) == 1
    )
    for level in (() if record_only else (1, 2)):
        active: dict[str, dict[str, Any]] = {}
        for source in contexts:
            cards = list(source["l1_cards"])
            if level == 2:
                cards += list(source["l2_cards"])
            context_id = str(source["context_card_id"])
            counts = {"L1": int(source["available_l1"])}
            if level == 2:
                counts["L2"] = int(source["available_l2"])
            active[context_id] = {
                "context_card_id": context_id,
                "canonical_smiles": source["canonical_smiles"],
                "condition_group": source["condition_group"],
                score_field: source[score_field],
                "available_record_counts": counts,
                "first_seen_level": 1,
                "_selection_rank": source["_selection_rank"],
                "_gold_record_id": source["_gold_record_id"],
                "_context_origin": source.get("_context_origin"),
                "_reference_molecule_id": source.get("_reference_molecule_id"),
                "cards": {str(card["card_id"]): dict(card) for card in cards},
            }
        output[level] = active
    if indirect_records is not None:
        if task is None:
            raise ValueError("task is required with cache-backed later-level records")
        active = dict(output[max(output)]) if output else {}
        for level, bundle in build_level_record_bundles(
            task,
            indirect_records,
            record_limit=indirect_record_limit,
            prompt_version=prompt_version,
            level_names=record_levels,
            transfer_model=transfer_model,
        ).items():
            active = dict(active)
            active[str(bundle["level_card_id"])] = bundle
            output[level] = active
    return output


def render_active_evidence(
    active: Mapping[str, Mapping[str, Any]],
    *,
    card_id_to_alias: Mapping[str, str],
    prompt_version: str = "v1",
    compact: bool = False,
) -> list[dict[str, Any]]:
    contract = card_contract(prompt_version)
    bundle_contract = level_record_bundle_contract(prompt_version)
    score_field = _score_field(prompt_version)
    rendered = []
    for context in sorted(active.values(), key=lambda row: int(row["_selection_rank"])):
        if context.get("_card_kind") == "level_record_bundle":
            public = _project(context, bundle_contract["bundle"]["fields"])
            cards = sorted(
                (context.get("cards") or {}).values(),
                key=lambda row: int(row["_selection_rank"]),
            )
            molecule_ids: dict[str, str] = {}
            source_columns: dict[str, list[str]] = {}
            for card in cards:
                smiles = str(card["reference_smiles"])
                molecule_ids.setdefault(smiles, f"M{len(molecule_ids) + 1:02d}")
                columns = source_columns.setdefault(str(card["source_id"]), [])
                for name in card["source_values"]:
                    if name not in columns:
                        columns.append(name)
            source_ids = {
                source: f"S{index:02d}"
                for index, source in enumerate(source_columns, start=1)
            }
            molecules = [
                _project(
                    {"molecule_id": molecule_id, "smiles": smiles},
                    bundle_contract["molecule"]["fields"],
                )
                for smiles, molecule_id in molecule_ids.items()
            ]
            schemas = [
                _project(
                    {
                        "schema_id": source_ids[source],
                        "source_id": source,
                        "source_columns": columns,
                    },
                    bundle_contract["source_schema"]["fields"],
                )
                for source, columns in source_columns.items()
            ]
            rows = []
            for card in cards:
                source = str(card["source_id"])
                values = card["source_values"]
                item = {
                    "card_id": card_id_to_alias[str(card["card_id"])],
                    "reference_molecule_id": molecule_ids[str(card["reference_smiles"])],
                    score_field: card[score_field],
                    "source_schema_id": source_ids[source],
                    "measurement_kind": card.get("measurement_kind"),
                    "source_values": [values.get(name) for name in source_columns[source]],
                }
                rows.append([
                    _project(item, [field]).get(str(field["name"]))
                    for field in bundle_contract["record"]["fields"]
                ])
            public[str(bundle_contract["bundle"]["molecules_field"])] = molecules
            public[str(bundle_contract["bundle"]["source_schemas_field"])] = schemas
            public["record_columns"] = [
                str(field["name"]) for field in bundle_contract["record"]["fields"]
            ]
            public[str(bundle_contract["bundle"]["records_field"])] = rows
            rendered.append(public)
            continue
        public = _project(context, contract["context"]["fields"])
        cards = []
        for card in sorted(
            (context.get("cards") or {}).values(),
            key=lambda row: (int(row["first_seen_level"]), str(row["card_id"])),
        ):
            item = dict(card)
            item["card_id"] = card_id_to_alias[str(card["card_id"])]
            cards.append(_project(item, contract["record"]["fields"]))
        if compact:
            columns = list(dict.fromkeys(name for card in cards for name in card))
            transport = bundle_contract["cumulative_context_records"]
            public[str(transport["columns_field"])] = columns
            public[str(transport["rows_field"])] = [
                [card.get(name) for name in columns] for card in cards
            ]
        else:
            public[str(contract["context"]["records_field"])] = cards
        rendered.append(public)
    return rendered


def build_messages(
    *,
    contract: ProgressiveTaskContract,
    current_level: int,
    query_smiles: str,
    condition_sentence: str,
    query_prior: Mapping[str, Any] | None,
    query_tool_summary: Mapping[str, Any] | None,
    active: Mapping[str, Mapping[str, Any]],
    prior_state: Mapping[str, Any] | None,
    prompt_version: str = "v1",
    record_limit: int = RECORD_LIMIT,
    l2_record_limit: int | None = None,
    indirect_record_limit: int | Mapping[str, int] = INDIRECT_RECORD_LIMIT,
    include_indirect: bool = False,
) -> list[dict[str, Any]]:
    prompt_text = prompt_assets(prompt_version)["user_shared"]
    if l2_record_limit is None:
        l2_record_limit = record_limit
    if record_limit < 1:
        raise ValueError("record_limit must be positive")
    if l2_record_limit < 1:
        raise ValueError("l2_record_limit must be positive")
    if isinstance(indirect_record_limit, Mapping):
        if not indirect_record_limit or any(
            int(value) < 1 for value in indirect_record_limit.values()
        ):
            raise ValueError(prompt_text['messages'])
    elif indirect_record_limit < 1:
        raise ValueError(prompt_text['messages_2'])
    bundle_levels = tuple(
        str(context["level"])
        for context in active.values()
        if context.get("_card_kind") == "level_record_bundle"
    )
    indirect_visible = include_indirect and bool(bundle_levels)
    score_field = _score_field(prompt_version)
    if (
        l2_record_limit == record_limit
        and indirect_record_limit == INDIRECT_RECORD_LIMIT
    ):
        record_limit_label = "ten" if record_limit == 10 else str(record_limit)
        record_sampling = (
            prompt_text['messages_3'].format(record_limit_label=record_limit_label)
        )
    else:
        if indirect_visible and isinstance(indirect_record_limit, Mapping):
            limit_summary = ", ".join(
                f"{level}={int(limit)}"
                for level, limit in indirect_record_limit.items()
            )
            later_sampling = (
                prompt_text['messages_4'].format(limit_summary=limit_summary)
            )
        else:
            later_sampling = (
                prompt_text['messages_5'].format(indirect_record_limit=indirect_record_limit)
                if indirect_visible
                else ""
            )
        record_sampling = (
            prompt_text['messages_6'].format(record_limit=record_limit, l2_record_limit=l2_record_limit, later_sampling=later_sampling)
        )
    card_id_to_alias, _ = card_alias_maps(active)
    context_count = sum(
        context.get("_card_kind") != "level_record_bundle"
        for context in active.values()
    )
    record_only = indirect_visible and context_count == 0
    v21_molecule_cards = context_count > 0 and all(
        context.get("_context_origin") == "v21_record_ranked_molecule"
        for context in active.values()
        if context.get("_card_kind") != "level_record_bundle"
    )
    if v21_molecule_cards:
        later_sampling = (
            f" Later-level record caps are {', '.join(f'{level}={int(limit)}' for level, limit in indirect_record_limit.items())}."
            if indirect_visible and isinstance(indirect_record_limit, Mapping)
            else (
                prompt_text['messages_7'].format(indirect_record_limit=indirect_record_limit)
                if indirect_visible
                else ""
            )
        )
        record_sampling = (
            prompt_text['messages_8'].format(record_limit=record_limit, l2_record_limit=l2_record_limit, later_sampling=later_sampling)
        )
    if record_only:
        limits = (
            {str(level): int(limit) for level, limit in indirect_record_limit.items()}
            if isinstance(indirect_record_limit, Mapping)
            else {level: indirect_record_limit for level in bundle_levels}
        )
        limit_summary = ", ".join(
            f"{level}={limits[level]}" for level in bundle_levels
        )
        record_sampling = (
            prompt_text['messages_9'].format(limit_summary=limit_summary)
        )
    new_ids = sorted(
        card_id_to_alias[str(card["card_id"])]
        for context in active.values()
        for card in (context.get("cards") or {}).values()
        if int(card.get("first_seen_level") or 0) == current_level
    )
    is_initial = prior_state is None
    indirect_label = (
        "/".join(
            record_level_names(contract.task, first_level=1)
            if record_only
            else indirect_level_names(contract.task)
        )
        if indirect_visible
        else ""
    )
    protocol_version = INDIRECT_PROTOCOL_VERSION if indirect_visible else PROTOCOL_VERSION
    schema = {
        contract.prediction_field: f"{contract.positive_prediction} | {contract.negative_prediction}",
        "confidence": "high | moderate | low",
        "revision_action": "initial" if is_initial else "keep | strengthen | weaken | flip",
        "supportive_card_ids": ["C01"],
        "contradictory_card_ids": ["C02"],
        "prediction_basis_card_ids": ["C01"],
        "claims": [{"claim": "concise source-grounded statement", "card_ids": ["C01"]}],
        "new_evidence_assessment": [
            {
                "family": "current level name",
                "applicability": prompt_text['applicability'],
                "direction": prompt_text['direction'],
                "decision_effect": prompt_text['decision_effect'],
                "card_ids": ["new card alias"],
            }
        ],
        "evidence_gaps": ["string"],
        "decision_summary": "concise string",
    }
    payload: dict[str, Any] = {
        "protocol": {
            "version": protocol_version,
            "mode": "initial decision" if is_initial else "progressive update",
            "architecture": (
                prompt_text['architecture'].format(indirect_label=indirect_label)
                if record_only
                else (
                    prompt_text['architecture_2'].format(context_count=context_count)
                    + (
                        prompt_text['architecture_3'].format(indirect_label=indirect_label)
                        if indirect_visible
                        else ""
                    )
                    if v21_molecule_cards
                    else
                    prompt_text['architecture_4'].format(context_count=context_count)
                    + (
                        prompt_text['architecture_5'].format(indirect_label=indirect_label)
                        if indirect_visible
                        else ""
                    )
                )
            ),
            "record_sampling": record_sampling,
            "level_interpretation": (
                prompt_text['level_interpretation']
                if record_only
                else (
                    prompt_text['level_interpretation_2']
                    + (
                        prompt_text['level_interpretation_3'].format(indirect_label=indirect_label)
                        if indirect_visible
                        else ""
                    )
                )
            ),
            "transfer_rule": (
                prompt_text['transfer_rule']
                if score_field == "morgan_similarity"
                else (
                    prompt_text['transfer_rule_2']
                    if v21_molecule_cards
                    else prompt_text['transfer_rule_3'].format(indirect_label=indirect_label)
                    if indirect_visible
                    else prompt_text['transfer_rule_4']
                )
            ),
            "structure_rule": (
                prompt_text['structure_rule']
                if score_field == "morgan_similarity"
                else prompt_text['structure_rule_2']
            ),
            "prior_rule": (
                prompt_text['prior_rule']
            ),
            "citation_rule": (
                prompt_text['citation_rule']
            ),
            "flip_rule": (
                prompt_text['flip_rule']
            ),
        },
        "task_definition": {
            "task": contract.task,
            "endpoint": contract.endpoint_name,
            "label_scope": contract.label_scope,
            "prediction_values": {
                contract.positive_prediction: "positive class (label 1)",
                contract.negative_prediction: "negative class (label 0)",
            },
            "instructions": _task_instructions(
                contract, structure_score_visible=score_field == "morgan_similarity"
            ),
        },
        "level_context": {
            "current_level": current_level,
            "current_family": levels(
                current_level,
                task=contract.task,
                include_indirect=indirect_visible,
                prompt_version=prompt_version,
            )[-1],
            "full_level_plan": levels(
                task=contract.task,
                include_indirect=indirect_visible,
                prompt_version=prompt_version,
            ),
            "new_card_ids": new_ids,
        },
        "query": {"canonical_smiles": query_smiles},
        "active_evidence": render_active_evidence(
            active,
            card_id_to_alias=card_id_to_alias,
            prompt_version=prompt_version,
            compact=indirect_visible,
        ),
        "required_json_schema": schema,
    }
    if condition_sentence:
        payload["query"]["external_condition"] = condition_sentence
    if query_prior:
        payload["query_prior"] = dict(query_prior)
    if query_tool_summary:
        payload["query"]["molecule_property_tool_summary"] = dict(query_tool_summary)
    if prior_state is not None:
        payload["prior_state"] = render_prior_state(
            prior_state, card_id_to_alias=card_id_to_alias
        )
    system_role = contract.system_role
    if indirect_visible:
        definition = level_record_bundle_contract(prompt_version)["task_levels"][contract.task].get(
            f"L{current_level}"
        )
        if definition:
            system_role += "\n\n" + str(definition["guidance"])
    return render_progressive_messages(
        system_role=system_role,
        payload=payload,
        prompt_version=prompt_version,
        compact=indirect_visible,
    )
