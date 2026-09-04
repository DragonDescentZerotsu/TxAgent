"""Build and render the raw context-record progressive profile.

The V9 cache chooses a bounded set of conditioned-gold training contexts per query.
This module then joins those contexts to normalized V7 source rows. L1 shows a
deterministic sample of current-gold constituent records; L2 appends a separate
sample of other records in the task's gold source domain with the same parent
and exact condition. The optional cache-backed continuation appends one bundle
of 50 independently ranked Stage 3 records at each task-configured later level.

Selection metadata and source-policy fields remain in preparation audits. Only
the fields declared in the selected ``card_v*.yaml`` and
``level_record_bundle.yaml`` are projected into the model prompt.
``runner.py`` owns query-prior reuse, model calls, checkpoints, and summaries.
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

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import pyarrow.parquet as pq
import yaml

from predict.harnesses.progressive.state import (
    ProgressiveTaskContract,
    card_alias_maps,
    render_prior_state,
)
from predict.retrieval.assay_reranking.v9 import (
    RANKING_SCHEMA_VERSION,
    model_profile,
    verify_vendored_assets,
)
from predict.retrieval.policies import normalize_molecule_identity, seeded_rank_tie_key
from predict.utils.json import read_jsonl, sha256_file


PROTOCOL_VERSION = "conditioned_gold_context_records_progressive.v1"
INDIRECT_PROTOCOL_VERSION = (
    "conditioned_gold_context_records_plus_stage3_later_levels.v2"
)
PROMPT_PROFILES = {
    "v1": {
        "name": "progressive.assay_transfer.v1",
        "card": Path(__file__).with_name("card_v1.yaml"),
        "template": Path(__file__).with_name("prompt_v1.jinja"),
    },
    "v2": {
        "name": "progressive.assay_transfer.v2",
        "card": Path(__file__).with_name("card_v2.yaml"),
        "template": Path(__file__).with_name("prompt_v2.jinja"),
    },
    "v3": {
        "name": "progressive.assay_transfer.v3",
        "card": Path(__file__).with_name("card_v2.yaml"),
        "card_schema_version": "v2",
        "template": Path(__file__).with_name("prompt_v3.jinja"),
    },
    "v4": {
        "name": "progressive.assay_transfer.v4",
        "card": Path(__file__).with_name("card_v4.yaml"),
        "template": Path(__file__).with_name("prompt_v4.jinja"),
    },
    "v4.1": {
        "name": "progressive.assay_transfer.v4.1",
        "card": Path(__file__).with_name("card_v1.yaml"),
        "card_schema_version": "v1",
        "template": Path(__file__).with_name("prompt_v4_1.jinja"),
    },
    "v4.2": {
        "name": "progressive.assay_transfer.v4.2",
        "card": Path(__file__).with_name("card_v1.yaml"),
        "card_schema_version": "v1",
        "template": Path(__file__).with_name("prompt_v4_2.jinja"),
    },
    "v4.3": {
        "name": "progressive.assay_transfer.v4.3",
        "card": Path(__file__).with_name("card_v4.yaml"),
        "card_schema_version": "v4",
        "template": Path(__file__).with_name("prompt_v4.jinja"),
    },
    "v5": {
        "name": "progressive.assay_transfer.v5",
        "card": Path(__file__).with_name("card_v1.yaml"),
        "card_schema_version": "v1",
        "template": Path(__file__).with_name("prompt_v5.jinja"),
    },
    "v6": {
        "name": "progressive.assay_transfer.v6",
        "card": Path(__file__).with_name("card_v4.yaml"),
        "card_schema_version": "v4",
        "template": Path(__file__).with_name("prompt_v5.jinja"),
    },
    "v7": {
        "name": "progressive.assay_transfer.v7",
        "card": Path(__file__).with_name("card_v2.yaml"),
        "card_schema_version": "v2",
        "template": Path(__file__).with_name("prompt_v1.jinja"),
    },
    "v8": {
        "name": "progressive.assay_transfer.v8",
        "card": Path(__file__).with_name("card_v8.yaml"),
        "template": Path(__file__).with_name("prompt_v8.jinja"),
    },
    "v8.1": {
        "name": "progressive.assay_transfer.v8.1",
        "card": Path(__file__).with_name("card_v8.yaml"),
        "card_schema_version": "v8",
        "template": Path(__file__).with_name("prompt_v8.jinja"),
    },
    "v8.2": {
        "name": "progressive.assay_transfer.v8.2",
        "card": Path(__file__).with_name("card_v2.yaml"),
        "card_schema_version": "v2",
        "template": Path(__file__).with_name("prompt_v8.jinja"),
    },
    "morgan_v3": {
        "name": "progressive.morgan_ranked.v3",
        "card": Path(__file__).with_name("card_v2.yaml"),
        "card_schema_version": "v2",
        "template": Path(__file__).with_name("prompt_morgan_v3.jinja"),
        "ranking": "morgan",
    },
    "morgan_v4": {
        "name": "progressive.morgan_ranked.v4",
        "card": Path(__file__).with_name("card_v4.yaml"),
        "card_schema_version": "v4",
        "template": Path(__file__).with_name("prompt_morgan_v4.jinja"),
        "ranking": "morgan",
    },
    "morgan_v6": {
        "name": "progressive.morgan_ranked.v6",
        "card": Path(__file__).with_name("card_v4.yaml"),
        "card_schema_version": "v4",
        "template": Path(__file__).with_name("prompt_morgan_v5.jinja"),
        "ranking": "morgan",
    },
}
TASK_BEST_PROMPT_VERSIONS = {
    "bbb_martins": "v4",
    "bioavailability_ma": "v3",
    "skin_reaction": "v6",
}
TASK_MORGAN_PROMPT_VERSIONS = {
    "bbb_martins": "morgan_v4",
    "bioavailability_ma": "morgan_v3",
    "skin_reaction": "morgan_v6",
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
    task_levels = level_record_bundle_contract()["task_levels"].get(task)
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
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in LEVELS]
    if include_indirect:
        if task is None:
            raise ValueError("task is required for cache-backed later levels")
        task_levels = level_record_bundle_contract()["task_levels"][task]
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
    if prompt_version != "task_best":
        prompt_profile(prompt_version)
        return prompt_version
    try:
        versions = (
            TASK_MORGAN_PROMPT_VERSIONS
            if ranking == "morgan"
            else TASK_BEST_PROMPT_VERSIONS
        )
        return versions[task]
    except KeyError as exc:
        raise ValueError(f"no frozen task-best context-record prompt for {task}") from exc


def prompt_profile(prompt_version: str) -> Mapping[str, Any]:
    try:
        return PROMPT_PROFILES[prompt_version]
    except KeyError as exc:
        raise ValueError(f"unknown assay-transfer prompt version: {prompt_version}") from exc


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
        yaml.safe_load(path.read_text(encoding="utf-8")), prompt_version
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
        yaml.safe_load(LEVEL_RECORD_BUNDLE_PATH.read_text(encoding="utf-8")),
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
    result = _first(
        row,
        "canonical_measurement_text",
        "resolved_measurement_text",
        "measurement_text",
    )
    if not result:
        result = "not explicitly reported"
    surface = {
        "card_id": card_id,
        "first_seen_level": level,
        "level": f"L{level}",
        "endpoint": _first(row, "canonical_endpoint_name", "endpoint_name"),
        "result": result,
        "unit": _first(row, "canonical_unit_text", "resolved_unit_text", "unit_text"),
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
        "_canonical_record_id": canonical_id,
    }
    surface.update({f"source__{field}": row.get(field) for field in _SOURCE_RECORD_FIELDS})
    surface.update({field: row.get(field) for field in _CANONICAL_RECORD_FIELDS})
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
    """Project V21-ranked raw records into append-only molecule cards."""
    contexts = []
    for molecule in ranked_molecules:
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
                card["_v21_transfer_likelihood"] = float(
                    ranked_record["transfer_likelihood"]
                )
                output.append(card)
            return output

        contexts.append(
            {
                "context_card_id": context_id,
                "canonical_smiles": str(molecule["canonical_smiles"]),
                "condition_group": "source_native_assay_contexts",
                "transfer_likelihood": round(
                    float(molecule["transfer_likelihood"]), 4
                ),
                "available_l1": int(molecule["available_l1"]),
                "available_l2": int(molecule["available_l2"]),
                "l1_cards": cards(1),
                "l2_cards": cards(2),
                "_selection_rank": int(molecule["selection_rank"]),
                "_gold_record_id": None,
                "_context_origin": "v21_record_ranked_molecule",
                "_reference_molecule_id": molecule_id,
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
    current_path = benchmark_root / task_dir / "CURRENT"
    release = current_path.read_text(encoding="utf-8").strip()
    if not release or "/" in release or "\\" in release:
        raise ValueError(f"invalid active gold pointer: {current_path}")
    gold_dir = benchmark_root / task_dir / release / "scaffold"
    gold_path = gold_dir / "train_molecule_condition_labels.jsonl"
    gold_rows = read_jsonl(gold_path)
    gold_by_id = {str(row["benchmark_row_id"]): row for row in gold_rows}

    cache_dir = ranking_root / task / "scaffold" / "valid"
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
            selected = [row for row in rows if int(row["model_rank"]) < context_limit]
            selected.sort(key=lambda row: int(row["model_rank"]))
        else:
            selected = sorted(
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
            )[:context_limit]
        for selection_rank, row in enumerate(selected):
            row["_selection_rank"] = selection_rank
        selected_by_query[query_id] = selected
    for query_id, rows in selected_by_query.items():
        if len(rows) != context_limit:
            raise ValueError(f"{task} query {query_id} lacks {context_limit} contexts")
        if ranking == "assay_transfer":
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
        if len(rows) != level_record_limit:
            raise ValueError(
                f"{task} {level_name} requires exactly {level_record_limit} records"
            )
        definition = task_levels[level_name]
        cards: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(rows, start=1):
            payload = row.get("payload") or {}
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
                    payload.get("source_fields") or {}
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


@lru_cache(maxsize=1)
def _prompt_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        undefined=StrictUndefined,
        autoescape=False,
    )


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
            raise ValueError("indirect_record_limit values must be positive")
    elif indirect_record_limit < 1:
        raise ValueError("indirect_record_limit must be positive")
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
            f"Each context and level shows a deterministic sample of at most {record_limit_label} raw records. "
            "available_record_counts reports the full pool before sampling. Repeated records "
            "and records from one context are not independent votes."
        )
    else:
        if indirect_visible and isinstance(indirect_record_limit, Mapping):
            limit_summary = ", ".join(
                f"{level}={int(limit)}"
                for level, limit in indirect_record_limit.items()
            )
            later_sampling = (
                f"The later-level record caps are {limit_summary}; each level "
                "appends independently ranked records. "
            )
        else:
            later_sampling = (
                f"Each later level appends at most {indirect_record_limit} ranked records. "
                if indirect_visible
                else ""
            )
        record_sampling = (
            f"Each context shows at most {record_limit} raw records at L1 and appends at most "
            f"{l2_record_limit} new raw records at L2. "
            f"{later_sampling}available_record_counts reports the full pool before sampling. "
            "Repeated records and records from one context are not independent votes."
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
                f" Each later level appends at most {indirect_record_limit} ranked records."
                if indirect_visible
                else ""
            )
        )
        record_sampling = (
            f"Each reference molecule shows up to {record_limit} V21-ranked L1 records "
            f"and appends up to {l2_record_limit} V21-ranked L2 records."
            f"{later_sampling} Repeated records and records from one molecule are not "
            "independent votes."
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
            f"The per-level record caps are {limit_summary}; each level appends "
            "independently ranked raw records. "
            "available_record_count reports the full pool before selection. Repeated records "
            "and records from one molecule are not independent votes."
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
                "applicability": "high | moderate | low | not_applicable",
                "direction": "supportive | contradictory | neutral_or_unclear",
                "decision_effect": "changed | strengthened | weakened | no_change",
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
                f"{indirect_label} append one independently ranked Stage 3 record bundle each; "
                "all earlier record bundles remain visible."
                if record_only
                else (
                    f"The same {context_count} V21-ranked scaffold-disjoint reference "
                    "molecules remain visible. L1 contains direct BBB records; L2 appends "
                    "near-direct records from those molecules; all L1 cards remain visible."
                    + (
                        f" {indirect_label} append one V21-ranked Stage 3 record bundle "
                        "each; all earlier cards remain visible."
                        if indirect_visible
                        else ""
                    )
                    if v21_molecule_cards
                    else
                    f"The same {context_count} exact training parent-condition "
                    "contexts remain visible. "
                    "L1 contains current-gold constituent measurements. L2 appends other raw "
                    "gold-source-domain records from those same contexts; all L1 cards remain visible."
                    + (
                        f" {indirect_label} append one ranked Stage 3 record bundle "
                        "each; all earlier cards remain visible."
                        if indirect_visible
                        else ""
                    )
                )
            ),
            "record_sampling": record_sampling,
            "level_interpretation": (
                "Each record belongs to its displayed evidence family. L1 contains direct BBB "
                "experiments; later levels are supporting or mechanistic context, "
                "not additional gold votes."
                if record_only
                else (
                    "L2 records did not contribute a current gold vote. That does not make them negative, "
                    "invalid, or independent votes; use them only as additional experimental context."
                    + (
                        f" {indirect_label} are separately retrieved mechanistic families, not additional gold votes."
                        if indirect_visible
                        else ""
                    )
                )
            ),
            "transfer_rule": (
                "morgan_similarity is radius-2, 2048-bit Morgan-fingerprint Tanimoto similarity. "
                "It is structural proximity used for ranking, not a task-label probability, "
                "endpoint result, or guarantee of biological transferability."
                if score_field == "morgan_similarity"
                else (
                    "For each molecule card, transfer_likelihood is its highest V21 L1 "
                    "record-transfer score and is used to rank molecules; V21 scores also "
                    "rank the records selected inside each card. On later-level record cards, "
                    "transfer_likelihood estimates whether that individual source record applies. "
                    "It is never a task-label probability or an experimental result."
                    if v21_molecule_cards
                    else "transfer_likelihood estimates whether a training context applies to this query. "
                    f"On {indirect_label} record cards it instead estimates whether that individual source record applies. "
                    "It is never a task-label probability or an experimental result."
                    if indirect_visible
                    else "transfer_likelihood estimates whether a training context applies to this query. "
                    "It is not a task-label probability and is not an experimental result."
                )
            ),
            "structure_rule": (
                "The numeric Morgan similarity is supplied explicitly. Use it only as structural context; "
                "do not infer an endpoint outcome from it or invent hidden selection metadata."
                if score_field == "morgan_similarity"
                else "No numeric structure-comparison score is supplied. Interpret structures qualitatively "
                "and do not invent a numeric score or hidden selection metadata."
            ),
            "prior_rule": (
                "query_prior and molecule_property_tool_summary are physicochemical priors, not endpoint proof."
            ),
            "citation_rule": (
                "Every compound-specific experimental claim must cite record card aliases. "
                "Use only cards that materially affect the decision."
            ),
            "flip_rule": (
                "A flip is allowed only when new evidence overturns the prior decision; a flip must cite a new card."
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
            )[-1],
            "full_level_plan": levels(
                task=contract.task,
                include_indirect=indirect_visible,
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
    rendered = _prompt_environment().get_template(
        Path(prompt_profile(prompt_version)["template"]).name
    ).render(
        task=contract.task,
        prompt_version=prompt_version,
        system_role=system_role,
        user_content=json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":") if indirect_visible else None,
        ),
    )
    messages = json.loads(rendered)
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("context-record prompt must render exactly two messages")
    return messages
