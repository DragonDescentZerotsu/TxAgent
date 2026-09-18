"""Run legacy or cache-matched flat molecular-evidence inference.

The public entry point defaults to ``joseph-flat-v2``. It validates the official
BBB or oral split, asks the current cache-matched SQLite contract for the same frozen
level-specific record assignments consumed by progressive, then groups those
already-selected records by normalized parent into one ``Flat.all_evidence``
branch. It materializes a provenance-bound replay artifact and delegates
single/group/final model scheduling and checkpoints to ``branches.runner``.

``joseph-flat-v2`` presents source-contract-complete semantic cards without
group comparison tools. ``joseph-flat-v1`` and ``tianang-flat-v1`` retain their
historical presentation. Joseph supports Morgan, assay-transfer, and joint
cache-matched modes plus the three finalized record pools. Single-molecule and
final prompts remain task-owned in every version.

``retrieve_flat_evidence`` remains the legacy mechanism-family collapse;
``cache_matched_flat_retrieval`` performs only Joseph's presentation merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import yaml

from predict.harnesses.branches.assay_transfer import (
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
)
from predict.harnesses.branches.retrieval import (
    ASSAY_TRANSFER_TOOL_STRATEGY,
    BranchRetrievalConfig,
    RetrievalReranker,
    flatten_retrieval_groups,
    retrieve_mechanism_evidence,
    retrieval_coverage,
)
from predict.retrieval.policies import NeighborIdentityPolicy, SIMILARITY_SELECTOR
from predict.retrieval.policies import normalize_molecule_identity, selector_metadata
from predict.llm_io.evidence import minimal_evidence_from_row
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from data.processing.gold_labels.conditioned_benchmark import split_path


TIANANG_PROMPT_VERSION = "tianang_flat_v1"
JOSEPH_V1_PROMPT_VERSION = "joseph_flat_v1"
JOSEPH_PROMPT_VERSION = "joseph_flat_v2"
CONTEXT_V4_PROMPT_VERSION = "joseph_flat_context_v4_v1"
CONTEXT_V5_PROMPT_VERSION = "full_flat_context_v5"
JOSEPH_PROMPT_VERSIONS = (
    JOSEPH_V1_PROMPT_VERSION,
    JOSEPH_PROMPT_VERSION,
    CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_V5_PROMPT_VERSION,
)
# Backward-compatible default for direct prompt-library callers. The public
# The unified branches CLI selects Joseph flat behavior below.
PROMPT_VERSION = TIANANG_PROMPT_VERSION
PUBLIC_HARNESS_VERSION = "joseph-flat-v2"
CONTEXT_V4_HARNESS_VERSION = "joseph-flat-context-v4-v1"
CONTEXT_V5_HARNESS_VERSION = "full-flat-context-v5"
JOSEPH_V1_HARNESS_VERSION = "joseph-flat-v1"
LEGACY_HARNESS_VERSION = "tianang-flat-v1"
JOSEPH_HARNESS_PROMPTS = {
    JOSEPH_V1_HARNESS_VERSION: JOSEPH_V1_PROMPT_VERSION,
    PUBLIC_HARNESS_VERSION: JOSEPH_PROMPT_VERSION,
    CONTEXT_V4_HARNESS_VERSION: CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_V5_HARNESS_VERSION: CONTEXT_V5_PROMPT_VERSION,
}
JOSEPH_PROMPT_HARNESSES = {
    prompt: harness for harness, prompt in JOSEPH_HARNESS_PROMPTS.items()
}
PROMPT_DIR = Path(__file__).with_name("prompts")
MORGAN_VARIANT = "morgan"
ASSAY_TRANSFER_VARIANT = "assay-transfer"
JOINT_VARIANT = "joint"
VARIANTS = (MORGAN_VARIANT, ASSAY_TRANSFER_VARIANT, JOINT_VARIANT)
CONTEXT_V4_VARIANTS = (
    MORGAN_VARIANT,
    "morgan-contrastive",
    ASSAY_TRANSFER_VARIANT,
    "assay-transfer-contrastive",
    "assay-transfer-within-morgan",
)
CONTEXT_V4_LAYOUTS = ("global", "level-grouped")
CONTEXT_V4_SELECTION_CONTRACT = "joseph_flat_context_retrieval.v1"
CONTEXT_V5_SELECTION_CONTRACT = "joseph_flat_context_retrieval.v2"
CONTEXT_PROMPT_VERSIONS = {CONTEXT_V4_PROMPT_VERSION, CONTEXT_V5_PROMPT_VERSION}
DEFAULT_QUERY_PRIOR_ROOT = Path(
    "outputs/paper/legacy/"
    "starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/"
    "runs_deployment_visible_parent_disjoint"
)
MOLECULE_DESCRIPTION_ARTIFACTS = {
    "v1": (
        Path("/vast/projects/myatskar/design-documents/canonical_smiles_quotient.parquet"),
        "abd6f6d31ee74d854fd42330e816516c39a5ca63c67df56965cc8ec44468f183",
    ),
    "v2": (
        Path("/vast/projects/myatskar/design-documents/canonical_smiles_quotient_v2.parquet"),
        "3e1ca4ece0f137b492c4d6c60713cf9771fbab7871a11b4c6c126df674b0abb7",
    ),
}
MOLECULE_DESCRIPTION_COLUMNS = {
    "raw": "description_raw",
    "motif": "description_motif",
    "coarse": "description_coarse",
}
PROMPT_VARIANTS = {
    TIANANG_PROMPT_VERSION: (MORGAN_VARIANT, ASSAY_TRANSFER_VARIANT),
    JOSEPH_V1_PROMPT_VERSION: VARIANTS,
    JOSEPH_PROMPT_VERSION: VARIANTS,
    CONTEXT_V4_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V5_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
}
TASKS = {"bbb_martins": 5, "bioavailability_ma": 6}
RECORD_POOLS = {
    "assay-transfer-trained": "tool-accepted",
    "all_transfer_eligible": "tool-compatible",
    "all": "all",
}
SQLITE_SELECTION_CONTRACTS = {
    "cache_matched_retrieval.v2": "cache_matched_v2.py",
    "cache_matched_retrieval.v3": "cache_matched_v3.py",
    "ranked_evidence_retrieval.v1": "ranked_retrieval.py",
    "ranked_level_retrieval.v2": "ranked_level_retrieval.py",
    "ranked_uid_retrieval.v1": "ranked_uid_retrieval.py",
}
EVIDENCE_PROJECTION = "source_contract_complete_semantics.v1"
EXTRA_DETAILS_POLICY = "raw_nonempty_visible"
MOLECULE_NAME_FIELDS = {
    "compound_aliases",
    "compound_name",
    "compound_names",
    "drug_name",
    "molecule_aliases",
    "molecule_name",
    "molecule_names",
    "preferred_name",
    "source_molecule_name",
    "source_molecule_names",
}
NONSEMANTIC_SOURCE_FIELDS = {
    "canonical_record_id",
    "canonical_smiles",
    "confidence",
    "doi",
    "extraction_id",
    "global_identifier",
    "needs_more_context",
    "paragraph_idx",
    "pmid",
    "record_id",
    "smiles",
    "source_doi",
    "source_id",
    "source_index",
    "source_name",
    "source_record_id",
    "source_row_number",
    "source_row_uid",
    "source_smiles",
    "source_url",
}
CORE_FIELD_SOURCES = {
    "assay_context": (
        "assay_context",
        "assay_description",
        "assay_model",
        "assay_system",
    ),
    "endpoint": (
        "endpoint_name",
        "standard_type",
        "measurement_type",
        "endpoint",
        "endpoint_type",
        "parameter_name",
        "metric_type",
        "bioavailability_report_type",
    ),
    "reported_value": (
        "measurement_text",
        "standard_value",
        "reported_value",
        "parameter_value",
        "quant_value",
        "value",
    ),
    "reported_unit": (
        "unit_text",
        "standard_units",
        "reported_units",
        "parameter_units",
        "quant_units",
        "unit",
    ),
    "species": ("species", "organism", "species_or_population"),
    "qualifying_conditions": (
        "qualifying_conditions",
        "experimental_conditions",
    ),
    "support_text": ("support_text", "description"),
}
FLAT_V2_CARD_FIELDS = (
    "card_id",
    "assay_context",
    "endpoint",
    "reported_value",
    "reported_unit",
    "species",
    "qualifying_conditions",
    "experimental_details",
    "extra_details",
    "support_text",
    "transfer_likelihood",
)


def _source_semantic_prompt(version: str) -> bool:
    return version in {JOSEPH_PROMPT_VERSION, *CONTEXT_PROMPT_VERSIONS}
FLAT_V2_MOLECULE_SCORE_FIELDS = (
    "morgan_similarity",
    "transfer_likelihood",
    "morgan_top5_rank",
    "assay_transfer_top5_rank",
)


def prompt_directory(version: str) -> Path:
    """Resolve one explicit flat prompt bundle without fallback."""
    if Path(version).name != version or version not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown flat prompt version: {version!r}")
    directory = PROMPT_DIR / version
    if not directory.is_dir():
        raise ValueError(f"Flat prompt bundle is missing: {directory}")
    return directory


@lru_cache(maxsize=None)
def prompt_assets(version: str = PROMPT_VERSION) -> dict[str, Any]:
    """Load and validate one version-owned flat group prompt contract."""
    directory = prompt_directory(version)
    tasks = yaml.safe_load((directory / "tasks.yaml").read_text(encoding="utf-8"))
    modes = {
        mode: yaml.safe_load(
            (directory / "modes" / mode / "user.yaml").read_text(encoding="utf-8")
        )
        for mode in PROMPT_VARIANTS[version]
    }
    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    levels = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "levels").glob("L*.yaml"))
    } if (directory / "levels").is_dir() else {}
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"Flat prompt tasks are invalid: {directory / 'tasks.yaml'}")
    if any(
        not isinstance(mode, dict) or not mode.get("guidance")
        for mode in modes.values()
    ):
        raise ValueError(f"Flat prompt modes are invalid: {directory / 'modes'}")
    if provenance.get("version") != version or set(tasks) != set(
        provenance.get("supports_tasks") or []
    ):
        raise ValueError(f"Flat prompt provenance does not match {directory}")
    for task_id, contract in tasks.items():
        source = (provenance.get("tasks") or {}).get(task_id) or {}
        if source.get("prompt_profile") != contract.get("prompt_profile"):
            raise ValueError(f"Flat prompt profile provenance mismatch: {task_id}")
    return {
        "tasks": tasks,
        "modes": modes,
        "provenance": provenance,
        "levels": levels,
    }


def prompt_asset_manifest(version: str = PROMPT_VERSION) -> dict[str, Any]:
    """Fingerprint the complete prompt bundle and its assembly code."""
    prompt_assets(version)
    directory = prompt_directory(version)
    hashes = {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.suffix in {".jinja", ".json", ".yaml"}
    }
    assembly = {"flat.py": _sha256(Path(__file__))}
    digest = hashlib.sha256(
        json.dumps(
            {"files_sha256": hashes, "assembly_files_sha256": assembly},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": version,
        "directory": str(directory),
        "files_sha256": hashes,
        "assembly_files_sha256": assembly,
        "sha256": digest,
    }


def flat_prompt_variant(
    retrieval_strategy: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    flat_reranking: str = "",
) -> str:
    """Resolve the prompt mode from the legacy or cache-matched CLI."""
    if prompt_version in JOSEPH_PROMPT_VERSIONS:
        if flat_reranking not in PROMPT_VARIANTS[prompt_version]:
            raise ValueError(
                f"{prompt_version} does not support reranking {flat_reranking!r}"
            )
        return flat_reranking
    if retrieval_strategy == "morgan_fingerprint":
        return MORGAN_VARIANT
    if retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        return ASSAY_TRANSFER_VARIANT
    raise ValueError(
        "tianang_flat_v1 requires morgan_fingerprint or assay_transfer_tool retrieval"
    )


def apply_flat_group_prompt(
    payload: Mapping[str, Any],
    *,
    task_id: str,
    task_prompt_profile: str,
    include_query_tool_guidance: bool,
    prompt_version: str = PROMPT_VERSION,
    retrieval_strategy: str,
    flat_reranking: str = "",
) -> dict[str, Any]:
    """Apply a frozen task contract and retrieval surface to a payload copy."""
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    output = deepcopy(dict(payload))
    group = output.get("group") or {}
    if group.get("group_id") != "Flat.all_evidence":
        raise ValueError("tianang_flat_v1 requires the Flat.all_evidence group")

    variant = flat_prompt_variant(
        retrieval_strategy,
        prompt_version=prompt_version,
        flat_reranking=flat_reranking,
    )
    output["task"] = str(contract["task"])
    output["required_json_schema"] = deepcopy(contract["required_json_schema"])
    instruction_key = (
        "with_comparisons" if include_query_tool_guidance else "without_comparisons"
    )
    instructions: list[str] = []
    for entry in contract["instructions"]:
        if isinstance(entry, str):
            instructions.append(entry)
        elif isinstance(entry, dict) and entry.get(instruction_key):
            selected = entry[instruction_key]
            instructions.extend(
                [str(line) for line in selected]
                if isinstance(selected, list)
                else [str(selected)]
            )
    if prompt_version == JOSEPH_PROMPT_VERSION:
        if output.get("prompt_transport"):
            raise ValueError(
                "joseph_flat_v2 refuses prompt transport truncation; "
                "all selected semantic fields must remain visible"
            )
        output = {
            "task": str(contract["task"]),
            "query": _query_without_tools(output.get("query") or {}),
            "molecules": _flat_v2_prompt_molecules(output.get("neighbors") or []),
            "instructions": instructions,
            "required_json_schema": deepcopy(contract["required_json_schema"]),
        }
    elif prompt_version == TIANANG_PROMPT_VERSION and variant == MORGAN_VARIANT:
        _remove_keys(output, _is_assay_transfer_key)
    elif prompt_version == TIANANG_PROMPT_VERSION:
        if isinstance(group, dict):
            group.pop("assay_transfer_score_policy", None)
        for neighbor in output.get("neighbors") or []:
            if isinstance(neighbor, dict):
                neighbor.pop("similarity", None)
                neighbor.pop("similarity_bucket", None)
                neighbor.pop("similarity_metric", None)
    elif prompt_version == JOSEPH_V1_PROMPT_VERSION:
        # Cache-matched selection records the method and its applicable scores
        # per physical evidence row. An outer molecule score would be ambiguous
        # after several levels are merged into one molecule card.
        for neighbor in output.get("neighbors") or []:
            if isinstance(neighbor, dict):
                neighbor.pop("similarity", None)
                neighbor.pop("similarity_bucket", None)
                neighbor.pop("similarity_metric", None)
    instructions.append(str(prompt_assets(prompt_version)["modes"][variant]["guidance"]))
    output["instructions"] = instructions
    return output


def render_flat_group_messages(
    payload: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
    task_id: str,
    task_prompt_profile: str,
    group_tools_enabled: bool,
    prompt_version: str = PROMPT_VERSION,
    retrieval_strategy: str,
    flat_reranking: str = "",
) -> list[dict[str, str]]:
    """Render the version-owned flat group system and JSON user messages."""
    if prompt_version == JOSEPH_PROMPT_VERSION and group_tools_enabled:
        raise ValueError("joseph_flat_v2 does not expose group comparison tools")
    include_comparisons = bool(group.get("tools_prefetched")) or group_tools_enabled
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    system = _environment().get_template(
        f"{prompt_version}/system.jinja"
    ).render(
        system_role=str(contract["system_role"]).strip(),
        tools_prefetched=bool(group.get("tools_prefetched")),
        group_tools_enabled=group_tools_enabled,
        identity_blind=bool(group.get("identity_blind")),
    )
    user = apply_flat_group_prompt(
        payload,
        task_id=task_id,
        task_prompt_profile=task_prompt_profile,
        include_query_tool_guidance=include_comparisons,
        prompt_version=prompt_version,
        retrieval_strategy=retrieval_strategy,
        flat_reranking=flat_reranking,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def flat_group_validation(
    task_id: str,
    *,
    task_prompt_profile: str,
    prompt_version: str = PROMPT_VERSION,
    group: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the frozen response fields required by one task bundle."""
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    validation: dict[str, Any] = {
        "required_fields": tuple(contract["required_fields"])
    }
    if prompt_version == JOSEPH_PROMPT_VERSION and group is not None:
        card_ids = {
            str((row.get("prompt_evidence") or {}).get("card_id") or "")
            for neighbor in group.get("neighbors") or []
            for row in neighbor.get("evidence_rows") or []
        }
        card_ids.discard("")
        validation.update(
            forbidden_field_names=("key_evidence", "tool_summary"),
            content_validator=lambda content: _claim_errors(content, card_ids),
        )
    return validation


def flat_prompt_provenance(
    task_id: str,
    *,
    retrieval_strategy: str,
    task_prompt_profile: str,
    prompt_version: str = PROMPT_VERSION,
    flat_reranking: str = "",
) -> dict[str, Any]:
    """Describe and fingerprint one frozen task and retrieval specialization."""
    variant = flat_prompt_variant(
        retrieval_strategy,
        prompt_version=prompt_version,
        flat_reranking=flat_reranking,
    )
    _task_contract(task_id, task_prompt_profile, prompt_version)
    assets = prompt_assets(prompt_version)
    manifest = prompt_asset_manifest(prompt_version)
    upstream = assets["provenance"]
    contract = {
        "prompt_version": prompt_version,
        "variant": variant,
        "task_id": task_id,
        "task_prompt_profile": task_prompt_profile,
        "upstream_commit": upstream.get("upstream_commit", ""),
        "upstream_sources": upstream["tasks"][task_id]["upstream_sources"],
        "asset_manifest_sha256": manifest["sha256"],
        "asset_files_sha256": manifest["files_sha256"],
        "assembly_files_sha256": manifest["assembly_files_sha256"],
        "base_prompt_contract": "task_standard_group",
        "flat_group_id": "Flat.all_evidence",
        "retrieval_guidance": assets["modes"][variant]["guidance"],
    }
    if _source_semantic_prompt(prompt_version):
        contract.update(
            evidence_projection=EVIDENCE_PROJECTION,
            extra_details_policy=EXTRA_DETAILS_POLICY,
            molecule_name_visible=False,
            group_tools_enabled=False,
        )
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return contract


def _task_contract(
    task_id: str,
    task_prompt_profile: str,
    prompt_version: str,
) -> Mapping[str, Any]:
    tasks = prompt_assets(prompt_version)["tasks"]
    try:
        contract = tasks[task_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported {prompt_version} task: {task_id}") from exc
    expected_profile = str(contract.get("prompt_profile") or "")
    if task_prompt_profile != expected_profile:
        raise ValueError(
            f"{prompt_version} requires task prompt profile {expected_profile!r} "
            f"for {task_id}, got {task_prompt_profile!r}"
        )
    return contract


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
    )


def _is_assay_transfer_key(key: str) -> bool:
    normalized = key.lower()
    return "assay_transfer" in normalized or "assay-transfer" in normalized


def _remove_keys(value: Any, predicate: Callable[[str], bool]) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if predicate(str(key)):
                value.pop(key)
            else:
                _remove_keys(value[key], predicate)
    elif isinstance(value, list):
        for item in value:
            _remove_keys(item, predicate)


def _query_without_tools(query: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(query))
    output.pop("prefetched_molecule_properties", None)
    output.pop("tools_prefetched", None)
    return output


def _flat_v2_prompt_molecules(
    neighbors: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    molecules: list[dict[str, Any]] = []
    seen_cards: set[str] = set()
    for neighbor in neighbors:
        scores: dict[str, Any] = {}
        cards: list[dict[str, Any]] = []
        for raw_card in neighbor.get("evidence_rows") or []:
            if not isinstance(raw_card, Mapping):
                raise ValueError("joseph_flat_v2 evidence card is not an object")
            source = deepcopy(dict(raw_card))
            for name, value in dict(source.pop("_molecule_scores", {})).items():
                _set_consistent(scores, name, value)
            card = {
                name: source[name]
                for name in FLAT_V2_CARD_FIELDS
                if name in source and _is_nonempty(source[name])
            }
            card_id = str(card.get("card_id") or "")
            if not card_id or card_id in seen_cards:
                raise ValueError(
                    f"joseph_flat_v2 card ID is missing or duplicated: {card_id!r}"
                )
            seen_cards.add(card_id)
            cards.append(card)
        if not cards:
            continue
        molecule = {
            "canonical_smiles": str(neighbor.get("canonical_smiles") or ""),
            **{
                name: scores[name]
                for name in FLAT_V2_MOLECULE_SCORE_FIELDS
                if name in scores
            },
            "evidence_cards": cards,
        }
        molecules.append(molecule)
    return molecules


def build_flat_context_request(
    retrieval: Mapping[str, Any],
    *,
    task_id: str,
    task_prompt_profile: str,
    layout: str,
    reranking: str,
    query_prior: Mapping[str, Any] | None,
    prompt_version: str = CONTEXT_V4_PROMPT_VERSION,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Render one all-level prediction request and its exact visible-ID index."""
    if layout not in CONTEXT_V4_LAYOUTS:
        raise ValueError(f"Unknown flat context layout: {layout!r}")
    if reranking not in CONTEXT_V4_VARIANTS:
        raise ValueError(f"Unsupported flat context reranking: {reranking!r}")
    groups = retrieval.get("groups") or []
    if len(groups) != 1 or groups[0].get("group_id") != "Flat.all_evidence":
        raise ValueError("Flat context prediction requires one Flat.all_evidence group")
    if prompt_version not in CONTEXT_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported flat context prompt: {prompt_version!r}")
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    assets = prompt_assets(prompt_version)
    sections, aliases, references = _flat_context_sections(
        groups[0], layout=layout, prompt_version=prompt_version
    )
    for section in sections:
        level = str(section.get("level") or "")
        section["description"] = str(
            ((assets.get("levels") or {}).get(level) or {}).get(task_id, {}).get(
                "description", ""
            )
        )
    query = deepcopy(dict(retrieval.get("query") or {}))
    query.pop("fingerprint", None)
    payload = {
        "task_id": task_id,
        "query": query,
        "query_prior": deepcopy(dict(query_prior or {})),
        "sections": sections,
        "required_json_schema": _flat_context_schema(contract),
    }
    system = _environment().get_template(
        f"{prompt_version}/system.jinja"
    ).render(
        **contract,
        query_prior_visible=bool(query_prior),
        ranking_guidance=assets["modes"][reranking]["guidance"],
    )
    user = _environment().get_template(
        f"{prompt_version}/user.jinja"
    ).render(**payload)
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ], {
        "card_alias_map": aliases,
        "reasoning_reference_contract": deepcopy(
            assets["provenance"]["reasoning_reference_contracts"][layout]
        ),
        "reasoning_reference_index": references,
    }


def flat_context_validation(
    task_id: str, *, task_prompt_profile: str,
    prompt_version: str = CONTEXT_V4_PROMPT_VERSION,
    reference_index: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the shared two-field flat prediction."""
    contract = _task_contract(
        task_id, task_prompt_profile, prompt_version
    )
    prediction_field = str(contract["prediction_field"])
    expected_fields = set(contract["required_fields"])

    visible = {
        str(row["visible_id"]): str(row["unit_kind"])
        for row in (reference_index or [])
    }

    def content_errors(content: Mapping[str, Any]) -> list[str]:
        if set(content) != expected_fields:
            return ["invalid_output_fields"]
        if prompt_version != CONTEXT_V5_PROMPT_VERSION:
            return []
        errors: list[str] = []
        claims = content.get("claims")
        if not isinstance(claims, list) or not claims:
            return ["claims_must_be_nonempty_array"]
        identifiers_by_role = {"supportive": set(), "contradictory": set()}
        for index, claim in enumerate(claims):
            if not isinstance(claim, Mapping) or set(claim) != {
                "claim", "molecule_ids", "record_ids", "evidence_role"
            }:
                errors.append(f"invalid_claim_fields:{index}")
                continue
            if not isinstance(claim["claim"], str) or not claim["claim"].strip():
                errors.append(f"invalid_claim_text:{index}")
            role = claim["evidence_role"]
            if role not in identifiers_by_role:
                errors.append(f"invalid_evidence_role:{index}")
            cited: list[str] = []
            for name, kind in (("molecule_ids", "molecule"), ("record_ids", "record")):
                values = claim[name]
                if not isinstance(values, list) or any(
                    not isinstance(value, str) for value in values
                ):
                    errors.append(f"invalid_identifier_array:{index}:{name}")
                    continue
                if len(values) != len(set(values)):
                    errors.append(f"duplicate_identifiers:{index}:{name}")
                invalid = sorted(value for value in values if visible.get(value) != kind)
                if invalid:
                    errors.append(
                        f"unknown_identifiers:{index}:{name}:{','.join(invalid)}"
                    )
                cited.extend(values)
            if not cited:
                errors.append(f"claim_without_evidence:{index}")
            if role in identifiers_by_role:
                identifiers_by_role[role].update(cited)
        overlap = sorted(
            identifiers_by_role["supportive"]
            & identifiers_by_role["contradictory"]
        )
        if overlap:
            errors.append(f"evidence_role_overlap:{','.join(overlap)}")
        return errors

    return {
        "required_fields": tuple(contract["required_fields"]),
        "allowed_values": {
            prediction_field: {
                str(contract["positive_prediction"]),
                str(contract["negative_prediction"]),
            },
        },
        "content_validator": content_errors,
    }


def derive_flat_claim_evidence(content: Mapping[str, Any]) -> dict[str, list[str]]:
    """Derive diagnostic role inventories from a validated V5 response."""
    output = {
        "supportive_molecule_ids": [],
        "supportive_record_ids": [],
        "contradictory_molecule_ids": [],
        "contradictory_record_ids": [],
    }
    for claim in content.get("claims") or []:
        role = str(claim["evidence_role"])
        for source, suffix in (("molecule_ids", "molecule_ids"), ("record_ids", "record_ids")):
            target = f"{role}_{suffix}"
            output[target].extend(
                value for value in claim[source] if value not in output[target]
            )
    return output


def _flat_context_schema(contract: Mapping[str, Any]) -> dict[str, Any]:
    schema = deepcopy(dict(contract.get("required_json_schema") or {}))
    if "summary" in contract.get("required_fields", ()):
        schema["summary"] = "concise overall conclusion"
    schema[str(contract["prediction_field"])] = (
        f"{contract['positive_prediction']} | {contract['negative_prediction']}"
    )
    return schema


def _flat_context_sections(
    group: Mapping[str, Any], *, layout: str,
    prompt_version: str = CONTEXT_V4_PROMPT_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    neighbors = list(group.get("neighbors") or [])
    levels = sorted({
        str((row.get("selection_provenance") or {}).get("level") or "")
        for neighbor in neighbors for row in neighbor.get("evidence_rows") or []
    }, key=lambda value: int(value[1:]) if value.startswith("L") else 99)
    section_levels: list[str | None] = [None] if layout == "global" else levels
    sections: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    references: list[dict[str, Any]] = []
    molecule_number = 0
    for section_level in section_levels:
        molecules = []
        for neighbor in neighbors:
            rows = [
                row for row in neighbor.get("evidence_rows") or []
                if section_level is None
                or str((row.get("selection_provenance") or {}).get("level")) == section_level
            ]
            if not rows:
                continue
            molecule_number += 1
            parent_id = str(neighbor.get("standard_inchi_key") or neighbor.get("molecule_chembl_id") or "")
            stable_molecule_id = (
                parent_id if section_level is None else f"{parent_id}@{section_level}"
            )
            cards = []
            molecule_scores: dict[str, Any] = {}
            for record_number, row in enumerate(rows, 1):
                provenance = dict(row.get("selection_provenance") or {})
                card = deepcopy(dict(row.get("prompt_evidence") or {}))
                alias = (
                    f"Record {molecule_number}-{record_number}"
                    if prompt_version == CONTEXT_V5_PROMPT_VERSION
                    else str(card.get("card_id") or "")
                )
                if prompt_version == CONTEXT_V5_PROMPT_VERSION:
                    card["card_id"] = f"{molecule_number}-{record_number}"
                stable_card_id = str(row.get("evidence_id") or "")
                if not alias or alias in aliases or not stable_card_id:
                    raise ValueError(f"Missing or duplicate flat card alias: {alias!r}")
                aliases[alias] = stable_card_id
                card.update(
                    level=str(provenance.get("level") or ""),
                    evidence_family=str(provenance.get("evidence_family") or ""),
                )
                for name in ("selected_condition", "morgan_similarity", "assay_transfer_score"):
                    if provenance.get(name) not in (None, ""):
                        target = "transfer_likelihood" if name == "assay_transfer_score" else name
                        level = str(provenance.get("level") or "")
                        if prompt_version == CONTEXT_V5_PROMPT_VERSION and name == "morgan_similarity":
                            _set_consistent(molecule_scores, target, provenance[name])
                        elif (
                            prompt_version == CONTEXT_V5_PROMPT_VERSION
                            and name == "assay_transfer_score" and level == "L1"
                        ):
                            _set_consistent(molecule_scores, target, provenance[name])
                        elif not (
                            prompt_version == CONTEXT_V5_PROMPT_VERSION
                            and name == "assay_transfer_score" and level == "L5"
                        ):
                            card[target] = provenance[name]
                cards.append(card)
                references.append({
                    "unit_kind": "record",
                    "visible_id": alias,
                    "stable_id": stable_card_id,
                    "source_analog_id": parent_id,
                    "containing_unit_id": stable_molecule_id,
                    "first_visible_level": int(str(provenance.get("level") or "L0")[1:]),
                    "is_new": True,
                    "prompt_heading": (
                        alias if prompt_version == CONTEXT_V5_PROMPT_VERSION
                        else f"Record {alias}"
                    ),
                })
            molecule = {
                "number": molecule_number,
                "canonical_smiles": str(neighbor.get("canonical_smiles") or ""),
                "evidence_cards": cards,
                **molecule_scores,
            }
            if neighbor.get("molecule_description"):
                molecule["molecule_description"] = neighbor["molecule_description"]
            molecules.append(molecule)
            references.insert(len(references) - len(cards), {
                "unit_kind": "molecule",
                "visible_id": f"Molecule {molecule_number}",
                "stable_id": stable_molecule_id,
                "source_analog_id": parent_id,
                "containing_unit_id": "",
                "first_visible_level": min(
                    int(str((row.get("selection_provenance") or {}).get("level") or "L0")[1:])
                    for row in rows
                ),
                "is_new": True,
                "prompt_heading": f"## Molecule {molecule_number}",
            })
        sections.append({
            "heading": f"Level {section_level}" if section_level else "",
            "level": section_level or "",
            "molecules": molecules,
        })
    return sections, aliases, references


def _claim_errors(content: Mapping[str, Any], card_ids: set[str]) -> list[str]:
    claims = content.get("claims")
    if not isinstance(claims, list):
        return ["invalid_claims:not_a_list"]
    errors: list[str] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            errors.append(f"invalid_claim:{index}:not_an_object")
            continue
        if set(claim) != {"claim", "card_ids"}:
            errors.append(f"invalid_claim:{index}:fields")
        if not str(claim.get("claim") or "").strip():
            errors.append(f"invalid_claim:{index}:empty_text")
        cited = claim.get("card_ids")
        if not isinstance(cited, list) or not cited:
            errors.append(f"invalid_claim:{index}:card_ids")
            continue
        if any(not isinstance(card_id, str) for card_id in cited):
            errors.append(f"invalid_claim:{index}:non_string_card_id")
            continue
        unknown = sorted(set(cited) - card_ids)
        if unknown:
            errors.append(f"unknown_card_ids:{index}:{','.join(unknown)}")
    return errors


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieve_flat_evidence(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    config: BranchRetrievalConfig,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str = NeighborIdentityPolicy.OPERATIONAL.value,
    neighbor_selector: str = SIMILARITY_SELECTOR,
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Retrieve mechanism families, then collapse them into one group branch."""
    retrieval = retrieve_mechanism_evidence(
        query_smiles,
        index,
        config=config,
        mode="full_flat",
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        neighbor_selector=neighbor_selector,
        reranker=reranker,
        assay_transfer_initial_morgan_filter=assay_transfer_initial_morgan_filter,
        assay_transfer_min_score=assay_transfer_min_score,
        assay_transfer_diversity_mode=assay_transfer_diversity_mode,
        assay_transfer_diversity_score_slack=assay_transfer_diversity_score_slack,
        assay_transfer_selection_unit=assay_transfer_selection_unit,
        assay_transfer_records_per_molecule=assay_transfer_records_per_molecule,
    )
    if retrieval.get("status") == "ok":
        retrieval["groups"] = [flatten_retrieval_groups(retrieval["groups"])]
        retrieval["coverage"] = retrieval_coverage(
            retrieval["groups"],
            min_similarity=min_similarity,
            top_k_per_group=top_k_per_group,
        )
    return retrieval


def cache_matched_flat_retrieval(
    query_id: str,
    query_smiles: str,
    molecules: list[dict[str, Any]],
    later_levels: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    reranking: str,
    query_audit: Mapping[str, Any],
    query_identity: Mapping[str, Any] | None = None,
    prompt_version: str = JOSEPH_V1_PROMPT_VERSION,
    selection_contract: str = "cache_matched_retrieval.v2",
) -> dict[str, Any]:
    """Flatten one cache-matched progressive selection without re-selecting it."""
    if query_identity:
        query_parent_smiles = str(query_identity.get("parent_smiles") or "")
        query_parent_id = str(query_identity.get("parent_id") or "")
        if not query_parent_smiles:
            raise ValueError(f"Incomplete cached query identity: {query_id}")
    else:
        identity = normalize_molecule_identity(query_smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(f"Unresolved query molecule: {query_id}")
        query_parent_smiles = identity.parent_smiles
        query_parent_id = identity.parent_inchi_key or ""

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for molecule in molecules:
        for row in molecule.get("l1_records") or []:
            selected.append((row, molecule))
    for level in sorted(later_levels, key=lambda value: int(value[1:])):
        for row in later_levels[level].get("records") or []:
            selected.append((row, {}))

    neighbors: dict[str, dict[str, Any]] = {}
    seen_records: set[str] = set()
    level_identity_policy = {
        f"L{level}": "scaffold_disjoint" if level == 1 else "parent_disjoint"
        for level in range(1, TASKS[task] + 1)
    }
    for row, molecule in selected:
        record_id = str(row["record_id"])
        if record_id in seen_records:
            continue
        seen_records.add(record_id)
        payload = dict(row["payload"])
        parent_id = str(row["reference_molecule_id"])
        method = str(
            row.get("ranking_method") or molecule.get("ranking_method") or ""
        ).replace("_", "-")
        morgan_method = method in {MORGAN_VARIANT, "morgan-contrastive"}
        assay_method = method in {
            ASSAY_TRANSFER_VARIANT,
            "assay-transfer-contrastive",
            "assay-transfer-within-morgan",
        }
        retrieval = {
            "record_id": record_id,
            "level": str(payload["progressive_level"]),
            "evidence_family": str(payload["family_key"]),
            "ranking_method": method,
        }
        if selection_contract in {
            "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
            CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
        }:
            retrieval["neighbor_identity_policy"] = level_identity_policy[
                retrieval["level"]
            ]
        if row.get("morgan_rank") is not None:
            retrieval["morgan_rank"] = int(row["morgan_rank"])
        if row.get("assay_rank") is not None:
            retrieval["assay_rank"] = int(row["assay_rank"])
        if morgan_method or method == JOINT_VARIANT:
            retrieval["morgan_similarity"] = round(
                float(row.get("morgan_similarity", molecule.get("morgan_similarity"))), 4
            )
        transfer_score = row.get("transfer_likelihood", molecule.get("transfer_likelihood"))
        if assay_method or method == JOINT_VARIANT:
            if transfer_score is None:
                raise ValueError(f"Selected assay-transfer record lacks a score: {record_id}")
            retrieval["assay_transfer_score"] = round(float(transfer_score), 4)
        if method == JOINT_VARIANT:
            retrieval["morgan_panel_rank"] = molecule.get("morgan_top5_rank")
            retrieval["assay_transfer_panel_rank"] = molecule.get(
                "assay_transfer_top5_rank"
            )
        if molecule.get("selected_condition"):
            retrieval["selected_condition"] = str(molecule["selected_condition"])
        for source, target in (
            ("_diagnostic_label", "diagnostic_label"),
            ("_diagnostic_context_id", "diagnostic_context_id"),
            ("_diagnostic_parent_id", "diagnostic_parent_id"),
        ):
            if molecule.get(source) is not None:
                retrieval[target] = molecule[source]

        neighbor = neighbors.setdefault(
            parent_id,
            {
                "rank": len(neighbors) + 1,
                "molecule_chembl_id": parent_id,
                "canonical_smiles": str(payload["canonical_smiles"]),
                "standard_inchi_key": parent_id,
                "similarity": round(float(row.get("morgan_similarity", molecule.get("morgan_similarity", 0.0))), 4),
                "similarity_bucket": _similarity_bucket(
                    float(row.get("morgan_similarity", molecule.get("morgan_similarity", 0.0)))
                ),
                "similarity_metric": "Morgan radius=2 bits=2048 Tanimoto",
                "molecule_relation": (
                    level_identity_policy[str(payload["progressive_level"])]
                    if selection_contract in {
                        "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                        CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
                    }
                    else "scaffold_disjoint"
                ),
                "source_group_ids": [],
                "evidence_rows": [],
            },
        )
        if (
            selection_contract in {
                "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
            }
            and neighbor["molecule_relation"]
            != level_identity_policy[str(payload["progressive_level"])]
        ):
            neighbor["molecule_relation"] = "level_specific_disjoint"
        card_score: float | None = None
        if prompt_version == JOSEPH_PROMPT_VERSION:
            level = str(payload["progressive_level"])
            scores = neighbor.setdefault("_prompt_scores", {})
            if morgan_method or method == JOINT_VARIANT:
                _set_consistent(scores, "morgan_similarity", retrieval["morgan_similarity"])
            if level == "L1" and (assay_method or method == JOINT_VARIANT):
                _set_consistent(
                    scores,
                    "transfer_likelihood",
                    retrieval["assay_transfer_score"],
                )
            elif level != "L1" and (assay_method or method == JOINT_VARIANT):
                card_score = retrieval["assay_transfer_score"]
            if level == "L1" and method == JOINT_VARIANT:
                _set_consistent(
                    scores, "morgan_top5_rank", retrieval["morgan_panel_rank"]
                )
                _set_consistent(
                    scores,
                    "assay_transfer_top5_rank",
                    retrieval["assay_transfer_panel_rank"],
                )
        family = str(payload["family_key"])
        if family not in neighbor["source_group_ids"]:
            neighbor["source_group_ids"].append(family)
        if _source_semantic_prompt(prompt_version):
            card, accounting = _source_semantic_card(
                payload,
                card_id=f"C{len(seen_records):02d}",
                transfer_likelihood=card_score,
            )
            card["_molecule_scores"] = deepcopy(
                neighbor.get("_prompt_scores") or {}
            )
            neighbor["evidence_rows"].append(
                {
                    "evidence_id": record_id,
                    "prompt_evidence": card,
                    "selection_provenance": deepcopy(retrieval),
                    "source_provenance": {
                        "source_id": str(payload["source_id"]),
                        "source_row_uid": str(payload.get("source_row_uid") or ""),
                        "measurement_kind": str(payload.get("measurement_kind") or ""),
                    },
                    "source_contract": deepcopy(payload["source_contract"]),
                    "source_fields": deepcopy(dict(payload.get("source_fields") or {})),
                    "field_accounting": accounting,
                }
            )
        else:
            neighbor["evidence_rows"].append(
                {
                    "evidence_id": record_id,
                    "minimal_evidence": _minimal_selected_evidence(
                        payload, parent_id, retrieval
                    ),
                }
            )
        neighbor["n_evidence_rows"] = len(neighbor["evidence_rows"])

    group = {
        "group_id": "Flat.all_evidence",
        "tier": "Flat",
        "endpoint_group": "all_evidence",
        "source_group_ids": sorted(
            {family for neighbor in neighbors.values() for family in neighbor["source_group_ids"]}
        ),
        "n_candidate_molecules": len(neighbors),
        "neighbors": list(neighbors.values()),
    }
    selector = selector_metadata(SIMILARITY_SELECTOR)
    return {
        "status": "ok",
        "evidence_source": {
            "type": "starling_v10_cache_matched",
            "selection_contract": selection_contract,
        },
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": query_parent_smiles,
            "standard_inchi_key": query_parent_id,
            "fingerprint": {"radius": 2, "bits": 2048},
        },
        "retrieval_policy": {
            "neighbor_selector": selector,
            "neighbor_identity_policy": (
                "level_specific_disjoint"
                if selection_contract in {
                    "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                    CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
                }
                else "scaffold_disjoint"
            ),
            **(
                {"neighbor_identity_policy_by_level": level_identity_policy}
                if selection_contract in {
                    "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                    CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
                }
                else {}
            ),
            "similarity_floor": None,
        },
        "experiment": {
            "mode": "full_flat",
            "source": "starling_v10_cache_matched",
            "neighbor_selector": selector,
            # Replay validation must not mistake cache-matched selection for the
            # legacy branch reranker. Its real provenance is the contract below.
            "retrieval_reranker": {"name": "none"},
            "cache_matched_selection": {
                "version": selection_contract,
                "reranking": reranking,
                "query_id": query_id,
                "query_audit": deepcopy(dict(query_audit)),
            },
        },
        "groups": [group],
        "coverage": retrieval_coverage(
            [group], min_similarity=0.0, top_k_per_group=len(neighbors)
        ),
    }


def _minimal_selected_evidence(
    payload: Mapping[str, Any],
    parent_id: str,
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        key: value
        for key, value in dict(payload.get("source_fields") or {}).items()
        if key != "extra_details"
    }
    endpoint = _first_field(
        fields,
        "standard_type",
        "endpoint",
        "endpoint_type",
        "endpoint_name",
        "parameter_name",
        "metric_type",
        "bioavailability_report_type",
    ) or str(payload["family_key"])
    evidence_row = {
        "evidence_source": str(payload["source_id"]),
        "source_record_id": str(payload["record_id"]),
        "molecule_chembl_id": parent_id,
        "canonical_smiles": str(payload["canonical_smiles"]),
        "group_id": str(payload["family_key"]),
        "tier": str(payload["progressive_level"]),
        "endpoint_group": str(payload["family_key"]),
        "standard_type": endpoint,
        "standard_relation": _first_field(fields, "standard_relation", "relation"),
        "standard_value": _first_field(
            fields,
            "measurement_text",
            "standard_value",
            "reported_value",
            "parameter_value",
            "quant_value",
            "value",
        ),
        "standard_units": _first_field(
            fields, "standard_units", "reported_units", "parameter_units", "quant_units", "unit", "unit_text"
        ),
        "evidence_text": _first_field(
            fields, "support_text", "assay_description", "description", "activity_comment"
        ) or f"Source-contracted {payload['measurement_kind']} evidence record.",
        "context_text": "; ".join(
            str(value)
            for value in (
                _first_field(fields, "species_or_population", "organism", "species"),
                _first_field(fields, "qualifying_conditions", "assay_context", "assay_model", "context"),
            )
            if value not in (None, "")
        ),
        "source_record_examples": [
            {
                "source_contract": {
                    "contract_version": payload["source_contract"]["contract_version"],
                    "source_or_simply_cleaned": {name: True for name in fields},
                },
                "source_fields": fields,
            }
        ],
    }
    evidence = minimal_evidence_from_row(evidence_row)
    evidence.setdefault("provenance", {})["retrieval"] = deepcopy(dict(retrieval))
    return evidence


def _source_semantic_card(
    payload: Mapping[str, Any],
    *,
    card_id: str,
    transfer_likelihood: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project every approved nonempty source field or account for its exclusion."""
    fields = dict(payload.get("source_fields") or {})
    approved_flags = (payload.get("source_contract") or {}).get(
        "source_or_simply_cleaned"
    )
    if not isinstance(approved_flags, Mapping):
        raise ValueError(f"Source contract is missing for record {payload.get('record_id')}")
    approved = {name for name, allowed in approved_flags.items() if allowed is True}
    if set(fields) != approved:
        raise ValueError(
            f"Source-field contract mismatch for record {payload.get('record_id')}: "
            f"missing={sorted(approved - set(fields))}, extra={sorted(set(fields) - approved)}"
        )

    nonempty = {name: value for name, value in fields.items() if _is_nonempty(value)}
    core: dict[str, Any] = {}
    core_sources: dict[str, str] = {}
    for output_name, candidates in CORE_FIELD_SOURCES.items():
        for source_name in candidates:
            if source_name in nonempty:
                core[output_name] = deepcopy(nonempty[source_name])
                core_sources[output_name] = source_name
                break

    consumed = set(core_sources.values())
    excluded = {
        name: (
            "evidence_molecule_name"
            if name in MOLECULE_NAME_FIELDS
            else "nonsemantic_metadata_or_duplicate_structure"
        )
        for name in nonempty
        if name in MOLECULE_NAME_FIELDS or name in NONSEMANTIC_SOURCE_FIELDS
    }
    experimental = {
        name: deepcopy(value)
        for name, value in nonempty.items()
        if name not in consumed
        and name not in excluded
        and name != "extra_details"
    }
    accounted = consumed | set(experimental) | set(excluded)
    if "extra_details" in nonempty:
        accounted.add("extra_details")
    if accounted != set(nonempty):
        raise ValueError(
            f"Unaccounted source fields for record {payload.get('record_id')}: "
            f"{sorted(set(nonempty) - accounted)}"
        )

    card: dict[str, Any] = {"card_id": card_id}
    for name in FLAT_V2_CARD_FIELDS[1:7]:
        if name in core:
            card[name] = core[name]
    if experimental:
        card["experimental_details"] = experimental
    if "extra_details" in nonempty:
        card["extra_details"] = deepcopy(nonempty["extra_details"])
    if "support_text" in core:
        card["support_text"] = core["support_text"]
    if transfer_likelihood is not None:
        card["transfer_likelihood"] = round(float(transfer_likelihood), 4)

    accounting = {
        "approved_nonempty_fields": list(nonempty),
        "core_field_sources": core_sources,
        "experimental_detail_fields": list(experimental),
        "extra_details_visible": "extra_details" in nonempty,
        "excluded_fields": excluded,
    }
    return card, accounting


def _set_consistent(target: dict[str, Any], name: str, value: Any) -> None:
    if value is None:
        return
    if name in target and target[name] != value:
        raise ValueError(
            f"Conflicting molecule-scoped {name}: {target[name]!r} != {value!r}"
        )
    target[name] = value


def _is_nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _first_field(fields: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = fields.get(name)
        if value not in (None, "", [], {}):
            return value
    return ""


def _similarity_bucket(similarity: float) -> str:
    for floor, name in (
        (0.95, "very_close_analog"),
        (0.80, "close_analog"),
        (0.60, "moderate_analog"),
        (0.40, "weak_analog"),
        (0.20, "distant_analog"),
    ):
        if similarity >= floor:
            return name
    return "very_distant_analog"


def _load_flat_context_candidates(
    args: argparse.Namespace,
    queries: Mapping[str, str],
    *,
    load_cache_policy: Any,
    load_candidates: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select cached UIDs, hydrate evidence rows, and compose the flat context."""
    selection_contract = (
        CONTEXT_V5_SELECTION_CONTRACT
        if args.prompt_version == CONTEXT_V5_PROMPT_VERSION
        else CONTEXT_V4_SELECTION_CONTRACT
    )
    policy = load_cache_policy(
        args.assay_transfer_cache,
        args.task,
        args.evaluation_subset,
        args.reranking,
        args.max_level,
    )
    molecules, ranked_later, audit = load_candidates(
        queries,
        task=args.task,
        subset=args.evaluation_subset,
        policy=policy,
        molecule_limit=args.l1_molecules,
        l1_limit=args.l1_records_per_molecule,
        later_limit={
            level: (
                args.record_limits_by_level[level]
                if args.prompt_version == CONTEXT_V5_PROMPT_VERSION else 100
            )
            for level in policy["stages"] if level != "L1"
        },
        tie_seed=args.ranking_tie_seed,
        min_contrast=args.l1_min_contrast,
        morgan_primary_parent_width=args.morgan_primary_parent_width,
        cache_pool="all",
    )

    later: dict[str, Any] = {}
    query_audits: dict[str, Any] = {}
    for query_id in queries:
        l1_record_ids = {
            str(record["record_id"])
            for molecule in molecules[query_id]
            for record in molecule.get("l1_records") or []
        }
        later[query_id] = {}
        query_audits[query_id] = {
            "L1": deepcopy(audit["query_audits"][query_id]["L1"])
        }
        for level, group in ranked_later[query_id].items():
            limit = args.record_limits_by_level[level]
            records = [
                row for row in group["records"]
                if str(row["record_id"]) not in l1_record_ids
            ][:limit]
            if args.prompt_version == CONTEXT_V5_PROMPT_VERSION and len(records) != limit:
                raise ValueError(
                    f"Requested {limit} records but {query_id}/{level} has "
                    f"{len(records)} after L1 exclusion"
                )
            selected_molecules = len({
                str(row["reference_molecule_id"]) for row in records
            })
            later[query_id][level] = {
                **group,
                "records": records,
                "selected_records": len(records),
                "selected_molecules": selected_molecules,
                "shortfall": max(0, limit - len(records)),
            }
            query_audits[query_id][level] = {
                key: value for key, value in later[query_id][level].items()
                if key not in {"records", "ranking_method", "allow_shortfall"}
            }

    later_content_ids = {
        level: content_id
        for level, content_id in audit.get("cache_content_ids", {}).items()
        if level != "L1"
    }
    active_levels = set(policy["stages"])
    return molecules, later, {
        "selection_policy": selection_contract,
        "inputs": dict(audit.get("inputs") or {}),
        "contract": {
            "policy": {
                "selection_contract": selection_contract,
                "uid_retrieval": policy,
            },
            "molecule_limit": args.l1_molecules,
            "l1_limit": args.l1_records_per_molecule,
            "later_limits": {
                level: args.record_limits_by_level[level]
                for level in active_levels if level != "L1"
            },
            "min_contrast": args.l1_min_contrast,
        },
        "cache_pool": "all",
        "query_identities": audit["query_identities"],
        "query_audits": query_audits,
        "neighbor_identity_policy": audit["neighbor_identity_policy"],
        "neighbor_identity_policy_by_level": {
            level: value
            for level, value in audit["neighbor_identity_policy_by_level"].items()
            if level in active_levels
        },
        "similarity_floor": None,
        "scaffold_overlap": 0,
        "parent_overlap": 0,
        "l1_cache": {
            "bundle": str(args.assay_transfer_cache),
            "selection_policy": audit["selection_policy"],
            "cache_index": audit.get("cache_index"),
            "content_id": audit.get("cache_content_ids", {}).get("L1"),
            "evidence_manifest": audit.get("evidence_manifest"),
        },
        "later_cache": (
            {
                "bundle": str(args.assay_transfer_cache),
                "selection_policy": audit["selection_policy"],
                "cache_index": audit.get("cache_index"),
                "content_ids": later_content_ids,
            }
            if later_content_ids else None
        ),
    }


def _load_molecule_descriptions(mode: str, version: str) -> tuple[dict[str, str], dict[str, Any]]:
    if mode == "none":
        return {}, {"mode": "none"}
    import pyarrow.parquet as pq

    column = MOLECULE_DESCRIPTION_COLUMNS[mode]
    path, expected_sha256 = MOLECULE_DESCRIPTION_ARTIFACTS[version]
    path = path.resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Molecule description cache hash mismatch: {path}")
    table = pq.read_table(path, columns=["canonical_smiles", column, "error"])
    values: dict[str, str] = {}
    for row in table.to_pylist():
        smiles = str(row.get("canonical_smiles") or "")
        description = row.get(column)
        if smiles in values:
            raise ValueError(f"Duplicate molecule description SMILES: {smiles}")
        if smiles and isinstance(description, str) and description.strip() and not row.get("error"):
            values[smiles] = description.strip()
    return values, {
        "mode": mode,
        "cache_version": version,
        "column": column,
        "path": str(path),
        "sha256": expected_sha256,
        "row_count": table.num_rows,
    }


def _attach_molecule_descriptions(
    retrieval: dict[str, Any], values: Mapping[str, str]
) -> list[str]:
    if not values:
        return []
    missing = []
    query = retrieval["query"]
    query_smiles = str(query.get("canonical_smiles") or "")
    if query_smiles in values:
        query["molecule_description"] = values[query_smiles]
    else:
        missing.append(query_smiles)
    for neighbor in retrieval["groups"][0]["neighbors"]:
        smiles = str(neighbor.get("canonical_smiles") or "")
        if smiles in values:
            neighbor["molecule_description"] = values[smiles]
        else:
            missing.append(smiles)
    return sorted(set(missing))
def _materialize_cache_matched_retrievals(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    from predict.retrieval.assay_reranking.cache_matched import (
        load_cache_policy,
        load_candidates,
    )

    records = read_jsonl(args.input_jsonl)
    indices = _selected_indices(args, len(records))
    queries = {
        str(records[index]["benchmark_row_id"]): str(records[index]["drug"])
        for index in indices
    }
    policy = None if args.prompt_version in CONTEXT_PROMPT_VERSIONS else load_cache_policy(
        args.assay_transfer_cache,
        args.task,
        args.evaluation_subset,
        args.reranking,
        args.max_level,
    )
    batch_dir = args.batch_root / args.batch_id
    source_batch = batch_dir / "cache_matched_retrieval"
    existing_path = source_batch / "manifest.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        expected = {
            "harness_version": args.harness_version,
            "prompt_version": args.prompt_version,
            "task": args.task,
            "evaluation_subset": args.evaluation_subset,
            "reranking": args.reranking,
            "record_pool": args.record_pool,
            "cache_pool": RECORD_POOLS[args.record_pool],
            "indices": indices,
            "input_jsonl": str(args.input_jsonl),
            "input_sha256": sha256_file(args.input_jsonl),
            "evidence_library": str(args.evidence_library),
            "level_mapper": str(args.level_mapper),
            "assay_transfer_cache": str(args.assay_transfer_cache),
            "gold_context_mapping": (
                str(args.gold_context_mapping) if args.task == "bbb_martins" else ""
            ),
            "allow_frozen_l1_vote_scores": args.allow_frozen_l1_vote_scores,
            "l1_molecules": args.l1_molecules,
            "l1_records_per_molecule": args.l1_records_per_molecule,
            "records_per_level": args.records_per_level,
            "record_limits_by_level": args.record_limits_by_level,
            "ranking_tie_seed": args.ranking_tie_seed,
            "max_level": args.max_level,
        }
        if args.prompt_version in CONTEXT_PROMPT_VERSIONS:
            expected.update(
                layout=args.layout,
                query_prior=args.query_prior,
                l1_min_contrast=args.l1_min_contrast,
                morgan_primary_parent_width=args.morgan_primary_parent_width,
                molecule_description_mode=args.molecule_description_mode,
                molecule_description_cache_version=args.molecule_description_cache_version,
            )
        if args.prompt_version == JOSEPH_PROMPT_VERSION:
            expected.update(
                evidence_projection=EVIDENCE_PROJECTION,
                extra_details_policy=EXTRA_DETAILS_POLICY,
                molecule_name_visible=False,
                group_tools_enabled=False,
            )
        mismatches = {
            key: {"expected": value, "observed": existing.get(key)}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        selector_path = Path(__file__).resolve().parents[2] / "retrieval/assay_reranking/cache_matched.py"
        if (existing.get("selector_code") or {}).get("sha256") != sha256_file(selector_path):
            mismatches["selector_code"] = "changed"
        selection_policy = existing.get("selection_audit", {}).get("selection_policy")
        if selection_policy in {CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT}:
            for name in ("ranked_uid_retrieval.py",):
                reader_path = selector_path.with_name(name)
                recorded = (existing.get("cache_reader_codes") or {}).get(name) or {}
                if recorded.get("sha256") != sha256_file(reader_path):
                    mismatches[f"cache_reader_codes[{name}]"] = "changed"
        if selection_policy in SQLITE_SELECTION_CONTRACTS:
            reader_path = selector_path.with_name(SQLITE_SELECTION_CONTRACTS[selection_policy])
            if (existing.get("cache_reader_code") or {}).get("sha256") != sha256_file(reader_path):
                mismatches["cache_reader_code"] = "changed"
        if (existing.get("prompt_assets") or {}).get("sha256") != prompt_asset_manifest(
            args.prompt_version
        )["sha256"]:
            mismatches["prompt_assets"] = "changed"
        retrieval_hashes = existing.get("retrieval_sha256_by_index") or {}
        if set(retrieval_hashes) != {str(index) for index in indices}:
            mismatches["retrieval_sha256_by_index"] = "incomplete_or_unexpected_indices"
        for index, expected_hash in retrieval_hashes.items():
            retrieval_path = source_batch / "runs" / f"{source_batch.name}_idx{int(index):05d}" / "retrieval.json"
            if not retrieval_path.exists() or sha256_file(retrieval_path) != expected_hash:
                mismatches[f"retrieval[{index}]"] = "missing_or_changed"
        if mismatches:
            raise ValueError(
                "Existing flat selection is immutable and differs from this request: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return source_batch, existing

    if args.prompt_version in CONTEXT_PROMPT_VERSIONS:
        molecules, later, audit = _load_flat_context_candidates(
            args,
            queries,
            load_cache_policy=load_cache_policy,
            load_candidates=load_candidates,
        )
    else:
        molecules, later, audit = load_candidates(
            queries,
            task=args.task,
            subset=args.evaluation_subset,
            library=args.evidence_library,
            mapper=args.level_mapper,
            policy=policy,
            molecule_limit=args.l1_molecules,
            l1_limit=args.l1_records_per_molecule,
            later_limit=(
                {
                    level: args.record_limits_by_level[level]
                    for level in policy["stages"]
                    if level != "L1"
                }
                if policy.get("selection_contract") in {
                    "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1"
                }
                else args.records_per_level
            ),
            tie_seed=args.ranking_tie_seed,
            gold_context_mapping=(
                args.gold_context_mapping if args.task == "bbb_martins" else None
            ),
            allow_frozen_l1_vote_scores=args.allow_frozen_l1_vote_scores,
            cache_pool=RECORD_POOLS[args.record_pool],
        )
    run_root = source_batch / "runs"
    retrieval_hashes = {}
    molecule_descriptions, molecule_description_receipt = (
        _load_molecule_descriptions(
            args.molecule_description_mode,
            args.molecule_description_cache_version,
        )
        if args.prompt_version in CONTEXT_PROMPT_VERSIONS
        else ({}, {"mode": "none"})
    )
    description_missing: set[str] = set()
    for index in indices:
        record = records[index]
        query_id = str(record["benchmark_row_id"])
        retrieval = cache_matched_flat_retrieval(
            query_id,
            str(record["drug"]),
            molecules[query_id],
            later[query_id],
            task=args.task,
            reranking=args.reranking,
            query_audit=audit["query_audits"][query_id],
            query_identity=(audit.get("query_identities") or {}).get(query_id),
            prompt_version=args.prompt_version,
            selection_contract=audit["selection_policy"],
        )
        description_missing.update(
            _attach_molecule_descriptions(retrieval, molecule_descriptions)
        )
        run_dir = run_root / f"{source_batch.name}_idx{index:05d}"
        path = run_dir / "retrieval.json"
        write_json_atomic(path, retrieval)
        retrieval_hashes[str(index)] = sha256_file(path)

    contract = deepcopy(audit["contract"])
    contract.pop("inputs", None)
    contract["policy"] = deepcopy(contract["policy"])
    contract["policy"].pop("inputs", None)
    compact_audit = {
        "selection_policy": audit["selection_policy"],
        "inputs": audit["inputs"],
        "contract": contract,
        "query_audits": audit["query_audits"],
        "cache_pool": audit["cache_pool"],
        "neighbor_identity_policy": audit.get("neighbor_identity_policy"),
        "neighbor_identity_policy_by_level": audit.get(
            "neighbor_identity_policy_by_level"
        ),
        "similarity_floor": audit["similarity_floor"],
        "scaffold_overlap": audit["scaffold_overlap"],
        "parent_overlap": audit["parent_overlap"],
    }
    if audit["selection_policy"] in {CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT}:
        compact_audit.update(
            query_identities=audit["query_identities"],
            l1_cache=audit["l1_cache"],
            later_cache=audit["later_cache"],
        )
    elif audit["selection_policy"] in SQLITE_SELECTION_CONTRACTS:
        compact_audit.update(cache_capacities=audit["cache_capacities"],
                             query_identities=audit["query_identities"])
        if audit["selection_policy"] in {
            "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1"
        }:
            compact_audit.update(
                cache_index=audit["cache_index"],
                cache_content_ids=audit["cache_content_ids"],
            )
        else:
            compact_audit.update(
                cache_version=audit["cache_version"],
                cache_content_id=audit["cache_content_id"],
                cache_assignment_counts=audit["cache_assignment_counts"],
                cache_record_count=audit["cache_record_count"],
            )
    else:
        compact_audit.update(
            v9=audit["v9"],
            l1_pool_size=audit["l1_pool_size"],
            gold_context_mapping=audit["gold_context_mapping"],
            library_compatibility=audit["library_compatibility"],
            cache_versions=[
                {
                    "path": path,
                    "schema_version": version.get("schema_version"),
                    "status": version.get("status"),
                    "morgan_pool_size": version.get("morgan_pool_size"),
                    "models": version.get("models"),
                    "cache_sha256": version.get("cache_sha256"),
                }
                for path, version in audit["cache_versions"].items()
            ],
            mapped_level_counts=audit["mapped_level_counts"],
            unmapped_record_count=len(audit["unmapped_record_ids"]),
            unmapped_record_ids_sha256=hashlib.sha256(
                json.dumps(sorted(audit["unmapped_record_ids"])).encode("utf-8")
            ).hexdigest(),
        )
    manifest = {
        "harness_version": args.harness_version,
        "prompt_version": args.prompt_version,
        "task": args.task,
        "evaluation_subset": args.evaluation_subset,
        "reranking": args.reranking,
        "record_pool": args.record_pool,
        "cache_pool": RECORD_POOLS[args.record_pool],
        "indices": indices,
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": sha256_file(args.input_jsonl),
        "evidence_library": str(args.evidence_library),
        "level_mapper": str(args.level_mapper),
        "assay_transfer_cache": str(args.assay_transfer_cache),
        "gold_context_mapping": (
            str(args.gold_context_mapping) if args.task == "bbb_martins" else ""
        ),
        "allow_frozen_l1_vote_scores": args.allow_frozen_l1_vote_scores,
        "l1_molecules": args.l1_molecules,
        "l1_records_per_molecule": args.l1_records_per_molecule,
        "records_per_level": args.records_per_level,
        "record_limits_by_level": args.record_limits_by_level,
        "ranking_tie_seed": args.ranking_tie_seed,
        "max_level": args.max_level,
        "retrieval_sha256_by_index": retrieval_hashes,
        "selector_code": {
            "path": str(Path(__file__).resolve().parents[2] / "retrieval/assay_reranking/cache_matched.py"),
            "sha256": sha256_file(
                Path(__file__).resolve().parents[2] / "retrieval/assay_reranking/cache_matched.py"
            ),
        },
        "prompt_assets": prompt_asset_manifest(args.prompt_version),
        "selection_audit": compact_audit,
    }
    if args.prompt_version in CONTEXT_PROMPT_VERSIONS:
        molecule_description_receipt.update(
            missing_policy="omit",
            missing_selected_count=len(description_missing),
            missing_selected_smiles=sorted(description_missing),
        )
        manifest.update(
            layout=args.layout,
            query_prior=args.query_prior,
            l1_min_contrast=args.l1_min_contrast,
            morgan_primary_parent_width=args.morgan_primary_parent_width,
            molecule_description_mode=args.molecule_description_mode,
            molecule_description_cache_version=args.molecule_description_cache_version,
            molecule_description=molecule_description_receipt,
            evidence_projection=EVIDENCE_PROJECTION,
            extra_details_policy=EXTRA_DETAILS_POLICY,
            molecule_name_visible=False,
            group_tools_enabled=False,
        )
    if audit["selection_policy"] in {CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT}:
        reader_root = Path(__file__).resolve().parents[2] / "retrieval/assay_reranking"
        manifest["cache_reader_codes"] = {
            name: {
                "path": str(reader_root / name),
                "sha256": sha256_file(reader_root / name),
            }
            for name in ("ranked_uid_retrieval.py",)
        }
    elif audit["selection_policy"] in SQLITE_SELECTION_CONTRACTS:
        reader_path = (
            Path(__file__).resolve().parents[2]
            / "retrieval/assay_reranking"
            / SQLITE_SELECTION_CONTRACTS[audit["selection_policy"]]
        )
        manifest["cache_reader_code"] = {
            "path": str(reader_path),
            "sha256": sha256_file(reader_path),
        }
    if args.prompt_version == JOSEPH_PROMPT_VERSION:
        manifest.update(
            evidence_projection=EVIDENCE_PROJECTION,
            extra_details_policy=EXTRA_DETAILS_POLICY,
            molecule_name_visible=False,
            group_tools_enabled=False,
        )
    manifest["selection_contract_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "contract": contract,
                "inputs": compact_audit["inputs"],
                "selector_code_sha256": manifest["selector_code"]["sha256"],
                "cache_reader_code_sha256": (
                    (manifest.get("cache_reader_code") or {}).get("sha256")
                ),
                "cache_reader_codes_sha256": {
                    name: value["sha256"]
                    for name, value in (manifest.get("cache_reader_codes") or {}).items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_json_atomic(source_batch / "manifest.json", manifest)
    return source_batch, manifest


def _selected_indices(args: argparse.Namespace, size: int) -> list[int]:
    if args.indices:
        output = []
        for token in args.indices:
            if "-" in token:
                start, end = token.split("-", 1)
                output.extend(range(int(start), int(end) + 1))
            else:
                output.append(int(token))
    else:
        end = size if args.limit == 0 else min(size, args.start + args.limit)
        output = list(range(args.start, end))
    output = sorted(dict.fromkeys(output))
    if not output or any(index < 0 or index >= size for index in output):
        raise SystemExit(f"Query indices must be within 0..{size - 1}")
    return output


def _validate_query_prior_batch(
    prior_batch: Path,
    records: list[Mapping[str, Any]],
    indices: list[int],
) -> None:
    """Bind positional legacy prior artifacts to current stable query identities."""
    for index in indices:
        run_dir = prior_batch / "runs" / f"{prior_batch.name}_idx{index:05d}"
        retrieval_path = run_dir / "retrieval.json"
        single_path = run_dir / "single_molecule_reasoning_output.json"
        if not retrieval_path.is_file() or not single_path.is_file():
            raise ValueError(f"Cached query prior is incomplete: {run_dir}")
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        current_smiles = str(records[index].get("drug") or "")
        cached_smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
        if cached_smiles != current_smiles:
            raise ValueError(
                f"Cached query prior identity differs at index {index}: "
                f"{cached_smiles!r} != {current_smiles!r}"
            )
        if json.loads(single_path.read_text(encoding="utf-8")).get("status") != "ok":
            raise ValueError(f"Cached query prior is not successful: {single_path}")


def mode_main(mode: str, argv: list[str] | None = None) -> int:
    """Load the shared launcher lazily so batch code can import flat contracts."""
    from predict.harnesses.branches.runner import mode_main as run_mode

    return run_mode(mode, argv)


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument(
        "--harness-version",
        choices=(
            PUBLIC_HARNESS_VERSION,
            CONTEXT_V4_HARNESS_VERSION,
            CONTEXT_V5_HARNESS_VERSION,
            JOSEPH_V1_HARNESS_VERSION,
            LEGACY_HARNESS_VERSION,
        ),
        default=PUBLIC_HARNESS_VERSION,
    )
    selected, remaining = selector.parse_known_args(argv)
    if selected.harness_version == LEGACY_HARNESS_VERSION:
        if "--flat-prompt-version" in remaining:
            selector.error("Tianang's harness version already fixes its prompt bundle")
        return mode_main(
            "full_flat",
            [*remaining, "--flat-prompt-version", TIANANG_PROMPT_VERSION],
        )
    return _joseph_main(["--harness-version", selected.harness_version, *remaining])


def _joseph_main(argv: list[str]) -> int:
    from predict.retrieval.assay_reranking.cache_matched import (
        DEFAULT_CACHE_BUNDLE,
        DEFAULT_GOLD_CONTEXT_MAPPING,
    )

    parser = argparse.ArgumentParser(
        description="Run a cache-matched Joseph flat harness.", allow_abbrev=False
    )
    parser.add_argument(
        "--harness-version", choices=tuple(JOSEPH_HARNESS_PROMPTS),
        default=PUBLIC_HARNESS_VERSION,
    )
    parser.add_argument("--task", required=True, choices=tuple(TASKS))
    parser.add_argument(
        "--reranking", choices=(*CONTEXT_V4_VARIANTS, JOINT_VARIANT),
        default=ASSAY_TRANSFER_VARIANT,
    )
    parser.add_argument("--layout", choices=CONTEXT_V4_LAYOUTS, default="global")
    parser.add_argument(
        "--query-prior", choices=("cached", "none"), default="cached"
    )
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_QUERY_PRIOR_ROOT)
    parser.add_argument("--l1-min-contrast", type=int, default=3)
    parser.add_argument(
        "--morgan-primary-parent-width", type=int, choices=(15, 25, 50, 100), default=25
    )
    parser.add_argument(
        "--molecule-description-mode",
        choices=("none", *MOLECULE_DESCRIPTION_COLUMNS),
        default="none",
    )
    parser.add_argument(
        "--molecule-description-cache-version",
        choices=tuple(MOLECULE_DESCRIPTION_ARTIFACTS),
        default="v1",
    )
    parser.add_argument("--assay-transfer-cache", type=Path, default=DEFAULT_CACHE_BUNDLE)
    parser.add_argument(
        "--record_pool", "-record_pool", choices=tuple(RECORD_POOLS),
        default="all",
    )
    parser.add_argument("--evaluation-subset", choices=("valid", "test"), default="valid")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--evidence-library", type=Path)
    parser.add_argument(
        "--level-mapper", type=Path,
        default=Path("data/evidence_libraries/level_mappings.v1.json"),
    )
    parser.add_argument("--gold-context-mapping", type=Path, default=DEFAULT_GOLD_CONTEXT_MAPPING)
    parser.add_argument("--allow-frozen-l1-vote-scores", action="store_true")
    parser.add_argument("--l1-molecules", type=int, default=10)
    parser.add_argument("--l1-records-per-molecule", type=int, default=10)
    parser.add_argument("--records-per-level", type=int, default=50)
    parser.add_argument(
        "--level-record-limit", action="append", default=[], metavar="LEVEL=K",
        help="Override --records-per-level for one later level; repeat as needed.",
    )
    parser.add_argument("--ranking-tie-seed", type=int, default=0)
    parser.add_argument("--max-level", type=int, default=0)
    parser.add_argument("--indices", nargs="*")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-root", "--output-root", type=Path)
    parser.add_argument("--batch-id", default="")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Materialize and verify flat retrieval without issuing model or tool calls.",
    )
    args, remaining = parser.parse_known_args(argv)
    overrides = {}
    for value in args.level_record_limit:
        level, separator, raw_limit = value.partition("=")
        if (not separator or level not in {"L2", "L3", "L4", "L5", "L6"}
                or level in overrides):
            parser.error('--level-record-limit requires unique LEVEL=K values for L2-L6')
        try:
            overrides[level] = int(raw_limit)
        except ValueError:
            parser.error('--level-record-limit values must be integers')
    if any(value < 1 for value in overrides.values()):
        parser.error('--level-record-limit values must be positive')
    args.record_limits_by_level = {
        f"L{level}": overrides.get(f"L{level}", args.records_per_level)
        for level in range(2, TASKS[args.task] + 1)
    }
    args.prompt_version = JOSEPH_HARNESS_PROMPTS[args.harness_version]
    forbidden = {
        "--experiment-mode", "--retrieval-strategy", "--retrieval-source",
        "--index", "--neighbor-identity-policy", "--morgan-neighbor-selector",
        "--neighbor-selector", "--flat-prompt-version", "--rerank-catalog",
        "--rerank-cache", "--rerank-candidate-manifest", "--rerank-cache-mode",
        "--assay-transfer-profile", "--assay-transfer-min-score",
        "--assay-transfer-diversity-mode", "--assay-transfer-selection-unit",
        "--enable-assay-transfer-scores", "--min-similarity", "--top-k-per-group",
    }
    used = sorted({token.split("=", 1)[0] for token in remaining} & forbidden)
    if used:
        parser.error(
            f"{args.harness_version} fixes cache-matched retrieval and cannot combine with: "
            + ", ".join(used)
        )
    if min(args.l1_molecules, args.l1_records_per_molecule, args.records_per_level) < 1:
        parser.error("selection limits must be positive")
    if min(args.start, args.limit, args.max_level) < 0:
        parser.error("start, limit, and max-level must be non-negative")
    if args.reranking == JOINT_VARIANT and args.l1_molecules != 10:
        parser.error("joint L1 requires ten molecule slots: five per panel, no refill")
    if args.prompt_version in CONTEXT_PROMPT_VERSIONS:
        remaining_flags = {token.split("=", 1)[0] for token in remaining}
        if "--single-analysis-source-batch" in remaining_flags:
            parser.error("query-prior reuse is owned by --query-prior and --prior-root")
        if args.reranking not in CONTEXT_V4_VARIANTS:
            parser.error(f"{args.harness_version} does not support joint reranking")
        if args.l1_min_contrast < 0:
            parser.error("--l1-min-contrast must be non-negative")
        if args.evaluation_subset != "valid":
            parser.error("V10.4 context L1 caches currently cover the valid split only")
    elif args.reranking not in VARIANTS:
        parser.error(f"{args.harness_version} does not support {args.reranking}")
    if args.assay_transfer_cache == DEFAULT_CACHE_BUNDLE:
        if args.record_pool != "all":
            parser.error("ranked_level_retrieval.v2 requires --record_pool all")
        if args.reranking == JOINT_VARIANT:
            parser.error("ranked_level_retrieval.v2 supports Morgan or assay-transfer only")
    canonical = split_path(args.task, args.evaluation_subset).with_name(
        f"{args.evaluation_subset}_molecule_condition_labels.jsonl"
    ).resolve()
    if args.input_jsonl is not None and args.input_jsonl.resolve() != canonical:
        parser.error(
            f"{args.harness_version} accepts only the official conditioned benchmark split"
        )
    args.input_jsonl = canonical
    args.evidence_library = (
        args.evidence_library
        or Path("data/evidence_libraries") / args.task / "v10"
    ).resolve()
    args.level_mapper = args.level_mapper.resolve()
    args.assay_transfer_cache = args.assay_transfer_cache.resolve()
    args.gold_context_mapping = args.gold_context_mapping.resolve()
    args.prior_root = args.prior_root.resolve()
    if args.batch_root is None and args.prompt_version in CONTEXT_PROMPT_VERSIONS:
        args.batch_root = (
            Path("outputs/paper/assay_transfer_harness/joseph")
            / "flat_context_v4"
            / args.reranking
            / ("with_query_prior" if args.query_prior == "cached" else "no_query_prior")
        )
    elif args.batch_root is None:
        args.batch_root = (
            Path("outputs/paper/assay_transfer_harness/joseph")
            / args.harness_version.replace("-", "_")
            / args.task / args.evaluation_subset / args.reranking / args.record_pool
        )
    args.batch_root = args.batch_root.resolve()
    args.batch_id = args.batch_id or time.strftime(
        f"k{args.l1_molecules}_m{args.l1_min_contrast}_%Y%m%d_%H%M%S"
        if args.prompt_version in CONTEXT_PROMPT_VERSIONS
        else f"{args.harness_version.replace('-', '_')}_%Y%m%d_%H%M%S"
    )

    source_batch, manifest = _materialize_cache_matched_retrievals(args)
    prior_batch = None
    if args.prompt_version in CONTEXT_PROMPT_VERSIONS and args.query_prior == "cached":
        prior_batch = args.prior_root / args.task / f"{args.task}__none"
        if not prior_batch.is_dir():
            parser.error(f"cached query-prior batch does not exist: {prior_batch}")
        _validate_query_prior_batch(
            prior_batch,
            read_jsonl(args.input_jsonl),
            manifest["indices"],
        )
    if args.prepare_only:
        print(json.dumps({"retrieval_source": str(source_batch), "manifest": manifest}, indent=2))
        return 0
    if (
        args.prompt_version in CONTEXT_PROMPT_VERSIONS
        and "--max-tokens" not in {token.split("=", 1)[0] for token in remaining}
    ):
        parser.error(
            f"{args.harness_version} requires an explicit reviewed --max-tokens"
        )
    forwarded = [
        *remaining,
        "--task", args.task,
        "--input-jsonl", str(args.input_jsonl),
        "--batch-root", str(args.batch_root),
        "--batch-id", args.batch_id,
        "--retrieval-replay-source-batch", str(source_batch),
        "--flat-selection-manifest", str(source_batch / "manifest.json"),
        "--flat-prompt-version", args.prompt_version,
        "--flat-reranking", args.reranking,
        "--flat-layout", args.layout,
        "--flat-query-prior", args.query_prior,
        "--retrieval-strategy", "morgan_fingerprint",
        "--neighbor-identity-policy", "scaffold_disjoint",
        "--min-similarity", "0",
    ]
    if args.prompt_version in {JOSEPH_PROMPT_VERSION, *CONTEXT_PROMPT_VERSIONS} and "--disable-flat-tools" not in remaining:
        forwarded.append("--disable-flat-tools")
    if prior_batch is not None:
        forwarded.extend(["--single-analysis-source-batch", str(prior_batch)])
    if args.indices:
        forwarded.extend(["--indices", *args.indices])
    else:
        forwarded.extend(["--start", str(args.start), "--limit", str(args.limit)])
    return mode_main("full_flat", forwarded)


if __name__ == "__main__":
    raise SystemExit(
        "Use: python -m predict.harnesses.branches --organization flat ..."
    )
