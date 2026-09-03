"""Run Bioavailability_Ma analog reasoning with group-level parallel LLM calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

from predict.retrieval.assay_reranking.v9 import (
    TEMPLATE_PROFILE as V9_TEMPLATE_PROFILE,
    V9CachedAssayReranker,
    default_cache_paths as v9_default_cache_paths,
    model_profile as v9_model_profile,
)
from predict.harnesses.branches.assay_transfer_prompt import (
    SCORED_NEIGHBORS_POLICY_NAME,
    prepare_assay_transfer_selected_neighbors,
    public_assay_transfer_score,
    scored_neighbors_prompt_enabled,
    validate_scored_neighbors_configuration,
)
from predict.harnesses.branches.assay_transfer import (
    ASSAY_TRANSFER_DIVERSITY_MODES,
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    ASSAY_TRANSFER_SELECTION_UNITS,
    assay_transfer_selection_policy,
    validate_assay_transfer_diversity,
    validate_assay_transfer_records_per_molecule,
)
from predict.harnesses.branches.retrieval_cli import add_retrieval_strategy_args
from predict.harnesses.branches.retrieval import (
    ASSAY_TRANSFER_TOOL_STRATEGY,
    EXPERIMENT_MODES,
    retrieve_experiment_view,
)
from predict.llm_io.evidence import evidence_for_group_llm, evidence_for_llm
from predict.harnesses.branches.reasoning.coverage import (
    NEIGHBOR_CONTEXT_PROFILES,
    STANDARD_NEIGHBOR_CONTEXT,
)
from predict.utils.files import ensure_dir
from predict.harnesses.branches.reasoning.final_evidence import (
    SUMMARY_ONLY,
    add_final_evidence_surface_argument,
    build_final_evidence_fields,
    compact_group_reasoning_outputs,
    final_evidence_instructions,
    prepare_resumed_final_inputs,
)
from predict.harnesses.branches.reasoning.final_decision import (
    EVIDENCE_STATES,
    STANDARD_FINAL_DECISION,
    TrainRatioPrior,
    add_final_decision_profile_argument,
    build_final_decision_prompt,
    final_decision_validation_errors,
)
from predict.harnesses.branches.reasoning.identity_blind import (
    expose_neighbor_smiles_only,
    prepare_reasoning_retrieval,
    query_without_prefetched_tools,
    sanitize_identity_blind_branch_outputs,
)
from predict.harnesses.branches.analogous_flat_prompt import (
    PROMPT_IDENTITY_VIEW,
    prompt_provenance as analogous_flat_prompt_provenance,
    reason_final as reason_analogous_flat_final,
    reason_group as reason_analogous_flat_group,
)
from predict.utils.json import parse_json_content
from predict.retrieval.policies import (
    NEIGHBOR_SELECTORS,
    SIMILARITY_SELECTOR,
)
from predict.retrieval.policies import NEIGHBOR_IDENTITY_POLICIES
from predict.llm_engine.client import OpenAICompatibleClient
from predict.llm_engine.pool import load_env_file as _load_env
from predict.tasks.prompt_profiles import (
    prompt_profile_from_manifest,
    require_matching_prompt_profiles,
)
from predict.harnesses.branches.payload import (
    attach_external_condition,
    clean_exact_match as _clean_exact_match,
    clean_shared_assay_context as _clean_shared_assay_context,
    llm_evidence_query_payload as _llm_evidence_query_payload,
    llm_query_payload as _llm_query_payload,
)
from predict.harnesses.branches.artifacts import (
    read_jsonl_record as _read_jsonl_record,
    write_trace_jsonl,
)
from predict.harnesses.branches.reasoning.calls import (
    bound_group_prompt_payload,
    call_group_branch,
    call_single_molecule_branch,
    load_frozen_single_analysis,
)
from predict.llm_io.response import (
    call_with_json_validation,
    structured_response_is_valid,
    validated_branch_content,
)
from predict.harnesses.branches.replay import load_retrieval_replay
from predict.harnesses.branches.reuse import load_reusable_group_outputs
from predict.harnesses.branches.tasks.bioavailability_ma.context import (
    DEFAULT_CHEMBL_SQLITE,
    enrich_retrieval_with_chembl_context,
)
from predict.tasks.bioavailability_ma.constants import BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT
from predict.harnesses.branches.tasks.bioavailability_ma.config import (
    get_source_config,
)
from predict.harnesses.branches.tasks.bioavailability_ma.group_prompt import (
    DEFAULT_GROUP_PROMPT_VERSION,
    GROUP_OUTPUT_SCHEMA_PROFILES,
    GROUP_PROMPT_VERSIONS,
    SUPPORTED_FORMATS as TEXT_GROUP_PROMPT_FORMATS,
    build_final_messages,
    build_group_messages,
    final_prompt_provenance,
    group_output_schema_provenance,
    group_output_validation,
    group_prompt_provenance,
    group_system_message,
    instruction_file_provenance,
)
from predict.tasks.bioavailability_ma.prompts import (
    BIOAVAILABILITY_PROMPT_PROFILES,
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
    FINAL_SCHEMA,
    GROUP_SCHEMA,
    SINGLE_SCHEMA,
    get_bioavailability_prompt_profile,
)
from predict.retrieval.retrieve import load_index

GROUP_PROMPT_FORMATS = ("legacy", *TEXT_GROUP_PROMPT_FORMATS)


DEFAULT_INPUT = "data/gold_labels/legacy/processed/Bioavailability_Ma/test.jsonl"
DEFAULT_INDEX = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl"
V9_DEFAULT_INDEX = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v7/08_neighbor_index/scaffold"
)
DEFAULT_OUT_ROOT = "outputs/chembl_tool/tasks/bioavailability_ma/reasoning/single_runs"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TOOL_SERVICE_URL = "http://127.0.0.1:8765"
V9_DEFAULT_PATHS = v9_default_cache_paths("bioavailability_ma")
V9_MODEL_PROFILE = v9_model_profile("bioavailability_ma")
_write_trace_jsonl = partial(
    write_trace_jsonl,
    prediction_field="bioavailability_prediction",
)
TRAIN_RATIO_PRIOR = TrainRatioPrior(
    dataset_lineage="record_supported_v2.bioavailability_canonical_direct.v2",
    split="scaffold/train",
    positive_count=1213,
    negative_count=461,
    positive_label="high",
    negative_label="low",
)


GROUP_REASONING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mmp_structure_compare",
            "description": (
                "Compare the query molecule to one neighbor using Morgan Tanimoto, MCS coverage, "
                "and mmpdb matched-pair transformation. Use this to judge structural transferability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "reference_smiles": {"type": "string", "description": "The neighbor/reference molecule SMILES."},
                    "max_mmp_alternatives": {
                        "type": "integer",
                        "description": "Maximum matched-pair alternatives to return.",
                        "default": 5,
                    },
                    "mcs_timeout_s": {
                        "type": "integer",
                        "description": "MCS search timeout in seconds.",
                        "default": 5,
                    },
                },
                "required": ["query_smiles", "reference_smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "properties_compare",
            "description": (
                "Compare query and neighbor molecule properties, including RDKit descriptors and MolGpKa/logD "
                "features. Use this to assess whether property changes affect oral bioavailability evidence transferability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "reference_smiles": {"type": "string", "description": "The neighbor/reference molecule SMILES."},
                    "logd_ph": {
                        "type": "number",
                        "description": "pH for logD comparison.",
                        "default": 7.4,
                    },
                },
                "required": ["query_smiles", "reference_smiles"],
            },
        },
    },
]

SINGLE_MOLECULE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "molecule_properties",
            "description": (
                "Compute the query molecule's RDKit descriptors, MolGpKa pKa/logD features, "
                "and functional-group profile. This is the only tool allowed for single-molecule analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_smiles": {"type": "string", "description": "The query molecule SMILES."},
                    "logd_ph": {
                        "type": "number",
                        "description": "pH for logD estimation.",
                        "default": 7.4,
                    },
                },
                "required": ["query_smiles"],
            },
        },
    }
]

SINGLE_MOLECULE_TOOL_CHOICE = {"type": "function", "function": {"name": "molecule_properties"}}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_analogous_reasoning_only(args)
    if args.assay_transfer_profile == "v9_direct_gold":
        if args.assay_transfer_template_profile != V9_TEMPLATE_PROFILE:
            raise SystemExit(
                "v9_direct_gold requires --assay-transfer-template-profile "
                f"{V9_TEMPLATE_PROFILE}"
            )
    elif args.assay_transfer_template_profile == V9_TEMPLATE_PROFILE:
        raise SystemExit(
            f"{V9_TEMPLATE_PROFILE} requires --assay-transfer-profile v9_direct_gold"
        )
    # --retrieval-strategy is the source of truth; it locks the compatible group-prompt-format.
    if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        if args.group_prompt_format != "assay_transfer_tool":
            raise SystemExit(
                "--retrieval-strategy assay_transfer_tool requires "
                "--group-prompt-format assay_transfer_tool"
            )
        if args.morgan_neighbor_selector != SIMILARITY_SELECTOR:
            raise SystemExit(
                "--morgan-neighbor-selector applies only to "
                "--retrieval-strategy morgan_fingerprint"
            )
    else:  # morgan_fingerprint
        if args.group_prompt_format == "assay_transfer_tool":
            raise SystemExit(
                "--group-prompt-format assay_transfer_tool requires "
                "--retrieval-strategy assay_transfer_tool"
            )
        if args.assay_transfer_min_score is not None:
            raise SystemExit(
                "--assay-transfer-min-score requires --retrieval-strategy assay_transfer_tool"
            )
        if args.assay_transfer_diversity_mode != ASSAY_TRANSFER_DIVERSITY_NONE:
            raise SystemExit(
                "--assay-transfer-diversity-mode requires --retrieval-strategy assay_transfer_tool"
            )
    if args.assay_transfer_min_score is not None and not 0.0 <= args.assay_transfer_min_score <= 1.0:
        raise SystemExit("--assay-transfer-min-score must be between 0 and 1 inclusive")
    try:
        validate_assay_transfer_diversity(
            mode=args.assay_transfer_diversity_mode,
            score_slack=args.assay_transfer_diversity_score_slack,
        )
        validate_assay_transfer_records_per_molecule(
            args.assay_transfer_records_per_molecule,
            selection_unit=args.assay_transfer_selection_unit,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        args.assay_transfer_records_per_molecule > 1
        and args.retrieval_strategy != ASSAY_TRANSFER_TOOL_STRATEGY
    ):
        raise SystemExit(
            "--assay-transfer-records-per-molecule greater than 1 requires "
            "--retrieval-strategy assay_transfer_tool"
        )
    reranker_provenance_name = (
        "assay_transfer" if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY else "none"
    )
    try:
        validate_scored_neighbors_configuration(
            enabled=args.enable_assay_transfer_scores,
            experiment_mode=args.experiment_mode,
            retrieval_source=args.retrieval_source,
            retrieval_reranker=reranker_provenance_name,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        args.group_output_schema == "assay-transfer"
        and args.group_prompt_format != "assay_transfer_tool"
    ):
        raise SystemExit(
            "--group-output-schema assay-transfer requires "
            "--group-prompt-format assay_transfer_tool"
        )
    if args.group_prompt_instructions_file and args.group_prompt_format not in TEXT_GROUP_PROMPT_FORMATS:
        raise SystemExit(
            "--group-prompt-instructions-file requires a text group prompt format"
        )
    if (
        args.group_prompt_instructions_file
        and args.group_prompt_version != DEFAULT_GROUP_PROMPT_VERSION
    ):
        raise SystemExit(
            "A named --group-prompt-version is immutable and cannot be combined with "
            "--group-prompt-instructions-file"
        )
    group_prompt_instruction_provenance: dict[str, Any] = {}
    if args.group_prompt_format in TEXT_GROUP_PROMPT_FORMATS:
        try:
            group_prompt_instruction_provenance = instruction_file_provenance(
                args.group_prompt_format,
                args.group_prompt_instructions_file or None,
                output_schema_profile=args.group_output_schema,
                prompt_version=args.group_prompt_version,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        observed_hash = str(group_prompt_instruction_provenance["sha256"])
        if (
            args.group_prompt_instructions_sha256
            and observed_hash != args.group_prompt_instructions_sha256
        ):
            raise SystemExit(
                "Group prompt instructions SHA-256 mismatch: "
                f"expected {args.group_prompt_instructions_sha256}, observed {observed_hash}"
            )
        args.group_prompt_instructions_file = str(
            group_prompt_instruction_provenance["path"]
        )
    _validate_prompt_profile_reuse(args)
    _load_env(Path(args.env_file), override=True)
    api_key = os.getenv(args.api_key_env)
    if not api_key and not args.prepare_only:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    if args.resume_final_from_run_dir:
        client = OpenAICompatibleClient(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            tool_service_url=args.tool_service_url,
            enable_group_tools=(
                not args.disable_group_tools
                and not args.disable_flat_tools
                and not args.analogous_reasoning_only
            ),
            max_tool_rounds=args.max_tool_rounds,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        return _resume_final_from_run_dir(
            Path(args.resume_final_from_run_dir),
            client,
            analogous_reasoning_only=args.analogous_reasoning_only,
        )

    run_id = args.run_id or time.strftime("bioavailability_reasoning_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out_dir = ensure_dir(Path(args.out_root) / run_id)
    _log(f"run_id={run_id}")

    query_record = _read_jsonl_record(Path(args.input_jsonl), args.query_index)
    query_smiles = str(query_record.get(args.smiles_field) or "")
    if not query_smiles:
        raise SystemExit(f"Input record has no `{args.smiles_field}` value.")

    reranker = None
    if args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        if args.retrieval_source != "starling":
            raise SystemExit("assay_transfer reranking requires --retrieval-source starling")
        if args.experiment_mode not in {"direct", "full_flat", "full_mechanism"}:
            raise SystemExit("assay_transfer reranking requires direct, full_flat, or full_mechanism mode")
        if args.rerank_cache_mode != "read_only":
            raise SystemExit("reasoning runs require --rerank-cache-mode read_only; populate scores with precompute")
        reranker = V9CachedAssayReranker(
            task_id="bioavailability_ma",
            catalog_path=args.rerank_catalog,
            cache_path=args.rerank_cache,
            cache_mode="read_only",
            model=args.assay_transfer_model,
            model_revision=args.assay_transfer_model_revision,
            candidate_manifest_path=args.rerank_candidate_manifest,
        )
    expected_reranker = reranker.provenance() if reranker is not None else {"name": "none"}
    retrieval = load_retrieval_replay(
        args.retrieval_replay_run_dir,
        query_smiles,
        expected_neighbor_selector=args.morgan_neighbor_selector,
        expected_reranker_provenance=expected_reranker,
        expected_assay_transfer_selection_policy=(
            {
                "min_score": args.assay_transfer_min_score,
                "diversity": assay_transfer_selection_policy(
                    mode=args.assay_transfer_diversity_mode,
                    score_slack=args.assay_transfer_diversity_score_slack,
                    selection_unit=args.assay_transfer_selection_unit,
                    records_per_molecule=args.assay_transfer_records_per_molecule,
                ),
            }
            if reranker is not None
            else None
        ),
    )
    if retrieval is not None and args.assay_transfer_min_score is not None:
        replay_policy = (retrieval.get("experiment") or {}).get(
            "assay_transfer_selection_policy"
        ) or {}
        if replay_policy.get("min_score") != args.assay_transfer_min_score:
            raise SystemExit(
                "Retrieval replay assay-transfer threshold mismatch: "
                f"expected {args.assay_transfer_min_score}, "
                f"observed {replay_policy.get('min_score')!r}"
            )
    index = None
    if retrieval is None and args.experiment_mode != "none":
        _log("loading neighbor index")
        index = load_index(Path(args.index))
    if retrieval is None:
        _log(f"building retrieval view mode={args.experiment_mode} source={args.retrieval_source}")
        retrieval = retrieve_experiment_view(
            query_smiles,
            index,
            mode=args.experiment_mode,
            config=(
                get_source_config(args.retrieval_source)
                if args.experiment_mode not in {"none", "native"}
                else None
            ),
            top_k_per_group=args.top_k_per_group,
            min_similarity=args.min_similarity,
            native_groups=args.groups,
            neighbor_identity_policy=args.neighbor_identity_policy,
            neighbor_selector=args.morgan_neighbor_selector,
            reranker=reranker,
            assay_transfer_initial_morgan_filter=args.assay_transfer_initial_morgan_filter,
            assay_transfer_min_score=args.assay_transfer_min_score,
            assay_transfer_diversity_mode=args.assay_transfer_diversity_mode,
            assay_transfer_diversity_score_slack=args.assay_transfer_diversity_score_slack,
            assay_transfer_selection_unit=args.assay_transfer_selection_unit,
            assay_transfer_records_per_molecule=(
                args.assay_transfer_records_per_molecule
            ),
        )
    else:
        _log(f"replaying frozen retrieval from {args.retrieval_replay_run_dir}")
    if retrieval.get("status") != "ok":
        raise SystemExit(json.dumps(retrieval.get("errors", []), ensure_ascii=False))
    if reranker is not None:
        reranker.cache.close()
    if reranker is not None:
        try:
            prepare_assay_transfer_selected_neighbors(
                retrieval, expose_scores=args.enable_assay_transfer_scores
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    attach_external_condition(retrieval, query_record)
    if args.enable_chembl_exact_context and index is not None:
        _log("enriching retrieval with exact ChEMBL context")
        retrieval = enrich_retrieval_with_chembl_context(
            retrieval,
            index,
            chembl_sqlite=args.chembl_sqlite,
        )
    retrieval_path = out_dir / "retrieval.json"
    _write_json(retrieval_path, retrieval)
    _log(f"wrote {retrieval_path}")

    groups = [group for group in retrieval["groups"] if group.get("neighbors")]
    if args.max_groups:
        groups = groups[: args.max_groups]
    _log(f"group reasoning calls={len(groups)}")
    if args.prepare_only:
        _log("prepare-only complete; reasoning stages deferred to the global pool")
        return 0

    client = OpenAICompatibleClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tool_service_url=args.tool_service_url,
        enable_group_tools=(
            not args.disable_group_tools
            and not args.disable_flat_tools
            and not args.analogous_reasoning_only
        ),
        max_tool_rounds=args.max_tool_rounds,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
    )
    reasoning_retrieval = prepare_reasoning_retrieval(
        retrieval,
        client.tool_service,
        identity_blind=args.identity_blind,
        harness_prefetch_tools=args.harness_prefetch_tools,
        prefetched_tool_replay_run_dir=args.prefetched_tool_replay_run_dir,
        neighbor_context_profile=args.neighbor_context_profile,
        include_query_tools=not args.analogous_reasoning_only,
        include_neighbor_tools=(
            not args.disable_flat_tools and not args.analogous_reasoning_only
        ),
    )
    analogous_flat = args.analogous_reasoning_only and args.experiment_mode == "full_flat"
    if analogous_flat:
        reasoning_retrieval = expose_neighbor_smiles_only(
            reasoning_retrieval, retrieval
        )
    reasoning_groups = [group for group in reasoning_retrieval["groups"] if group.get("neighbors")]
    if args.max_groups:
        reasoning_groups = reasoning_groups[: args.max_groups]
    if args.group_prompt_format == "assay_transfer_tool":
        missing = [
            (group["group_id"], neighbor.get("molecule_chembl_id"))
            for group in reasoning_groups
            for neighbor in group.get("neighbors") or []
            if not neighbor.get("transfer_winning_record")
        ]
        if missing:
            raise SystemExit(
                "assay_transfer_tool format requires an assay record on every selected "
                f"neighbor; missing for {missing[:5]} (re-run precompute/retrieval)."
            )
    frozen_single = (
        _omitted_single_output()
        if args.analogous_reasoning_only
        else load_frozen_single_analysis(args.single_analysis_source_run_dir)
    )
    frozen_groups = load_reusable_group_outputs(
        args.group_analysis_source_run_dir,
        retrieval,
        target_neighbor_context_profile=args.neighbor_context_profile,
    )

    group_prompt_options = {
        "prompt_min_similarity": (
            args.group_prompt_min_similarity
            if args.group_prompt_min_similarity is not None
            else args.min_similarity
        ),
        "instructions_file": args.group_prompt_instructions_file or None,
        "output_schema_profile": args.group_output_schema,
        "presentation_style": args.presentation_style,
        "prompt_version": args.group_prompt_version,
        "omit_query_tools": args.analogous_reasoning_only or args.disable_flat_tools,
    }
    if analogous_flat:
        single_output = _omitted_single_output()
        group_outputs = [
            reason_analogous_flat_group(
                client, group, task_id="bioavailability_ma"
            )
            for group in reasoning_groups
        ]
    else:
        single_output, group_outputs = _run_parallel_reasoning(
            client,
            reasoning_retrieval,
            reasoning_groups,
            max_workers=args.max_workers,
            single_output=frozen_single,
            group_outputs=frozen_groups,
            group_prompt_format=args.group_prompt_format,
            group_prompt_options=group_prompt_options,
            prompt_profile=args.bioavailability_prompt_profile,
            disable_flat_tools=args.disable_flat_tools,
        )
    single_path = out_dir / "single_molecule_reasoning_output.json"
    _write_json(single_path, single_output)
    _log(f"wrote {single_path}")

    raw_group_path = out_dir / "group_reasoning_outputs_raw.jsonl"
    if args.identity_blind:
        _write_jsonl(raw_group_path, group_outputs)
        group_outputs = sanitize_identity_blind_branch_outputs(group_outputs, retrieval)
        _log(f"wrote {raw_group_path}")
    group_path = out_dir / "group_reasoning_outputs.jsonl"
    _write_jsonl(group_path, group_outputs)
    _log(f"wrote {group_path}")

    final_output = (
        reason_analogous_flat_final(
            client, group_outputs, task_id="bioavailability_ma"
        )
        if analogous_flat
        else _run_final_reasoning(
            client,
            reasoning_retrieval,
            single_output,
            group_outputs,
            final_evidence_surface=args.final_evidence_surface,
            final_decision_profile=args.final_decision_profile,
            prompt_profile=args.bioavailability_prompt_profile,
            analogous_reasoning_only=args.analogous_reasoning_only,
        )
    )
    final_path = out_dir / "final_reasoning_output.json"
    _write_json(final_path, final_output)
    _log(f"wrote {final_path}")

    trace_path = out_dir / "trace_messages.jsonl"
    _write_trace_jsonl(
        trace_path,
        query_record=query_record,
        query_index=args.query_index,
        smiles="[identity_blind]" if args.identity_blind else query_smiles,
        single_output=single_output,
        group_outputs=group_outputs,
        final_output=final_output,
    )
    _log(f"wrote {trace_path}")

    manifest = {
        "run_id": run_id,
        "input_jsonl": args.input_jsonl,
        "query_index": args.query_index,
        "smiles_field": args.smiles_field,
        "experiment_mode": args.experiment_mode,
        "retrieval_source": args.retrieval_source,
        "retrieval_strategy": args.retrieval_strategy,
        "assay_transfer_profile": args.assay_transfer_profile,
        "retrieval_reranker": (retrieval.get("experiment") or {}).get("retrieval_reranker", {"name": "none"}),
        "enable_assay_transfer_scores": args.enable_assay_transfer_scores,
        "assay_transfer_template_profile": args.assay_transfer_template_profile,
        "assay_transfer_prompt_provenance": (
            (retrieval.get("experiment") or {}).get("retrieval_reranker", {})
            if reranker is not None
            else {}
        ),
        "llm_neighbor_score_policy": (retrieval.get("experiment") or {}).get("llm_neighbor_score_policy", {}),
        "assay_transfer_initial_morgan_filter": args.assay_transfer_initial_morgan_filter,
        "assay_transfer_diversity_mode": args.assay_transfer_diversity_mode,
        "assay_transfer_diversity_score_slack": args.assay_transfer_diversity_score_slack,
        "assay_transfer_selection_unit": args.assay_transfer_selection_unit,
        "assay_transfer_records_per_molecule": (
            args.assay_transfer_records_per_molecule
        ),
        "assay_transfer_selection_policy": (retrieval.get("experiment") or {}).get(
            "assay_transfer_selection_policy", {}
        ),
        "rerank_catalog": args.rerank_catalog if reranker is not None else "",
        "rerank_cache": args.rerank_cache if reranker is not None else "",
        "neighbor_identity_policy": args.neighbor_identity_policy,
        "morgan_neighbor_selector": args.morgan_neighbor_selector,
        "neighbor_selector": args.morgan_neighbor_selector,
        "neighbor_context_profile": args.neighbor_context_profile,
        "final_evidence_surface": args.final_evidence_surface,
        "final_decision_profile": args.final_decision_profile,
        "task_prompt_profile": args.bioavailability_prompt_profile,
        "label_scope": get_bioavailability_prompt_profile(
            args.bioavailability_prompt_profile
        ).label_scope,
        "retrieval_replay_source_run_dir": args.retrieval_replay_run_dir,
        "prefetched_tool_replay_source_run_dir": args.prefetched_tool_replay_run_dir,
        "identity_blind": args.identity_blind,
        "disable_flat_tools": args.disable_flat_tools,
        "analogous_reasoning_only": args.analogous_reasoning_only,
        "prompt_identity_view": (
            PROMPT_IDENTITY_VIEW if analogous_flat else "identity_blind"
            if args.identity_blind else "deployment_visible"
        ),
        "analogous_flat_prompt_provenance": (
            analogous_flat_prompt_provenance("bioavailability_ma")
            if analogous_flat else {}
        ),
        "single_branch_execution": (
            "omitted" if args.analogous_reasoning_only else "executed_or_reused"
        ),
        "single_branch_omission_reason": (
            "analogous_reasoning_only" if args.analogous_reasoning_only else ""
        ),
        "query_tool_execution": (
            "omitted" if args.analogous_reasoning_only else "enabled"
        ),
        "group_query_tool_instruction_policy": (
            "omitted.v1" if args.analogous_reasoning_only else "standard.v1"
        ),
        "final_prompt_provenance": (
            {}
            if analogous_flat
            else final_prompt_provenance(
                analogous_reasoning_only=args.analogous_reasoning_only
            )
        ),
        "harness_prefetch_tools": (
            False
            if args.analogous_reasoning_only
            else args.identity_blind or args.harness_prefetch_tools
        ),
        "neighbor_index": args.index if args.experiment_mode != "none" else "",
        "retrieval_evidence_source": retrieval.get("evidence_source", {}),
        "model": args.model,
        "base_url": args.base_url,
        "tool_service_url": args.tool_service_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "thinking": {"type": "enabled"} if args.enable_thinking else {"type": "disabled"},
        "group_tools_enabled": (
            not args.disable_group_tools
            and not args.disable_flat_tools
            and not args.analogous_reasoning_only
        ),
        "tool_execution_mode": (
            "omitted"
            if args.analogous_reasoning_only
            else "harness_prefetch_query_only"
            if args.disable_flat_tools
            else "harness_prefetch"
            if args.identity_blind or args.harness_prefetch_tools
            else "llm_function_call"
        ),
        "single_analysis_source_run_dir": args.single_analysis_source_run_dir,
        "group_analysis_source_run_dir": args.group_analysis_source_run_dir,
        "chembl_exact_context_enabled": args.enable_chembl_exact_context,
        "chembl_sqlite": args.chembl_sqlite,
        "group_tool_names": [tool["function"]["name"] for tool in GROUP_REASONING_TOOLS]
        if not args.disable_group_tools
        and not args.disable_flat_tools
        and not args.analogous_reasoning_only
        else [],
        "max_tool_rounds": args.max_tool_rounds,
        "top_k_per_group_requested": args.top_k_per_group,
        "top_k_per_group": args.top_k_per_group,
        "min_similarity": args.min_similarity,
        "assay_transfer_min_score": args.assay_transfer_min_score,
        "group_prompt_format": args.group_prompt_format,
        "group_prompt_version": args.group_prompt_version,
        "group_prompt_provenance": (
            group_prompt_provenance(
                args.group_prompt_format,
                prompt_version=args.group_prompt_version,
                instructions_file=args.group_prompt_instructions_file or None,
                output_schema_profile=args.group_output_schema,
            )
            if args.group_prompt_format in TEXT_GROUP_PROMPT_FORMATS
            else {}
        ),
        "group_output_schema": args.group_output_schema,
        "group_output_schema_provenance": group_output_schema_provenance(
            args.group_output_schema
        ),
        "group_prompt_instructions_file": group_prompt_instruction_provenance.get(
            "path", ""
        ),
        "group_prompt_instructions_sha256": group_prompt_instruction_provenance.get(
            "sha256", ""
        ),
        "group_prompt_instruction_count": group_prompt_instruction_provenance.get(
            "instruction_count", 0
        ),
        "group_evidence_presentation": (
            "minimal_evidence.v1"
            if args.group_prompt_format in {"morganfingerprint", "assay_transfer_tool"}
            else "legacy"
        ),
        "group_prompt_min_similarity": group_prompt_options["prompt_min_similarity"],
        "presentation_style": args.presentation_style,
        "n_groups_with_neighbors": len(groups),
        "paths": {
            "retrieval": str(retrieval_path),
            "single_molecule_reasoning_output": str(single_path),
            "group_reasoning_outputs": str(group_path),
            "group_reasoning_outputs_raw": str(raw_group_path) if args.identity_blind else "",
            "final_reasoning_output": str(final_path),
            "trace_messages": str(trace_path),
        },
        "query_label_for_eval_only": query_record.get("Y"),
    }
    _write_json(out_dir / "manifest.json", manifest)
    _print_summary(final_output, manifest)
    return 0


def _run_parallel_reasoning(
    client: OpenAICompatibleClient,
    retrieval: dict[str, Any],
    groups: list[dict[str, Any]],
    *,
    max_workers: int,
    single_output: dict[str, Any] | None = None,
    group_outputs: list[dict[str, Any]] | None = None,
    group_prompt_format: str = "legacy",
    group_prompt_options: dict[str, Any] | None = None,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    disable_flat_tools: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs = list(group_outputs or [])
    reused_group_ids = {str(output.get("group_id") or "") for output in outputs}
    include_assay_transfer_score = scored_neighbors_prompt_enabled(retrieval)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _reason_one_group,
                client,
                _llm_evidence_query_payload(
                    query_without_prefetched_tools(retrieval["query"])
                    if disable_flat_tools
                    else retrieval["query"]
                ),
                group,
                include_assay_transfer_score=include_assay_transfer_score,
                prompt_format=group_prompt_format,
                prompt_options=group_prompt_options,
                prompt_profile=prompt_profile,
            ): group["group_id"]
            for group in groups
            if str(group.get("group_id") or "") not in reused_group_ids
        }
        if single_output is None:
            futures[
                executor.submit(
                    _reason_single_molecule,
                    client,
                    _llm_query_payload(retrieval["query"]),
                    _clean_query_chembl_context(retrieval.get("query_chembl_context") or {}),
                    prompt_profile=prompt_profile,
                )
            ] = "single_molecule"
        else:
            _log("reusing frozen single molecule analysis")
        for future in concurrent.futures.as_completed(futures):
            item_id = futures[future]
            try:
                result = future.result()
                if item_id == "single_molecule":
                    single_output = result
                    _log("single molecule analysis done")
                else:
                    outputs.append(result)
                    _log(f"group done: {item_id}")
            except Exception as exc:
                if item_id == "single_molecule":
                    single_output = {"analysis_id": "single_molecule", "status": "error", "error": str(exc)}
                    _log(f"single molecule analysis error: {exc}")
                else:
                    outputs.append({"group_id": item_id, "status": "error", "error": str(exc)})
                    _log(f"group error: {item_id}: {exc}")
    return single_output or {"analysis_id": "single_molecule", "status": "error", "error": "missing result"}, sorted(
        outputs,
        key=lambda item: str(item.get("group_id", "")),
    )


def _omitted_single_output() -> dict[str, Any]:
    return {
        "analysis_id": "single_molecule",
        "status": "omitted",
        "reason": "analogous_reasoning_only",
    }


def single_molecule_messages(
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
    *,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
) -> list[dict[str, str]]:
    """Compile the single-molecule stage [system, user] messages (no LLM needed)."""
    profile = get_bioavailability_prompt_profile(prompt_profile)
    instructions = [
        "Call molecule_properties for the query molecule before analysis.",
        *profile.single_instructions,
    ]
    if query.get("prefetched_molecule_properties"):
        instructions[0] = "Use the harness-prefetched molecule_properties result."
        if query.get("identity_hidden"):
            instructions[0] += " Do not identify or name the query."
    payload: dict[str, Any] = {
        "task": profile.single_task,
        "query": query,
        "instructions": instructions,
        "required_json_schema": SINGLE_SCHEMA,
    }
    if chembl_context:
        instructions.append(
            "If exact_query_chembl_context is found, distinguish direct same-molecule ChEMBL bioavailability evidence from the physicochemical prior."
        )
        payload["exact_query_chembl_context"] = chembl_context
    return [
        {
            "role": "system",
            "content": (
                profile.single_system_role
                + (
                    "The harness already supplied molecule_properties; do not call tools. "
                    + ("Do not infer query identity. " if query.get("identity_hidden") else "")
                    if query.get("prefetched_molecule_properties")
                    else "You may call exactly one tool: molecule_properties. "
                )
                + "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _reason_single_molecule(
    client: OpenAICompatibleClient,
    query: dict[str, Any],
    chembl_context: dict[str, Any] | None = None,
    *,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
) -> dict[str, Any]:
    messages = single_molecule_messages(
        query,
        chembl_context,
        prompt_profile=prompt_profile,
    )
    response = call_single_molecule_branch(
        client,
        messages,
        query=query,
        tools=SINGLE_MOLECULE_TOOLS,
        first_tool_choice=SINGLE_MOLECULE_TOOL_CHOICE,
    )
    return {
        "analysis_id": "single_molecule",
        "status": "ok" if structured_response_is_valid(response) else "error",
        "llm": response,
    }


def legacy_group_messages(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    include_assay_transfer_score: bool = False,
    group_tools_enabled: bool = True,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    omit_query_tools: bool = False,
) -> list[dict[str, str]]:
    """Compile the legacy JSON group-branch [system, user] messages (no LLM needed)."""
    profile = get_bioavailability_prompt_profile(prompt_profile)
    return [
        {
            "role": "system",
            "content": group_system_message(
                group,
                group_tools_enabled=group_tools_enabled,
                system_role=profile.group_system_role,
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _group_prompt_payload(
                    query,
                    group,
                    include_assay_transfer_score=include_assay_transfer_score,
                    prompt_profile=prompt_profile,
                    include_query_tool_guidance=not omit_query_tools,
                ),
                ensure_ascii=False,
            ),
        },
    ]


def _reason_one_group(
    client: OpenAICompatibleClient,
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    include_assay_transfer_score: bool = False,
    prompt_format: str = "legacy",
    prompt_options: dict[str, Any] | None = None,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
) -> dict[str, Any]:
    profile = get_bioavailability_prompt_profile(prompt_profile)
    if prompt_format == "legacy":
        messages = legacy_group_messages(
            query,
            group,
            include_assay_transfer_score=include_assay_transfer_score,
            group_tools_enabled=client.enable_group_tools,
            prompt_profile=prompt_profile,
            omit_query_tools=bool((prompt_options or {}).get("omit_query_tools")),
        )
    else:
        effective_prompt_options = {
            **(prompt_options or {}),
            "group_tools_enabled": client.enable_group_tools,
            "additional_instructions": list(profile.group_instructions),
            "system_role": profile.group_system_role,
        }
        messages = build_group_messages(
            query,
            group,
            prompt_format=prompt_format,
            options=effective_prompt_options,
        )
    response = call_group_branch(
        client,
        messages,
        group=group,
        tools=GROUP_REASONING_TOOLS,
        **group_output_validation(
            str((prompt_options or {}).get("output_schema_profile", "legacy"))
        ),
    )
    return {
        "group_id": group["group_id"],
        "status": "ok" if structured_response_is_valid(response) else "error",
        "tier": group["tier"],
        "endpoint_group": group["endpoint_group"],
        "n_neighbors": len(group["neighbors"]),
        "llm": response,
    }


def final_messages(
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    *,
    analogous_reasoning_only: bool = False,
) -> list[dict[str, str]]:
    """Compile the final-synthesis stage [system, user] messages (no LLM needed)."""
    return build_final_messages(
        retrieval,
        single_output,
        group_outputs,
        high_f_cutoff=BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT,
        analogous_reasoning_only=analogous_reasoning_only,
    )


def _run_final_reasoning(
    client: OpenAICompatibleClient,
    retrieval: dict[str, Any],
    single_output: dict[str, Any],
    group_outputs: list[dict[str, Any]],
    *,
    final_evidence_surface: str = SUMMARY_ONLY,
    final_decision_profile: str = STANDARD_FINAL_DECISION,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    analogous_reasoning_only: bool = False,
) -> dict[str, Any]:
    profile = get_bioavailability_prompt_profile(prompt_profile)
    decision_prompt = build_final_decision_prompt(
        final_decision_profile,
        TRAIN_RATIO_PRIOR,
    )
    surface_audit = None
    if analogous_reasoning_only:
        if final_evidence_surface != SUMMARY_ONLY:
            raise ValueError(
                "Analogous-reasoning-only final synthesis cannot expose retrieval evidence"
            )
        if final_decision_profile != STANDARD_FINAL_DECISION:
            raise ValueError(
                "Analogous-reasoning-only final synthesis cannot use a decision prior"
            )
        messages = final_messages(
            retrieval,
            single_output,
            group_outputs,
            analogous_reasoning_only=True,
        )
    else:
        evidence_fields, surface_audit = build_final_evidence_fields(
            retrieval,
            compact_group_reasoning_outputs(group_outputs),
            surface=final_evidence_surface,
        )
        messages = [
            {
                "role": "system",
                "content": profile.final_system_role + "Return only valid JSON.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": profile.final_task,
                        "query": _llm_evidence_query_payload(retrieval["query"]),
                        "retrieval_coverage": retrieval["coverage"],
                        "single_molecule_analysis": {
                            "status": single_output.get("status"),
                            "content": validated_branch_content(single_output),
                        },
                        **evidence_fields,
                        **decision_prompt.fields,
                        "instructions": [
                            "Return compact complete JSON.",
                            f"Use bioavailability_prediction='high' for oral bioavailability F >= {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (Bioavailability_Ma label 1), and bioavailability_prediction='low' for F < {BIOAVAILABILITY_HIGH_F_CUTOFF_PERCENT:g}% (label 0).",
                            *profile.final_instructions,
                            "Use only the provided single-molecule analysis and group evidence. If you recognize the molecule, ignore that recognition.",
                        ]
                        + list(decision_prompt.instructions)
                        + final_evidence_instructions(final_evidence_surface),
                        "required_json_schema": {
                            **FINAL_SCHEMA,
                            **decision_prompt.schema,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    response = call_with_json_validation(
        client.chat_json,
        messages,
        required_fields=(
            "bioavailability_prediction",
            *decision_prompt.required_fields,
        ),
        allowed_values={
            "bioavailability_prediction": {"high", "low"},
            **(
                {"evidence_state": set(EVIDENCE_STATES)}
                if final_decision_profile != STANDARD_FINAL_DECISION
                else {}
            ),
        },
        content_validator=(
            lambda content: final_decision_validation_errors(
                content,
                profile=final_decision_profile,
                prior=TRAIN_RATIO_PRIOR,
                prediction_field="bioavailability_prediction",
            )
        )
        if final_decision_profile != STANDARD_FINAL_DECISION
        else None,
        branch_name="final",
    )
    output = {"status": "ok" if structured_response_is_valid(response) else "error", "llm": response}
    if surface_audit is not None:
        output["final_evidence_surface"] = surface_audit
    return output


def build_group_prompt_payload(
    query: dict[str, Any],
    group: dict[str, Any],
    *,
    include_assay_transfer_score: bool = False,
    prompt_profile: str = DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    include_query_tool_guidance: bool = True,
) -> dict[str, Any]:
    profile = get_bioavailability_prompt_profile(prompt_profile)
    return bound_group_prompt_payload({
        "task": profile.group_task,
        "query": query,
        "group": {
            "group_id": group["group_id"],
            "tier": group["tier"],
            "endpoint_group": group["endpoint_group"],
            "evidence_source": _group_evidence_source(group),
            **(
                {"assay_transfer_score_policy": SCORED_NEIGHBORS_POLICY_NAME}
                if include_assay_transfer_score
                else {}
            ),
        },
        "neighbors": [
            {
                "rank": neighbor["rank"],
                "molecule_chembl_id": neighbor["molecule_chembl_id"],
                "canonical_smiles": neighbor["canonical_smiles"],
                "similarity": neighbor["similarity"],
                "similarity_bucket": neighbor["similarity_bucket"],
                **(
                    {"assay_transfer_score": public_assay_transfer_score(neighbor)}
                    if include_assay_transfer_score
                    else {}
                ),
                **(
                    {
                        "prefetched_comparisons": (
                            neighbor.get("prefetched_comparisons") or []
                        )
                    }
                    if include_query_tool_guidance
                    else {}
                ),
                "evidence_rows": [
                    evidence_for_group_llm(row, group)
                    for row in neighbor["evidence_rows"]
                ],
                "shared_assay_context": _clean_shared_assay_context(neighbor.get("shared_assay_context") or {}),
            }
            for neighbor in group["neighbors"]
        ],
        "instructions": [
            "Use only this group's evidence.",
            "Each evidence_rows item follows minimal_evidence.v1; read endpoint/measurement, text, annotations, quality, provenance, and examples without assuming a source-specific schema.",
            *(
                [
                    "assay_transfer_score is a 0-1 cached model estimate that the exact selected source assay record transfers to the query under the copied assay context. It is uncalibrated, may be extrapolative for qualitative records, and is not an oral-bioavailability probability, assay value, label vote, or deterministic override."
                ]
                if include_assay_transfer_score
                else []
            ),
            *profile.group_instructions,
            *(
                [
                    "Use mmp_structure_compare to inspect scaffold/MCS/matched-pair differences when similarity bucket alone is not enough.",
                    "Use properties_compare when property differences such as pKa, logD, TPSA, charge, HBD/HBA, logP, molecular size, or polarity could affect oral bioavailability transferability.",
                    "Tool outputs are authoritative only for the pair they compare; cite which neighbor each tool result supports.",
                ]
                if include_query_tool_guidance
                else [
                    "No query property or query-to-neighbor comparison tools are available; reason only from the supplied analog retrieval and assay evidence."
                ]
            ),
            "Use same_endpoint_activity as direct query-vs-neighbor assay comparison when present.",
            "Use same_assay_different_endpoint_activity only as same-assay context; do not directly compare numeric values across different endpoints.",
            "Distinguish direct oral bioavailability, in vivo oral exposure/absorption, in vitro permeability, solubility/dissolution, metabolism/clearance, formulation/food-effect context, and weak inhibition/binding evidence.",
            "Do not convert CYP IC50/inhibition into metabolic instability, and do not convert transporter IC50/inhibition directly into substrate/transport unless assay context supports it.",
            "Return key_evidence as structured evidence cards, not a plain list of molecule ids.",
            "For aggregated evidence, examples preserve endpoint, value, condition, and support-text pairings. Do not treat qualitative, relative, or surrogate_proxy evidence as a direct absolute F% measurement.",
            *(
                [
                    "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, derive tool_summary from tool outputs, and judge transferability/effect_on_bioavailability_reasoning yourself."
                ]
                if include_query_tool_guidance
                else [
                    "For each key_evidence item, derive assay_signal and activity_values from the provided evidence_rows, leave tool_summary empty, and judge transferability/effect_on_bioavailability_reasoning from the supplied analog evidence."
                ]
            ),
            "Return JSON with useful_for_bioavailability_reasoning, transferability, evidence_direction, confidence, reasoning_summary, key_evidence, caveats.",
        ],
        "required_json_schema": GROUP_SCHEMA,
    }, evidence_prompt_profile=str(group.get("evidence_prompt_profile") or ""))


# Historical internal callers keep working while external materializers use
# the explicit public adapter above.
_group_prompt_payload = build_group_prompt_payload


def _group_evidence_source(group: dict[str, Any]) -> str:
    sources: list[str] = []
    for neighbor in group.get("neighbors", []):
        for row in neighbor.get("evidence_rows", []):
            evidence = row.get("minimal_evidence") or row
            source = row.get("evidence_source") or (evidence.get("source") or {}).get("name")
            source = str(source or "").strip()
            if source and source not in sources:
                sources.append(source)
    if len(sources) > 1:
        return " + ".join(sources)
    if sources:
        return sources[0]
    return "ChEMBL"


def _clean_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compatibility adapter for legacy callers of the shared evidence view."""
    return evidence_for_llm(row)


def _clean_query_chembl_context(context: dict[str, Any]) -> dict[str, Any]:
    status = context.get("status") or "not_available"
    if status != "found":
        return {key: context.get(key, "") for key in ["status", "reason", "standard_inchi_key"] if context.get(key)}
    return {
        "status": "found",
        "selected_molecule_chembl_id": context.get("selected_molecule_chembl_id", ""),
        "exact_matches": [_clean_exact_match(match) for match in context.get("exact_matches", [])],
        "bioavailability_relevant_evidence_rows": [
            evidence_for_llm(row)
            for row in context.get("bioavailability_relevant_evidence_rows", [])
        ],
    }


def _parse_json_content(content: str) -> Any:
    return parse_json_content(content)


def _validate_prompt_profile_reuse(args: argparse.Namespace) -> None:
    """Reject single/group branch reuse across Bio prompt contracts."""
    require_matching_prompt_profiles(
        target_profile=str(args.bioavailability_prompt_profile),
        source_dirs=(
            args.single_analysis_source_run_dir,
            args.group_analysis_source_run_dir,
        ),
        historical_profile=HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
    )


def _manifest_prompt_profile(manifest: dict[str, Any]) -> str:
    return prompt_profile_from_manifest(
        manifest,
        historical_profile=HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
    )


def _resume_final_from_run_dir(
    run_dir: Path,
    client: OpenAICompatibleClient,
    *,
    analogous_reasoning_only: bool,
) -> int:
    retrieval = json.loads((run_dir / "retrieval.json").read_text(encoding="utf-8"))
    single_output = json.loads((run_dir / "single_molecule_reasoning_output.json").read_text(encoding="utf-8"))
    group_outputs = [
        json.loads(line)
        for line in (run_dir / "group_reasoning_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if bool(manifest.get("analogous_reasoning_only")) != analogous_reasoning_only:
        raise SystemExit(
            "--resume-final-from-run-dir must use the same analogous-reasoning-only mode "
            "as the source run"
        )
    prompt_profile = _manifest_prompt_profile(manifest)
    retrieval, group_outputs, final_surface = prepare_resumed_final_inputs(
        retrieval,
        single_output,
        group_outputs,
        manifest,
        tool_service=client.tool_service,
    )
    final_output = _run_final_reasoning(
        client,
        retrieval,
        single_output,
        group_outputs,
        final_evidence_surface=final_surface,
        final_decision_profile=str(
            manifest.get("final_decision_profile") or STANDARD_FINAL_DECISION
        ),
        prompt_profile=prompt_profile,
        analogous_reasoning_only=analogous_reasoning_only,
    )
    final_path = run_dir / "final_reasoning_output.json"
    _write_json(final_path, final_output)
    trace_path = run_dir / "trace_messages.jsonl"
    query_record = {"Y": manifest.get("query_label_for_eval_only")}
    query_index = int(manifest.get("query_index") or 0)
    smiles = (
        "[identity_blind]"
        if manifest.get("identity_blind")
        else str((retrieval.get("query") or {}).get("input_smiles") or "")
    )
    _write_trace_jsonl(
        trace_path,
        query_record=query_record,
        query_index=query_index,
        smiles=smiles,
        single_output=single_output,
        group_outputs=group_outputs,
        final_output=final_output,
    )
    manifest["final_rerun_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["task_prompt_profile"] = prompt_profile
    manifest["label_scope"] = get_bioavailability_prompt_profile(prompt_profile).label_scope
    manifest.setdefault("paths", {})["final_reasoning_output"] = str(final_path)
    manifest.setdefault("paths", {})["trace_messages"] = str(trace_path)
    _write_json(manifest_path, manifest)
    _print_summary(final_output, manifest)
    return 0


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _print_summary(final_output: dict[str, Any], manifest: dict[str, Any]) -> None:
    content = (final_output.get("llm") or {}).get("content") or {}
    print(json.dumps({"manifest": manifest, "final": content}, ensure_ascii=False, indent=2), flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--smiles-field", default="drug")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--experiment-mode", choices=sorted(EXPERIMENT_MODES), default="native")
    parser.add_argument("--retrieval-source", default="chembl")
    parser.add_argument(
        "--neighbor-identity-policy",
        choices=NEIGHBOR_IDENTITY_POLICIES,
        default="operational",
    )
    parser.add_argument("--identity-blind", action="store_true")
    parser.add_argument(
        "--analogous-reasoning-only",
        "--analogous_reasoning_only",
        dest="analogous_reasoning_only",
        action="store_true",
        help=(
            "full_mechanism only: omit the single-molecule branch and all "
            "model-facing query tools, then synthesize only analog branch analyses."
        ),
    )
    parser.add_argument("--harness-prefetch-tools", action="store_true")
    parser.add_argument("--single-analysis-source-run-dir", default="")
    parser.add_argument("--group-analysis-source-run-dir", default="")
    parser.add_argument("--retrieval-replay-run-dir", default="")
    parser.add_argument("--prefetched-tool-replay-run-dir", default="")
    parser.add_argument("--chembl-sqlite", default=DEFAULT_CHEMBL_SQLITE)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume-final-from-run-dir", default="")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write retrieval.json and stop before any LLM request.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--tool-service-url", default=DEFAULT_TOOL_SERVICE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-groups", type=int, default=0, help="Debug limit; 0 means all groups with neighbors.")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tool-rounds", type=int, default=10)
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        help="OpenAI-compatible reasoning_effort value. Use an empty string to omit this parameter.",
    )
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--disable-group-tools", action="store_true")
    parser.add_argument(
        "--disable-flat-tools",
        action="store_true",
        help=(
            "full_flat only: omit neighbor comparison tools from the flat group branch; "
            "the single-molecule branch still receives molecule_properties"
        ),
    )
    parser.add_argument("--enable-chembl-exact-context", action="store_true")
    parser.add_argument("--top-k-per-group", type=int, default=3)
    parser.add_argument(
        "--enable-assay-transfer-scores",
        action="store_true",
        help=(
            "For Starling full_mechanism assay-transfer retrieval, expose each selected neighbor's rounded "
            "transfer score to the group reasoning LLM without changing --top-k-per-group."
        ),
    )
    parser.add_argument("--min-similarity", type=float, default=0.3)
    add_retrieval_strategy_args(parser)
    parser.add_argument(
        "--assay-transfer-min-score",
        type=float,
        default=None,
        help=(
            "Optional inclusive cached transfer-probability floor applied before the final top-k; "
            "requires --retrieval-strategy assay_transfer_tool."
        ),
    )
    parser.add_argument(
        "--assay-transfer-diversity-mode",
        choices=ASSAY_TRANSFER_DIVERSITY_MODES,
        default=ASSAY_TRANSFER_DIVERSITY_NONE,
        help=(
            "Opt-in score-slack diversity policy for assay-transfer records: structural "
            "covers query Morgan bits; assay covers canonical endpoint keys."
        ),
    )
    parser.add_argument(
        "--assay-transfer-diversity-score-slack",
        type=float,
        default=0.0,
        help=(
            "Maximum raw transfer-score loss allowed versus the best remaining record at each "
            "selection slot; zero preserves score ranking exactly."
        ),
    )
    parser.add_argument(
        "--assay-transfer-profile",
        choices=["v9_direct_gold"],
        default="v9_direct_gold",
    )
    parser.add_argument(
        "--assay-transfer-selection-unit",
        choices=ASSAY_TRANSFER_SELECTION_UNITS,
        default=ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    )
    parser.add_argument(
        "--assay-transfer-records-per-molecule",
        type=int,
        choices=range(
            ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
            ASSAY_TRANSFER_RECORDS_PER_MOLECULE_MAX + 1,
        ),
        default=ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
        metavar="N",
    )
    parser.add_argument("--rerank-catalog", default=V9_DEFAULT_PATHS["catalog"])
    parser.add_argument("--rerank-cache", default=V9_DEFAULT_PATHS["cache"])
    parser.add_argument(
        "--rerank-candidate-manifest",
        default=V9_DEFAULT_PATHS["candidate_manifest"],
        help="Frozen Morgan top-train candidate manifest used to build the V9 cache.",
    )
    parser.add_argument("--rerank-cache-mode", choices=["read_only"], default="read_only")
    parser.add_argument("--assay-transfer-model", default=V9_MODEL_PROFILE["model"])
    parser.add_argument(
        "--assay-transfer-model-revision", default=V9_MODEL_PROFILE["revision"]
    )
    parser.add_argument(
        "--assay-transfer-template-profile",
        choices=[V9_TEMPLATE_PROFILE],
        default=V9_TEMPLATE_PROFILE,
    )
    parser.add_argument(
        "--neighbor-context-profile",
        choices=NEIGHBOR_CONTEXT_PROFILES,
        default=STANDARD_NEIGHBOR_CONTEXT,
    )
    add_final_evidence_surface_argument(parser)
    add_final_decision_profile_argument(parser)
    parser.add_argument(
        "--bioavailability-prompt-profile",
        choices=BIOAVAILABILITY_PROMPT_PROFILES,
        default=DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    )
    parser.add_argument("--groups", nargs="*", default=None, help="Optional exact Tier.endpoint_group ids to reason over.")
    parser.add_argument(
        "--group-prompt-format",
        choices=list(GROUP_PROMPT_FORMATS),
        default="legacy",
        help=(
            "Group sub-branch prompt format: legacy (JSON, unchanged), morganfingerprint "
            "(text: molecules + similarity + records), or assay_transfer_tool "
            "(text: top-k assay records + transfer scores; the same molecule may repeat; "
            "requires the assay_transfer reranker)."
        ),
    )
    parser.add_argument(
        "--group-output-schema",
        choices=list(GROUP_OUTPUT_SCHEMA_PROFILES),
        default="legacy",
        help=(
            "Structured output profile for group branches. The assay-transfer "
            "profile is evidence-centric and requires --group-prompt-format "
            "assay_transfer_tool."
        ),
    )
    parser.add_argument(
        "--group-prompt-version",
        choices=GROUP_PROMPT_VERSIONS,
        default=DEFAULT_GROUP_PROMPT_VERSION,
    )
    parser.add_argument(
        "--group-prompt-instructions-file",
        default="",
        help=(
            "Optional UTF-8 instruction file for a text group prompt format. "
            "The default remains prompt_instructions/<group-prompt-format>.txt."
        ),
    )
    parser.add_argument(
        "--group-prompt-instructions-sha256",
        default="",
        help=(
            "Optional launch-time SHA-256 guard for --group-prompt-instructions-file; "
            "used by the batch runner to prevent mixed prompts."
        ),
    )
    parser.add_argument(
        "--group-prompt-min-similarity",
        type=float,
        default=None,
        help="morganfingerprint only: drop neighbors below this similarity from the prompt (default: --min-similarity).",
    )
    parser.add_argument(
        "--presentation-style",
        choices=["legacy", "full"],
        default="legacy",
        help=(
            "Record presentation style for the text group prompts: legacy (unified minimal "
            "view, unchanged) or full (per-source expanded scientific fields, keyed by the "
            "neighbor evidence_source; invariant across retriever)."
        ),
    )
    args = parser.parse_args(argv)
    if args.disable_flat_tools and args.experiment_mode != "full_flat":
        parser.error("--disable-flat-tools requires --experiment-mode full_flat")
    if args.assay_transfer_profile == "v9_direct_gold":
        if (
            args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
            and args.index == DEFAULT_INDEX
        ):
            args.index = V9_DEFAULT_INDEX
        if args.assay_transfer_initial_morgan_filter == 100:
            args.assay_transfer_initial_morgan_filter = 50
        if args.min_similarity == 0.3:
            args.min_similarity = 0.0
    return args


def _validate_analogous_reasoning_only(args: argparse.Namespace) -> None:
    if not args.analogous_reasoning_only:
        return
    if args.experiment_mode not in {"full_flat", "full_mechanism"}:
        raise SystemExit(
            "--analogous-reasoning-only requires --experiment-mode "
            "full_flat or full_mechanism"
        )
    if args.experiment_mode == "full_flat":
        if not args.identity_blind:
            raise SystemExit(
                "flat --analogous-reasoning-only requires --identity-blind"
            )
        if args.retrieval_strategy not in {
            ASSAY_TRANSFER_TOOL_STRATEGY,
            "morgan_fingerprint",
        }:
            raise SystemExit(
                "flat --analogous-reasoning-only requires assay_transfer_tool or "
                "morgan_fingerprint retrieval"
            )
        if (
            args.retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY
            and not args.enable_assay_transfer_scores
        ):
            raise SystemExit(
                "flat assay-transfer --analogous-reasoning-only requires "
                "--enable-assay-transfer-scores"
            )
    incompatible = {
        "--single-analysis-source-run-dir": args.single_analysis_source_run_dir,
        "--group-analysis-source-run-dir": args.group_analysis_source_run_dir,
        "--prefetched-tool-replay-run-dir": args.prefetched_tool_replay_run_dir,
    }
    used = [name for name, value in incompatible.items() if value]
    if used:
        raise SystemExit(
            "--analogous-reasoning-only requires fresh reasoning and cannot use: "
            + ", ".join(used)
        )
    if args.neighbor_context_profile == "coverage_mmp_ledger":
        raise SystemExit(
            "--analogous-reasoning-only cannot use --neighbor-context-profile "
            "coverage_mmp_ledger because it invokes query comparison tools"
        )
    if args.enable_chembl_exact_context:
        raise SystemExit(
            "--analogous-reasoning-only cannot use --enable-chembl-exact-context"
        )
    if args.final_evidence_surface != SUMMARY_ONLY:
        raise SystemExit(
            "--analogous-reasoning-only requires --final-evidence-surface summary_only"
        )
    if args.final_decision_profile != STANDARD_FINAL_DECISION:
        raise SystemExit(
            "--analogous-reasoning-only requires --final-decision-profile standard"
        )


def _log(message: str) -> None:
    print(f"[bioavailability_reasoning_pipeline] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
