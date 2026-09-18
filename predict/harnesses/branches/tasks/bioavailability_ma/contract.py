"""Define Oral branch retrieval/context policy and shared-runner configuration."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from predict.harnesses.branches.runner import BatchConfig, main as run_batch
from predict.harnesses.branches.exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    QUERY_ACTIVITY_FIELDS,
    attach_shared_assay_context,
    exact_evidence_rows,
    load_query_activities_for_assays,
    lookup_exact_molecule,
    enrich_retrieval_with_chembl_context as enrich_common_retrieval_with_chembl_context,
)
from predict.harnesses.branches.retrieval import BranchDefinition, BranchRetrievalConfig
from predict.retrieval.assay_reranking.v9 import (
    default_cache_paths,
    model_profile,
    preflight_cache_coverage,
)
from predict.harnesses.branches.tasks.bioavailability_ma.group_prompt import (
    DEFAULT_GROUP_PROMPT_VERSION,
    GROUP_PROMPT_VERSIONS,
    final_prompt_provenance,
    group_prompt_provenance,
)


from predict.tasks.bioavailability_ma.prompts import (
    BIOAVAILABILITY_PROMPT_PROFILES,
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
)


DIRECT_ORAL_BIOAVAILABILITY_GROUP = "Observed.direct_oral_bioavailability"
NONDIRECT_ORAL_BIOAVAILABILITY_GROUP = "Observed.nondirect_oral_bioavailability"

CHEMBL = BranchRetrievalConfig(
    source_name="chembl",
    direct_groups=(
        BranchDefinition(
            DIRECT_ORAL_BIOAVAILABILITY_GROUP, "Observed", "direct_oral_bioavailability",
            source_groups=("Tier 1.direct_absolute_bioavailability",),
        ),
    ),
    mechanism_groups=(
        BranchDefinition(DIRECT_ORAL_BIOAVAILABILITY_GROUP, "Observed", "direct_oral_bioavailability", source_groups=("Tier 1.direct_absolute_bioavailability",)),
        BranchDefinition("Observed.oral_auc_cmax_exposure", "Observed", "oral_auc_cmax_exposure", source_groups=("Tier 2.oral_auc_exposure", "Tier 2.oral_cmax_exposure", "Tier 6.food_effect_or_fed_fasted", "Tier 6.relative_bioavailability_or_formulation")),
        BranchDefinition("Fa.absorption_solubility_permeability", "Fa", "absorption_solubility_permeability", source_groups=("Tier 2.absorption_fraction_or_hia", "Tier 2.in_vivo_intestinal_permeability", "Tier 3.cell_permeability_papp", "Tier 3.pampa_or_artificial_membrane", "Tier 4.dissolution", "Tier 4.gi_or_chemical_stability", "Tier 4.solubility")),
        BranchDefinition("Fg.gut_wall_efflux_intestinal_metabolism", "Fg", "gut_wall_efflux_intestinal_metabolism", source_groups=("Tier 3.cell_bidirectional_efflux_ratio", "Tier 3.cell_secretory_permeability", "Tier 3.transporter_inhibition_or_binding", "Tier 3.transporter_substrate_or_efflux")),
        BranchDefinition("Fh.hepatic_clearance_metabolic_stability", "Fh", "hepatic_clearance_metabolic_stability", source_groups=("Tier 5.first_pass_or_extraction", "Tier 5.intrinsic_or_hepatic_clearance", "Tier 5.metabolic_stability")),
    ),
)

STARLING = BranchRetrievalConfig(
    source_name="starling",
    direct_groups=(
        BranchDefinition(DIRECT_ORAL_BIOAVAILABILITY_GROUP, "Observed", "direct_oral_bioavailability", source_groups=(DIRECT_ORAL_BIOAVAILABILITY_GROUP,)),
    ),
    mechanism_groups=tuple(
        BranchDefinition(group_id, group_id.split(".", 1)[0], group_id.split(".", 1)[1], source_groups=(group_id,))
        for group_id in (
            DIRECT_ORAL_BIOAVAILABILITY_GROUP,
            NONDIRECT_ORAL_BIOAVAILABILITY_GROUP,
            "Observed.oral_auc_cmax_exposure",
            "Fa.absorption_solubility_permeability",
            "Fg.gut_wall_efflux_intestinal_metabolism",
            "Fh.hepatic_clearance_metabolic_stability",
        )
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}


def get_source_config(source: str) -> BranchRetrievalConfig:
    return SOURCES[source]


def enrich_retrieval_with_chembl_context(
    retrieval: dict[str, Any],
    index: dict[str, Any],
    *,
    chembl_sqlite: str | Path = DEFAULT_CHEMBL_SQLITE,
    max_exact_bioavailability_rows: int = 30,
    max_shared_per_neighbor: int = 8,
) -> dict[str, Any]:
    return enrich_common_retrieval_with_chembl_context(
        retrieval,
        index,
        relevant_evidence_key="bioavailability_relevant_evidence_rows",
        context_label="Bioavailability_Ma",
        chembl_sqlite=chembl_sqlite,
        max_exact_rows=max_exact_bioavailability_rows,
        max_shared_per_neighbor=max_shared_per_neighbor,
    )


def exact_bioavailability_evidence_rows(
    index: dict[str, Any], molecule_chembl_id: str, *, max_rows: int = 30
) -> list[dict[str, Any]]:
    return exact_evidence_rows(index, molecule_chembl_id, max_rows=max_rows)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/gold_labels/legacy/processed/Bioavailability_Ma/test.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="bioavailability_batch",
    pipeline_module="predict.harnesses.branches.tasks.bioavailability_ma.pipeline",
    log_prefix="bioavailability_reasoning_batch",
    report_title="Bioavailability_Ma Batch Report",
    prediction_field="bioavailability_prediction",
    canonical_positive="high",
    canonical_negative="low",
    positive_predictions=frozenset({"high", "pass", "positive", "bioavailability_positive", "1"}),
    negative_predictions=frozenset({"low", "fail", "negative", "bioavailability_negative", "0"}),
    rerank_preflight=partial(preflight_cache_coverage, task_id="bioavailability_ma"),
    supports_assay_transfer_scores=True,
    supports_retrieval_strategy=True,
    supports_analogous_reasoning_only=True,
    analogous_reasoning_modes=("full_flat", "full_mechanism"),
    final_prompt_provenance=final_prompt_provenance,
    group_prompt_formats=("legacy", "morganfingerprint", "assay_transfer_tool"),
    default_group_prompt_format="legacy",
    group_output_schemas=("legacy", "assay-transfer"),
    default_group_output_schema="legacy",
    group_prompt_versions=GROUP_PROMPT_VERSIONS,
    default_group_prompt_version=DEFAULT_GROUP_PROMPT_VERSION,
    group_prompt_provenance=group_prompt_provenance,
    assay_transfer_profile_default="v9_direct_gold",
    rerank_catalog_default=default_cache_paths("bioavailability_ma")["catalog"],
    rerank_cache_default=default_cache_paths("bioavailability_ma")["cache"],
    rerank_candidate_manifest_default=default_cache_paths("bioavailability_ma")[
        "candidate_manifest"
    ],
    rerank_version_manifest_default=default_cache_paths("bioavailability_ma")["version"],
    assay_transfer_model_default=model_profile("bioavailability_ma")["model"],
    assay_transfer_model_revision_default=model_profile("bioavailability_ma")["revision"],
    assay_transfer_template_profile_default="v9_context_conditioned",
    v9_rerank_catalog_default=default_cache_paths("bioavailability_ma")["catalog"],
    v9_rerank_cache_default=default_cache_paths("bioavailability_ma")["cache"],
    v9_rerank_candidate_manifest_default=default_cache_paths("bioavailability_ma")[
        "candidate_manifest"
    ],
    v9_rerank_version_manifest_default=default_cache_paths("bioavailability_ma")["version"],
    v9_assay_transfer_model_default=model_profile("bioavailability_ma")["model"],
    v9_assay_transfer_model_revision_default=model_profile("bioavailability_ma")["revision"],
    v9_index_default=(
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
        "starling_normalized_v7/08_neighbor_index/scaffold"
    ),
    supports_final_decision_profiles=True,
    prompt_profile_option="--bioavailability-prompt-profile",
    prompt_profile_choices=BIOAVAILABILITY_PROMPT_PROFILES,
    default_prompt_profile=DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    historical_prompt_profile=HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
