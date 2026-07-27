#!/usr/bin/env bash
#SBATCH --job-name=bioavail-schema-smoke
#SBATCH --account=myatskar-design-documents
#SBATCH --partition=genoa-std-mem
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=01:00:00
#SBATCH --output=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/schema-smoke-%j.out
#SBATCH --error=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/schema-smoke-%j.out

set -euo pipefail

export TXAGENT_RUNTIME_PROFILE=vast_slurm
export TXAGENT_GLM_PARALLELISM=1
export TXAGENT_GLM_GROUP_WORKERS=5
export TXAGENT_GLM_LIMIT=1
export TXAGENT_ASSAY_TRANSFER_TOP_K_VALUES=5
export TXAGENT_RERANK_EXPECTED_SCORE_COUNT=0
export TXAGENT_GLM_BATCH_ROOT=outputs/paper/molecular_evidence_agent/validation_parent_disjoint/bioavailability_ma/smoke_assay_transfer_group_schema

bash tools/chembl_tool/tasks/bioavailability_ma/reranking/run_in_distribution_glm_validation_assay_transfer_schema.sh
