#!/usr/bin/env bash
#SBATCH --job-name=bioavail-ignore-glm
#SBATCH --account=myatskar-design-documents
#SBATCH --partition=genoa-std-mem
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=12:00:00
#SBATCH --output=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/ignore-glm-%j.out
#SBATCH --error=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/ignore-glm-%j.out

set -euo pipefail

export TXAGENT_RUNTIME_PROFILE=vast_slurm

TXAGENT_ASSAY_TRANSFER_TOP_K_VALUES=15 \
  bash tools/chembl_tool/tasks/bioavailability_ma/reranking/run_in_distribution_glm_validation_no_query_extra_details.sh

bash tools/chembl_tool/tasks/bioavailability_ma/reranking/run_in_distribution_glm_validation_ignore_structure.sh
