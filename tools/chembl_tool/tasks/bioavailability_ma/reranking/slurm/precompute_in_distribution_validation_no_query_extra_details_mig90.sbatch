#!/usr/bin/env bash
#SBATCH --job-name=bioavail-v65-noextra
#SBATCH --account=myatskar-design-documents
#SBATCH --partition=b200-mig90
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=448G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/slurm-%j.out
#SBATCH --error=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export TXAGENT_RUNTIME_PROFILE=vast_slurm
exec bash "${PROJECT_ROOT}/tools/chembl_tool/tasks/bioavailability_ma/reranking/precompute_in_distribution_validation_no_query_extra_details.sh"
