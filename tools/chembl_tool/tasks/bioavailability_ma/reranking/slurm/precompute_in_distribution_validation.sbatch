#!/usr/bin/env bash
#SBATCH --job-name=bioavail-id-rerank-v65
#SBATCH --account=myatskar-design-documents
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=112
#SBATCH --mem=896G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/validation_parent_disjoint_r100_c100_min0_v6_5/slurm-%j.out
#SBATCH --error=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/validation_parent_disjoint_r100_c100_min0_v6_5/slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export TXAGENT_RUNTIME_PROFILE=vast_slurm
exec bash "${PROJECT_ROOT}/tools/chembl_tool/tasks/bioavailability_ma/reranking/precompute_in_distribution_validation.sh"
