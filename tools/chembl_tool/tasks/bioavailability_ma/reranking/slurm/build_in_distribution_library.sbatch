#!/usr/bin/env bash
#SBATCH --job-name=bioavail-id-library
#SBATCH --account=myatskar-design-documents
#SBATCH --partition=genoa-lrg-mem
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=15872M
#SBATCH --time=02:00:00
#SBATCH --output=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/build-%j.out
#SBATCH --error=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/build-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export TXAGENT_RUNTIME_PROFILE=vast_slurm
exec bash "${PROJECT_ROOT}/tools/chembl_tool/tasks/bioavailability_ma/reranking/build_in_distribution_library.sh"
