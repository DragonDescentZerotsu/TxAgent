#!/usr/bin/env bash
#SBATCH --job-name=bioavail-schema-glm
#SBATCH --account=myatskar-design-documents
#SBATCH --partition=genoa-std-mem
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=12:00:00
#SBATCH --output=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/schema-glm-%j.out
#SBATCH --error=outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution/v6_5_no_query_extra_details/schema-glm-%j.out

set -euo pipefail

export TXAGENT_RUNTIME_PROFILE=vast_slurm

bash tools/chembl_tool/tasks/bioavailability_ma/reranking/run_in_distribution_glm_validation_assay_transfer_schema.sh
