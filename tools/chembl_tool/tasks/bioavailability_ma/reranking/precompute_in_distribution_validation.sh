#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_profile.sh"

ARTIFACT_ROOT="outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution"
CONDITION_ROOT="${ARTIFACT_ROOT}/validation_parent_disjoint_r100_c100_min0_v6_5"
EXTRA_ARGS=()
if [[ -n "${TXAGENT_HF_HOME}" ]]; then
  export HF_HOME="${TXAGENT_HF_HOME}"
fi
if [[ "${TXAGENT_LOCAL_FILES_ONLY}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  EXTRA_ARGS+=(--local-files-only)
fi

cd "${TXAGENT_PROJECT_ROOT}"
mkdir -p "${CONDITION_ROOT}"

exec "${TXAGENT_PYTHON}" \
  -m tools.chembl_tool.tasks.bioavailability_ma.reranking.precompute_assay_transfer_rerank \
  --input-jsonl data/processed/Bioavailability_Ma/valid.jsonl \
  --retrieval-source starling_in_distribution \
  --index "${ARTIFACT_ROOT}/starling_in_distribution_neighbor_index.pkl" \
  --experiment-mode full_mechanism \
  --neighbor-identity-policy parent_disjoint \
  --top-k-per-group 10 \
  --min-similarity 0.0 \
  --assay-transfer-initial-morgan-filter 100 \
  --rerank-catalog "${ARTIFACT_ROOT}/starling_in_distribution_catalog.jsonl" \
  --candidate-manifest "${CONDITION_ROOT}/manifest.jsonl" \
  --rerank-cache "${CONDITION_ROOT}/scores.sqlite3" \
  --cache-version-manifest "${CONDITION_ROOT}/VERSION.json" \
  --reuse-prebuilt-catalog \
  --condition-id validation__starling_in_distribution__parent_disjoint__r100_c100_min0__v6_5 \
  --assay-transfer-template-profile v6_5_query_context_copy \
  --assay-transfer-model-revision 9515603b1a5c4586e41c221dcdbc5e7487c0c3f5 \
  --rerank-devices "${TXAGENT_RERANK_DEVICES}" \
  --rerank-batch-size "${TXAGENT_RERANK_BATCH_SIZE}" \
  --rerank-dtype bfloat16 \
  --rerank-min-free-vram-gib 64 \
  "${EXTRA_ARGS[@]}"

