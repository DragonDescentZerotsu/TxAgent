#!/usr/bin/env bash
# Run the frozen in-distribution Bioavailability validation condition against hosted GLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_profile.sh"
cd "${TXAGENT_PROJECT_ROOT}"

: "${TXAGENT_TOOL_SERVICE_PORT:=8766}"
: "${TXAGENT_GLM_PARALLELISM:=32}"
: "${TXAGENT_GLM_GROUP_WORKERS:=5}"
: "${TXAGENT_ASSAY_TRANSFER_TOP_K:=3}"
: "${TXAGENT_ASSAY_TRANSFER_MIN_SCORE:=0.5}"
: "${TXAGENT_GLM_BATCH_ID:=bioavailability_ma__starling_in_distribution_full_mechanism_assay_transfer_tool_scored_k${TXAGENT_ASSAY_TRANSFER_TOP_K}_scoremin050_minsim000_parent_disjoint_val_v65_minimal_evidence_v1}"
: "${TXAGENT_GLM_BATCH_ROOT:=outputs/paper/molecular_evidence_agent/validation_parent_disjoint/bioavailability_ma}"

if [[ -z "${LITELLM_API_KEY:-}" ]]; then
  LITELLM_API_KEY="$("${TXAGENT_REASONING_PYTHON}" -c \
    'from keys import LITELLM_API_KEY; print(LITELLM_API_KEY, end="")')"
  export LITELLM_API_KEY
fi
if [[ -z "${LITELLM_API_KEY}" ]]; then
  echo "LITELLM_API_KEY is unavailable; export it or provide ignored keys.py" >&2
  exit 2
fi

ROOT="outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_in_distribution"
CONDITION="${ROOT}/validation_parent_disjoint_r100_c100_min0_v6_5"
TOOL_URL="http://127.0.0.1:${TXAGENT_TOOL_SERVICE_PORT}"
SERVICE_LOG="${CONDITION}/tool-service-${SLURM_JOB_ID:-local}.log"

TXAGENT_TOOL_SERVICE_PORT="${TXAGENT_TOOL_SERVICE_PORT}" \
  "${TXAGENT_REASONING_PYTHON}" -m uvicorn tools.service.app:app \
  --host 127.0.0.1 --port "${TXAGENT_TOOL_SERVICE_PORT}" \
  >"${SERVICE_LOG}" 2>&1 &
service_pid=$!
cleanup() {
  kill "${service_pid}" 2>/dev/null || true
  wait "${service_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${TXAGENT_REASONING_PYTHON}" - "${TOOL_URL}/health" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
last_error = "service did not respond"
for _ in range(120):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
        if (
            payload.get("status") == "ok"
            and payload.get("tools_initialized") == payload.get("tools_total") == 3
            and not payload.get("tools_with_initialization_errors")
        ):
            print("tool service ready", flush=True)
            break
        last_error = f"unhealthy response: {payload}"
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
    time.sleep(2)
else:
    raise SystemExit(f"tool service health preflight failed: {last_error}")
PY

"${TXAGENT_REASONING_PYTHON}" -m tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch \
  --python-executable "${TXAGENT_REASONING_PYTHON}" \
  --input-jsonl data/processed/Bioavailability_Ma/valid.jsonl \
  --batch-root "${TXAGENT_GLM_BATCH_ROOT}" \
  --batch-id "${TXAGENT_GLM_BATCH_ID}" \
  --index "${ROOT}/starling_in_distribution_neighbor_index.pkl" \
  --retrieval-source starling_in_distribution \
  --experiment-mode full_mechanism \
  --neighbor-identity-policy parent_disjoint \
  --retrieval-reranker assay_transfer \
  --rerank-raw-pool-size 100 \
  --rerank-candidate-size 100 \
  --rerank-catalog "${ROOT}/starling_in_distribution_catalog.jsonl" \
  --rerank-candidate-manifest "${CONDITION}/manifest.jsonl" \
  --rerank-cache "${CONDITION}/scores.sqlite3" \
  --rerank-cache-mode read_only \
  --assay-transfer-model jiosephlee/assay-transfer-tool \
  --assay-transfer-model-revision 9515603b1a5c4586e41c221dcdbc5e7487c0c3f5 \
  --assay-transfer-template-profile v6_5_query_context_copy \
  --rerank-expected-score-count 293021 \
  --rerank-cache-version-manifest "${CONDITION}/VERSION.json" \
  --top-k-per-group "${TXAGENT_ASSAY_TRANSFER_TOP_K}" \
  --min-similarity 0.0 \
  --assay-transfer-min-score "${TXAGENT_ASSAY_TRANSFER_MIN_SCORE}" \
  --enable-assay-transfer-scores \
  --group-prompt-format assay_transfer_tool \
  --api-key-env LITELLM_API_KEY \
  --base-url https://litellm.parcc.upenn.edu/v1 \
  --model zai-org/GLM-5.2-FP8 \
  --disable-thinking \
  --reasoning-effort "" \
  --tool-service-url "${TOOL_URL}" \
  --parallelism "${TXAGENT_GLM_PARALLELISM}" \
  --group-workers "${TXAGENT_GLM_GROUP_WORKERS}" \
  --skip-existing
